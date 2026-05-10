import os
import random
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict, Counter
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────────────────
# CONFIGURATION & HYPERPARAMETERS
# ─────────────────────────────────────────────────────────
LID = {'Anger':0, 'Disgust':1, 'Fear':2, 'Happiness':3, 'Neutral':4, 'Sadness':5, 'Surprise':6}
REV_LID = {v: k for k, v in LID.items()}

# VERIFICATION FLAGS
DEBUG_OVERFIT = False
DEBUG_SHUFFLE_LABELS = False

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def log_verification(msg, mode='a'):
    print(msg)
    with open("verification_log.txt", mode, encoding="utf-8") as f:
        f.write(msg + "\n")

# ─────────────────────────────────────────────────────────
# MODEL COMPONENTS
# ─────────────────────────────────────────────────────────
class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)

    def forward(self, x):
        weights = F.softmax(self.attn(x), dim=1)
        return torch.sum(x * weights, dim=1)

class AdvancedProjection(nn.Module):
    def __init__(self, input_dim, d_model, layers=1, dropout=0.2):
        super().__init__()
        if layers == 1:
            self.proj = nn.Sequential(
                nn.Linear(input_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        else:
            mid_dim = max(input_dim // 2, d_model * 2)
            self.proj = nn.Sequential(
                nn.Linear(input_dim, mid_dim),
                nn.LayerNorm(mid_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mid_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            
    def forward(self, x):
        return self.proj(x)

class TemporalTransformerV2(nn.Module):
    def __init__(self, input_dim, d_model=512, nhead=8, num_layers=2, 
                 dropout=0.3, pooling_type='mean', proj_layers=1):
        super().__init__()
        self.projection = AdvancedProjection(input_dim, d_model, layers=proj_layers, dropout=0.2)
        self.pos_embed = nn.Parameter(torch.zeros(1, 17, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.pooling_type = pooling_type
        if pooling_type == 'cls':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        elif pooling_type == 'attn':
            self.attn_pool = AttentionPooling(d_model)
            
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        x = self.projection(x)
        if self.pooling_type == 'cls':
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed[:, :x.size(1), :]
        else:
            x = x + self.pos_embed[:, 1:x.size(1)+1, :]
            
        x = self.transformer(x)
        if self.pooling_type == 'cls':
            pooled = x[:, 0]
        elif self.pooling_type == 'attn':
            pooled = self.attn_pool(x)
        else: # mean
            pooled = x.mean(dim=1)
        return self.classifier(pooled)

# ─────────────────────────────────────────────────────────
# DATASET & LOSS
# ─────────────────────────────────────────────────────────
class VideoSequenceDataset(Dataset):
    def __init__(self, df, feat_dir, features_to_use, shuffle_labels=False):
        self.df = df.reset_index(drop=True)
        self.feat_dir = feat_dir
        self.features_to_use = features_to_use
        
        if shuffle_labels:
            log_verification("DEBUG SHUFFLE LABELS ENABLED")
            labels = self.df['emotion_final'].tolist()
            random.shuffle(labels)
            self.df['emotion_final'] = labels
            
    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = row['sample_id']
        fid = sid.replace("::", "__").replace("/", "_").replace(".mp4", "")
        combined_feat = []
        for suffix, _ in self.features_to_use:
            fpath = self.feat_dir / f"{fid}{suffix}"
            feat = np.load(fpath)
            combined_feat.append(feat)
        final_feat = np.concatenate(combined_feat, axis=-1)
        return torch.tensor(final_feat, dtype=torch.float32), torch.tensor(LID[row['emotion_final']], dtype=torch.long), sid

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()

def get_criterion(loss_type, weights=None):
    if loss_type == 'ce_smooth':
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    elif loss_type == 'focal':
        return FocalLoss(gamma=2.0, weight=weights)
    return nn.CrossEntropyLoss(weight=weights)

# ─────────────────────────────────────────────────────────
# EVALUATION & TUNING
# ─────────────────────────────────────────────────────────
def evaluate(model, loader, device, criterion=None):
    model.eval()
    all_preds, all_targets, all_probs, sids = [], [], [], []
    total_loss = 0
    with torch.no_grad():
        for x, y, sid in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if criterion: total_loss += criterion(logits, y).item()
            probs = F.softmax(logits, dim=1)
            all_preds.extend(torch.argmax(logits, 1).cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            sids.extend(sid)
            
    # VERIFICATION 4: PREDICTION DISTRIBUTION CHECK
    if len(all_preds) > 0:
        log_verification("\n--- Evaluation Metrics ---")
        log_verification(f"Prediction counts: {dict(Counter(all_preds))}")
        log_verification(f"True label counts: {dict(Counter(all_targets))}")
        
    # VERIFICATION 5: CONFIDENCE CHECK
    if len(all_probs) >= 5:
        log_verification(f"Softmax (first 5):\n{all_probs[:5]}")

    metrics = {
        'acc': accuracy_score(all_targets, all_preds),
        'macro_f1': f1_score(all_targets, all_preds, average='macro'),
        'uar': balanced_accuracy_score(all_targets, all_preds),
        'loss': total_loss / len(loader) if criterion else 0
    }
    return metrics, np.array(all_preds), np.array(all_targets), np.array(all_probs), sids

def tune_happiness_threshold(targets, probs):
    best_threshold, best_f1 = 0.0, 0.0
    happ_idx = LID['Happiness']
    base_preds = np.argmax(probs, axis=1)
    for thresh in np.linspace(0, 0.8, 81):
        temp_preds = base_preds.copy()
        temp_preds[probs[:, happ_idx] > thresh] = happ_idx
        f1 = f1_score(targets, temp_preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    return best_threshold, best_f1

# ─────────────────────────────────────────────────────────
# EXPERIMENT RUNNER
# ─────────────────────────────────────────────────────────
def run_experiment(config):
    set_seed(42)
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(r"d:\Thesis Project")
    feat_dir = root / "data" / "processed" / "features" / "video_sequences_v1"
    manifest_path = root / "video_manifest_trackA.csv"
    
    log_verification(f"\n" + "="*60, mode='w' if config['name'] == 'Baseline_V2' else 'a')
    log_verification(f"RUNNING EXPERIMENT: {config['name']}")
    log_verification("="*60)
    
    df = pd.read_csv(manifest_path)
    features_to_use = config['features']
    def feat_exists(sid):
        fid = sid.replace("::", "__").replace("/", "_").replace(".mp4", "")
        return all((feat_dir / f"{fid}{s}").exists() for s, _ in features_to_use)
    
    df = df[df['resolution_status'] == 'resolved']
    df = df[df['sample_id'].apply(feat_exists)]
    
    tr_df = df[df['split'] == 'train']
    va_df = df[df['split'] == 'val']
    te_df = df[df['split'] == 'test'].reset_index(drop=True)
    
    # VERIFICATION 6: OVERFIT DEBUG MODE
    if DEBUG_OVERFIT:
        log_verification("DEBUG OVERFIT MODE ENABLED")
        tr_df = tr_df.head(20)
        
    # VERIFICATION 1: DATASET SANITY CHECKS
    tr_ids = set(tr_df['sample_id'])
    va_ids = set(va_df['sample_id'])
    te_ids = set(te_df['sample_id'])
    
    log_verification(f"Train size: {len(tr_df)}")
    log_verification(f"Val size: {len(va_df)}")
    log_verification(f"Test size: {len(te_df)}")
    log_verification(f"Unique train IDs: {len(tr_ids)}")
    log_verification(f"Unique val IDs: {len(va_ids)}")
    log_verification(f"Unique test IDs: {len(te_ids)}")
    log_verification(f"Overlap (train ∩ val): {len(tr_ids.intersection(va_ids))}")
    log_verification(f"Overlap (train ∩ test): {len(tr_ids.intersection(te_ids))}")
    log_verification(f"Overlap (val ∩ test): {len(va_ids.intersection(te_ids))}")
    
    tr_ds = VideoSequenceDataset(tr_df, feat_dir, features_to_use, shuffle_labels=DEBUG_SHUFFLE_LABELS)
    va_ds = VideoSequenceDataset(va_df, feat_dir, features_to_use)
    te_ds = VideoSequenceDataset(te_df, feat_dir, features_to_use)
    
    # VERIFICATION 2: DATALOADER CHECKS
    BATCH_SIZE = 32
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    log_verification(f"Batches per epoch: {len(tr_loader)}")
    log_verification(f"Batch size: {BATCH_SIZE}")
    
    input_dim = sum(dim for _, dim in features_to_use)
    model = TemporalTransformerV2(
        input_dim=input_dim, num_layers=config['num_layers'],
        pooling_type=config['pooling'], proj_layers=config['proj_layers'],
        dropout=config['dropout']
    ).to(device)
    
    tr_labels = [LID[e] for e in tr_df['emotion_final']]
    counts = pd.Series(tr_labels).value_counts().sort_index()
    weights = torch.tensor([1.0/counts.get(i, 1) for i in range(7)], dtype=torch.float32).to(device)
    if config.get('boost_happiness', False): weights[LID['Happiness']] *= 1.5
    weights = weights / weights.sum() * 7.0
    
    criterion = get_criterion(config['loss_type'], weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=config['weight_decay'])
    
    best_val_f1 = 0
    patience = 10
    counter = 0
    save_path = root / "temp_v2.pt"
    
    for epoch in range(1, 51):
        ep_start = time.time()
        model.train()
        for batch_idx, (x, y, sid) in enumerate(tr_loader):
            # VERIFICATION 8: FEATURE SHAPE VALIDATION
            if x.shape[0] == BATCH_SIZE: # Only check full batches
                assert x.shape == (BATCH_SIZE, 16, input_dim), f"Shape mismatch: {x.shape}"
            
            # VERIFICATION 3: BATCH CONTENT CHECK (FIRST BATCH ONLY)
            if epoch == 1 and batch_idx == 0:
                log_verification("\n--- Batch 0 Check ---")
                log_verification(f"Feature shape: {x.shape}")
                log_verification(f"First 5 labels: {y[:5].tolist()}")
                log_verification(f"First 5 sample_ids: {sid[:5]}")
                log_verification(f"Feature Mean: {x.mean().item():.4f}, Std: {x.std().item():.4f}")
                
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        val_metrics, _, _, _, _ = evaluate(model, va_loader, device, criterion)
        
        # VERIFICATION 9: TIMING CHECK
        ep_time = time.time() - ep_start
        if epoch % 5 == 0:
            log_verification(f"Epoch {epoch:02d} | Val F1: {val_metrics['macro_f1']:.4f} | Time: {ep_time:.2f}s")

        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            torch.save(model.state_dict(), save_path)
            counter = 0
        else:
            counter += 1
            if counter >= patience: break
            
    # Final Eval
    model.load_state_dict(torch.load(save_path, weights_only=True))
    v_met, v_pred, v_true, v_prob, v_sids = evaluate(model, va_loader, device)
    best_thresh, tuned_val_f1 = tune_happiness_threshold(v_true, v_prob)
    t_met, t_pred, t_true, t_prob, t_sids = evaluate(model, te_loader, device)
    
    final_t_pred = t_pred.copy()
    final_t_pred[t_prob[:, LID['Happiness']] > best_thresh] = LID['Happiness']
    
    test_f1 = f1_score(t_true, final_t_pred, average='macro', zero_division=0)
    test_acc = accuracy_score(t_true, final_t_pred)
    test_uar = balanced_accuracy_score(t_true, final_t_pred)
    
    total_time = time.time() - start_time
    log_verification(f"\nCOMPLETED: {config['name']}")
    log_verification(f"Total time: {total_time:.2f}s")
    log_verification(f"Test Acc: {test_acc:.4f} | Macro F1: {test_f1:.4f} | UAR: {test_uar:.4f}")
    
    # VERIFICATION 10: FINAL SUMMARY
    summary_msg = f"""
    --- VERIFICATION SUMMARY ---
    Experiment: {config['name']}
    Train/Val/Test: {len(tr_df)} / {len(va_df)} / {len(te_df)}
    Batches: {len(tr_loader)}
    Feature Shape: (16, {input_dim})
    Debug Overfit: {DEBUG_OVERFIT}
    Debug Shuffle: {DEBUG_SHUFFLE_LABELS}
    """
    log_verification(summary_msg)
    
    return {
        'config': config, 'val_f1': tuned_val_f1, 'test_f1': test_f1,
        'test_acc': test_acc, 'test_uar': test_uar, 'thresh': best_thresh,
        'preds': final_t_pred, 'targets': t_true, 'probs': t_prob, 'sids': t_sids
    }

if __name__ == "__main__":
    fusion_feats = [("_clip_seq.npy", 768), ("_dinov2_seq.npy", 768)]
    ablation_configs = [
        {'name': 'Baseline_V2', 'features': fusion_feats, 'num_layers': 2, 'pooling': 'mean', 'proj_layers': 1, 'loss_type': 'ce', 'dropout': 0.3, 'weight_decay': 1e-4},
        {'name': 'CLS_Pooling', 'features': fusion_feats, 'num_layers': 2, 'pooling': 'cls', 'proj_layers': 1, 'loss_type': 'ce', 'dropout': 0.3, 'weight_decay': 1e-4},
        {'name': 'Attn_Pooling', 'features': fusion_feats, 'num_layers': 2, 'pooling': 'attn', 'proj_layers': 1, 'loss_type': 'ce', 'dropout': 0.3, 'weight_decay': 1e-4},
        {'name': 'Deep_Proj', 'features': fusion_feats, 'num_layers': 2, 'pooling': 'mean', 'proj_layers': 2, 'loss_type': 'ce', 'dropout': 0.3, 'weight_decay': 1e-4},
        {'name': 'Depth_1', 'features': fusion_feats, 'num_layers': 1, 'pooling': 'mean', 'proj_layers': 1, 'loss_type': 'ce', 'dropout': 0.3, 'weight_decay': 1e-4},
        {'name': 'Depth_3', 'features': fusion_feats, 'num_layers': 3, 'pooling': 'mean', 'proj_layers': 1, 'loss_type': 'ce', 'dropout': 0.3, 'weight_decay': 1e-4},
        {'name': 'CE_Smoothing', 'features': fusion_feats, 'num_layers': 2, 'pooling': 'mean', 'proj_layers': 1, 'loss_type': 'ce_smooth', 'dropout': 0.3, 'weight_decay': 1e-4},
        {'name': 'Focal_Loss_Boost', 'features': fusion_feats, 'num_layers': 2, 'pooling': 'mean', 'proj_layers': 1, 'loss_type': 'focal', 'dropout': 0.3, 'weight_decay': 1e-4, 'boost_happiness': True},
        {'name': 'Reg_Drop0.4', 'features': fusion_feats, 'num_layers': 2, 'pooling': 'mean', 'proj_layers': 1, 'loss_type': 'ce', 'dropout': 0.4, 'weight_decay': 1e-4},
        {'name': 'Reg_WD5e-4', 'features': fusion_feats, 'num_layers': 2, 'pooling': 'mean', 'proj_layers': 1, 'loss_type': 'ce', 'dropout': 0.3, 'weight_decay': 5e-4},
    ]
    
    results = []
    for cfg in ablation_configs:
        res = run_experiment(cfg)
        results.append(res)
        
    summary_df = pd.DataFrame([{
        'name': r['config']['name'], 'val_f1': r['val_f1'], 'test_f1': r['test_f1'],
        'test_acc': r['test_acc'], 'test_uar': r['test_uar'], 'thresh': r['thresh']
    } for r in results])
    summary_df.to_csv("improved_temporal_transformer_results.csv", index=False)
    
    best_res = max(results, key=lambda x: x['val_f1'])
    cm = confusion_matrix(best_res['targets'], best_res['preds'])
    pd.DataFrame(cm, index=list(LID.keys()), columns=list(LID.keys())).to_csv("confusion_matrix_best.csv")
    pd.DataFrame({
        'sample_id': best_res['sids'],
        'target': [REV_LID[t] for t in best_res['targets']],
        'pred': [REV_LID[p] for p in best_res['preds']]
    }).to_csv("predictions_test.csv", index=False)
