import os
import random
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────────────────
# CONFIGURATION & HYPERPARAMETERS
# ─────────────────────────────────────────────────────────
LID = {'Anger':0, 'Disgust':1, 'Fear':2, 'Happiness':3, 'Neutral':4, 'Sadness':5, 'Surprise':6}
REV_LID = {v: k for k, v in LID.items()}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ─────────────────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────
class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        weights = F.softmax(self.attn(x), dim=1)
        return torch.sum(x * weights, dim=1)

class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, d_model=512, nhead=8, num_layers=2, dim_feedforward=2048, dropout=0.3, pooling_type='mean'):
        super().__init__()
        
        # 1. Projection
        self.projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # 2. Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 16, d_model))
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # CLS Token if needed
        self.pooling_type = pooling_type
        if pooling_type == 'cls':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            # Adjust pos_embed for cls token if needed, or just add to seq
            self.pos_embed = nn.Parameter(torch.zeros(1, 17, d_model))
        elif pooling_type == 'attn':
            self.attn_pool = AttentionPooling(d_model)
            
        # 4. Classifier Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.3),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        # x: [batch, 16, input_dim]
        x = self.projection(x)
        
        if self.pooling_type == 'cls':
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed
        else:
            x = x + self.pos_embed
            
        x = self.transformer(x)
        
        if self.pooling_type == 'cls':
            pooled = x[:, 0]
        elif self.pooling_type == 'attn':
            pooled = self.attn_pool(x)
        else: # mean
            pooled = x.mean(dim=1)
            
        logits = self.classifier(pooled)
        return logits

# ─────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────
class VideoSequenceDataset(Dataset):
    def __init__(self, df, feat_dir, features_to_use):
        self.df = df.reset_index(drop=True)
        self.feat_dir = feat_dir
        self.features_to_use = features_to_use
        
    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = row['sample_id']
        fid = sid.replace("::", "__").replace("/", "_").replace(".mp4", "")
        
        combined_feat = []
        for suffix, _ in self.features_to_use:
            fpath = self.feat_dir / f"{fid}{suffix}"
            feat = np.load(fpath) # [16, D]
            combined_feat.append(feat)
            
        final_feat = np.concatenate(combined_feat, axis=-1)
        return torch.tensor(final_feat, dtype=torch.float32), torch.tensor(LID[row['emotion_final']], dtype=torch.long), sid

# ─────────────────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()

def get_criterion(exp_type, weights=None):
    if exp_type == 1:
        return nn.CrossEntropyLoss(weight=weights)
    elif exp_type == 2:
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    elif exp_type == 3:
        return FocalLoss(gamma=2.0, weight=weights)
    return nn.CrossEntropyLoss(weight=weights)

# ─────────────────────────────────────────────────────────
# EVALUATION & ANALYSIS
# ─────────────────────────────────────────────────────────
def evaluate(model, loader, device, criterion=None):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    total_loss = 0
    sids = []
    
    with torch.no_grad():
        for x, y, sid in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if criterion:
                total_loss += criterion(logits, y).item()
            
            probs = F.softmax(logits, dim=1)
            all_preds.extend(torch.argmax(logits, 1).cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            sids.extend(sid)
            
    metrics = {
        'acc': accuracy_score(all_targets, all_preds),
        'macro_f1': f1_score(all_targets, all_preds, average='macro'),
        'uar': balanced_accuracy_score(all_targets, all_preds),
        'weighted_f1': f1_score(all_targets, all_preds, average='weighted'),
        'loss': total_loss / len(loader) if criterion else 0
    }
    return metrics, all_preds, all_targets, np.array(all_probs), sids

def analyze_happiness(targets, preds, probs, sids):
    happ_idx = LID['Happiness']
    targets = np.array(targets)
    preds = np.array(preds)
    
    mask = (targets == happ_idx)
    happ_targets = targets[mask]
    happ_preds = preds[mask]
    happ_probs = probs[mask]
    happ_sids = np.array(sids)[mask]
    
    correct = (happ_preds == happ_idx).sum()
    total = len(happ_targets)
    
    print(f"\n[Happiness Analysis] Correct: {correct}/{total}")
    if total > 0:
        conf_correct = happ_probs[happ_preds == happ_idx, happ_idx].mean() if correct > 0 else 0
        print(f"  Avg confidence for correct Happiness: {conf_correct:.4f}")
        
        confusion = defaultdict(int)
        for p in happ_preds[happ_preds != happ_idx]:
            confusion[REV_LID[p]] += 1
        print(f"  Confused with: {dict(confusion)}")

# ─────────────────────────────────────────────────────────
# MAIN EXPERIMENT RUNNER
# ─────────────────────────────────────────────────────────
def run_experiment(exp_name, features_to_use, loss_exp_type, pooling_type='mean'):
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(r"d:\Thesis Project")
    feat_dir = root / "data" / "processed" / "features" / "video_sequences_v1"
    manifest_path = root / "video_manifest_trackA.csv"
    
    df = pd.read_csv(manifest_path)
    
    def all_features_exist(sid):
        fid = sid.replace("::", "__").replace("/", "_").replace(".mp4", "")
        for suffix, _ in features_to_use:
            if not (feat_dir / f"{fid}{suffix}").exists(): return False
        return True
    
    df['exists'] = df['sample_id'].apply(all_features_exist)
    df = df[(df['resolution_status'] == 'resolved') & (df['exists'] == True)]
    
    tr_df = df[df['split'] == 'train']
    va_df = df[df['split'] == 'val']
    te_df = df[df['split'] == 'test'].reset_index(drop=True)
    
    if len(tr_df) == 0 or len(va_df) == 0 or len(te_df) == 0:
        print(f"Skipping {exp_name}: Data is incomplete (Train={len(tr_df)}, Val={len(va_df)}, Test={len(te_df)}).")
        return None
    
    print(f"\n>>> Running Experiment: {exp_name} | Loss Type: {loss_exp_type} | Pooling: {pooling_type}")
    print(f"    Data: Train={len(tr_df)}, Val={len(va_df)}, Test={len(te_df)}")
    
    tr_ds = VideoSequenceDataset(tr_df, feat_dir, features_to_use)
    va_ds = VideoSequenceDataset(va_df, feat_dir, features_to_use)
    te_ds = VideoSequenceDataset(te_df, feat_dir, features_to_use)
    
    tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=32, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=32, shuffle=False)
    
    input_dim = sum(dim for _, dim in features_to_use)
    model = TemporalTransformer(input_dim=input_dim, pooling_type=pooling_type).to(device)
    
    # Class weights
    tr_labels = [LID[e] for e in tr_df['emotion_final']]
    counts = pd.Series(tr_labels).value_counts().sort_index()
    weights = torch.tensor([1.0/counts.get(i, 1) for i in range(7)], dtype=torch.float32).to(device)
    weights = weights / weights.sum() * 7.0
    
    criterion = get_criterion(loss_exp_type, weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    best_val_f1 = 0
    patience = 8
    counter = 0
    save_path = root / f"temp_best_{exp_name}.pt"
    
    for epoch in range(1, 51):
        model.train()
        for x, y, _ in tr_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        val_metrics, _, _, _, _ = evaluate(model, va_loader, device, criterion)
        print(f"  Epoch {epoch:02d} | Loss: {val_metrics['loss']:.4f} | Val F1: {val_metrics['macro_f1']:.4f}")
        
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            torch.save(model.state_dict(), save_path)
            print(f"    *** New Best Model Saved (F1: {best_val_f1:.4f})")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
                
    # Final Eval
    model.load_state_dict(torch.load(save_path, weights_only=True))
    test_metrics, p_test, t_test, probs_test, sids_test = evaluate(model, te_loader, device)
    
    print(f"Results: Test Acc: {test_metrics['acc']:.4f} | Macro F1: {test_metrics['macro_f1']:.4f} | UAR: {test_metrics['uar']:.4f}")
    
    return {
        'exp_name': exp_name,
        'pooling': pooling_type,
        'loss_type': loss_exp_type,
        'test_acc': test_metrics['acc'],
        'test_f1': test_metrics['macro_f1'],
        'test_uar': test_metrics['uar'],
        'targets': t_test,
        'preds': p_test,
        'probs': probs_test,
        'sids': sids_test
    }

if __name__ == "__main__":
    configs = [
        ("CLIP_Only", [("_clip_seq.npy", 768)]),
        ("DINO_Only", [("_dinov2_seq.npy", 768)]),
        ("CLIP_DINO", [("_clip_seq.npy", 768), ("_dinov2_seq.npy", 768)]),
        ("FUSED_ALL", [("_clip_seq.npy", 768), ("_dinov2_seq.npy", 768), ("_resnet50_seq.npy", 2048)])
    ]
    
    all_results = []
    
    # We will test Experiment 1 (Weighted CE) with 'mean' pooling first as a baseline transformer
    for name, feats in configs:
        res = run_experiment(name, feats, loss_exp_type=1, pooling_type='mean')
        if res:
            # Remove keys that aren't for the summary
            summary_row = {k: v for k, v in res.items() if k not in ['targets', 'preds', 'probs', 'sids']}
            all_results.append(summary_row)
            analyze_happiness(res['targets'], res['preds'], res['probs'], res['sids'])
            
            # Save individual confusion matrix
            cm = confusion_matrix(res['targets'], res['preds'])
            cm_df = pd.DataFrame(cm, index=list(LID.keys()), columns=list(LID.keys()))
            cm_df.to_csv(f"transformer_cm_{name}.csv")

            # Intermediate save
            summary = pd.DataFrame(all_results)
            summary.to_csv("temporal_transformer_results.csv", index=False)

    print("\nAll experiments completed. Final summary saved to temporal_transformer_results.csv")
