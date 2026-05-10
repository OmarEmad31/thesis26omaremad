import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.metrics import accuracy_score, f1_score, classification_report, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────
EPOCHS = 35
BATCH_SIZE = 64
K_PER_CLASS = 4  # For balanced batching
LR = 5e-4
WEIGHT_DECAY = 0.01
SUPCON_TEMP = 0.1
SUPCON_WEIGHT = 0.3

# Toggle which features to fuse. 
# Once extraction finishes, uncomment DINOv2 and ResNet50!
FEATURES_TO_USE = [
    ("_clip.npy", 1536), 
    ("_dinov2.npy", 1536),
    ("_resnet50.npy", 4096)
]
TOTAL_INPUT_DIM = sum(dim for _, dim in FEATURES_TO_USE)

LID = {'Anger':0, 'Disgust':1, 'Fear':2, 'Happiness':3, 'Neutral':4, 'Sadness':5, 'Surprise':6}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ─────────────────────────────────────────────────────────
# DATASET & SAMPLER
# ─────────────────────────────────────────────────────────
class VideoFeatureDataset(Dataset):
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
            feat = np.load(fpath)
            combined_feat.append(feat)
            
        final_feat = np.concatenate(combined_feat)
        return torch.tensor(final_feat, dtype=torch.float32), torch.tensor(LID[row['emotion_final']], dtype=torch.long)

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, k=K_PER_CLASS):
        self.k = k
        self.class_indices = defaultdict(list)
        for i, lbl in enumerate(labels):
            self.class_indices[int(lbl)].append(i)
        self.classes = sorted(list(self.class_indices.keys()))
        self.n_batches = max(1, min(len(v) for v in self.class_indices.values()) // k)

    def __iter__(self):
        pools = {c: random.sample(idxs, len(idxs)) for c, idxs in self.class_indices.items()}
        ptrs = {c: 0 for c in self.classes}
        for _ in range(self.n_batches):
            batch = []
            for c in self.classes:
                if ptrs[c] + self.k > len(pools[c]):
                    pools[c] = random.sample(self.class_indices[c], len(self.class_indices[c]))
                    ptrs[c] = 0
                batch.extend(pools[c][ptrs[c] : ptrs[c] + self.k])
                ptrs[c] += self.k
            random.shuffle(batch)
            yield batch

    def __len__(self): return self.n_batches

# ─────────────────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temp = temperature
        
    def forward(self, features, labels):
        device = features.device
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(device)
        mask *= (1.0 - torch.eye(labels.shape[0], device=device))
        
        sim = torch.matmul(features, features.T) / self.temp
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach()
        
        valid = mask.sum(1) > 0
        if not valid.any(): return torch.tensor(0.0).to(device)
        
        log_prob = sim - torch.log(torch.exp(sim).sum(1, keepdim=True) + 1e-8)
        mean_log_prob_pos = (mask[valid] * log_prob[valid]).sum(1) / (mask[valid].sum(1))
        return -mean_log_prob_pos.mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()

# ─────────────────────────────────────────────────────────
# MODEL (MLP + Multi-Sample Dropout)
# ─────────────────────────────────────────────────────────
class MultiSampleDropout(nn.Module):
    def __init__(self, fc, num_samples=5, p=0.4):
        super().__init__()
        self.fc = fc
        self.num_samples = num_samples
        self.dropouts = nn.ModuleList([nn.Dropout(p) for _ in range(num_samples)])

    def forward(self, x):
        return torch.mean(torch.stack([self.fc(drop(x)) for drop in self.dropouts], dim=0), dim=0)

class VideoMLP(nn.Module):
    def __init__(self, input_dim=1536, hidden_dim=512, num_classes=7):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.GELU()
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.act2 = nn.GELU()
        
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128),
            nn.LayerNorm(128)
        )
        
        self.classifier = MultiSampleDropout(nn.Linear(hidden_dim // 2, num_classes), num_samples=5, p=0.4)

    def forward(self, x):
        h = self.act1(self.bn1(self.fc1(x)))
        h = self.act2(self.bn2(self.fc2(h)))
        
        proj = F.normalize(self.proj_head(h), p=2, dim=1)
        logits = self.classifier(h)
        return logits, proj

# ─────────────────────────────────────────────────────────
# TRAINING ENGINE (5-FOLD CV)
# ─────────────────────────────────────────────────────────
def main():
    set_seed(42)
    device = "cpu"
    print(f"Using device: {device}", flush=True)
    
    root = Path(r"d:\Thesis Project")
    manifest_path = root / "video_manifest_trackA.csv"
    feat_dir = root / "data" / "processed" / "features" / "video_v2"
    
    df = pd.read_csv(manifest_path)
    
    # Check that ALL requested feature files exist for a given sample
    def all_features_exist(sid):
        fid = sid.replace("::", "__").replace("/", "_").replace(".mp4", "")
        for suffix, _ in FEATURES_TO_USE:
            if not (feat_dir / f"{fid}{suffix}").exists(): return False
        return True
        
    df['exists'] = df['sample_id'].apply(all_features_exist)
    df = df[(df['resolution_status'] == 'resolved') & (df['exists'] == True)]
    
    # 5-Fold pooling (Train + Val)
    tr_df = df[df['split'] == 'train']
    va_df = df[df['split'] == 'val']
    te_df = df[df['split'] == 'test'].reset_index(drop=True)
    
    pool_df = pd.concat([tr_df, va_df]).reset_index(drop=True)
    
    print(f"Loaded Features: Pool (Train+Val)={len(pool_df)} | Test={len(te_df)}", flush=True)
    
    pool_labels = np.array([LID[e] for e in pool_df['emotion_final']])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    te_ds = VideoFeatureDataset(te_df, feat_dir, FEATURES_TO_USE)
    te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model_paths = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(pool_labels)), pool_labels)):
        print("\n" + "="*50, flush=True)
        print(f"FOLD {fold+1}/5 - TRAINING VIDEO SOTA", flush=True)
        print("="*50, flush=True)
        
        fold_tr_df = pool_df.iloc[train_idx].reset_index(drop=True)
        fold_va_df = pool_df.iloc[val_idx].reset_index(drop=True)
        
        tr_ds = VideoFeatureDataset(fold_tr_df, feat_dir, FEATURES_TO_USE)
        va_ds = VideoFeatureDataset(fold_va_df, feat_dir, FEATURES_TO_USE)
        
        tr_labels_fold = [LID[e] for e in fold_tr_df['emotion_final']]
        bal_sampler = BalancedBatchSampler(tr_labels_fold, k=K_PER_CLASS)
        
        tr_loader = DataLoader(tr_ds, batch_sampler=bal_sampler, num_workers=0)
        va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        counts = pd.Series(tr_labels_fold).value_counts().sort_index()
        weights = torch.tensor([1.0/counts.get(i, 1) for i in range(7)], dtype=torch.float32).to(device)
        weights = weights / weights.sum() * 7.0
        
        model = VideoMLP(input_dim=TOTAL_INPUT_DIM).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * len(tr_loader))
        
        focal_loss = FocalLoss(gamma=2.0, weight=weights)
        supcon_loss = SupConLoss(temperature=SUPCON_TEMP)
        
        best_f1 = 0
        save_path = root / f"best_video_model_fold_{fold}.pt"
        model_paths.append(save_path)
        
        for epoch in range(1, EPOCHS + 1):
            model.train()
            ep_loss = 0
            for x, y in tr_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                
                logits, proj = model(x)
                ce = focal_loss(logits, y)
                scl = supcon_loss(proj, y)
                loss = (1 - SUPCON_WEIGHT) * ce + (SUPCON_WEIGHT * scl)
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                ep_loss += loss.item()
                
            model.eval()
            p_val, t_val = [], []
            with torch.no_grad():
                for x, y in va_loader:
                    x = x.to(device)
                    logits, _ = model(x)
                    p_val.extend(torch.argmax(logits, 1).cpu().numpy())
                    t_val.extend(y.numpy())
                    
            val_acc = accuracy_score(t_val, p_val)
            val_f1 = f1_score(t_val, p_val, average='macro')
            
            star = ""
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), save_path)
                star = "BEST"
                
            # Only print every 5 epochs to avoid massive logs, except when best
            if epoch % 5 == 0 or star == "BEST":
                print(f"Fold {fold+1} | Epoch {epoch:02d} | Loss: {ep_loss/len(tr_loader):.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} {star}", flush=True)

    print("\n" + "="*50, flush=True)
    print("FINAL TEST SET EVALUATION (5-FOLD SOFT-VOTING ENSEMBLE)", flush=True)
    print("="*50, flush=True)
    
    ensemble_probs = []
    t_test = []
    
    for path in model_paths:
        model = VideoMLP(input_dim=TOTAL_INPUT_DIM).to(device)
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        
        probs = []
        t_test_fold = []
        with torch.no_grad():
            for x, y in te_loader:
                x = x.to(device)
                logits, _ = model(x)
                probs.append(F.softmax(logits, dim=-1).cpu().numpy())
                t_test_fold.extend(y.numpy())
        
        ensemble_probs.append(np.vstack(probs))
        if len(t_test) == 0:
            t_test = t_test_fold
            
    avg_probs = np.mean(ensemble_probs, axis=0)
    p_test = np.argmax(avg_probs, axis=1)
            
    print(f"Final Test Accuracy:   {accuracy_score(t_test, p_test):.4f}")
    print(f"Final Test UAR:        {balanced_accuracy_score(t_test, p_test):.4f}")
    print(f"Final Test Macro F1:   {f1_score(t_test, p_test, average='macro'):.4f}")
    print(f"Final Test Weighted F1:{f1_score(t_test, p_test, average='weighted'):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(t_test, p_test, target_names=list(LID.keys()), zero_division=0))

if __name__ == "__main__":
    main()
