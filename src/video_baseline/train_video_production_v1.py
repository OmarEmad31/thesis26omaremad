import os
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, balanced_accuracy_score
from torch.optim.swa_utils import AveragedModel, SWALR
from pathlib import Path

# ─────────────────────────────────────────────────────────
# CONFIG & SEEDING (PRODUCTION BASELINE V1)
# ─────────────────────────────────────────────────────────
LID = {'Anger':0, 'Disgust':1, 'Fear':2, 'Happiness':3, 'Neutral':4, 'Sadness':5, 'Surprise':6}
REV_LID = {v: k for k, v in LID.items()}

def set_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

EPOCHS = 50 
BATCH_SIZE = 32
LR = 8e-5
WEIGHT_DECAY = 5e-2
D_MODEL = 512
ENSEMBLE_SEEDS = [42, 1337, 2024, 777, 999]

class DualAttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)
        self.proj = nn.Linear(d_model * 2, d_model)
    def forward(self, x):
        weights = F.softmax(self.attn(x), dim=1)
        avg_p = torch.sum(x * weights, dim=1)
        max_p, _ = torch.max(x, dim=1)
        return self.proj(torch.cat([avg_p, max_p], dim=-1))

class StabilizedDMCModel(nn.Module):
    def __init__(self, input_dim=3584, d_model=D_MODEL, dropout=0.5):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.pos_embed = nn.Parameter(torch.randn(1, 16, d_model))
        layer = nn.TransformerEncoderLayer(d_model, 8, d_model*2, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.pooling = DualAttentionPooling(d_model)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 7)
        )
        self.scl_head = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, 128)
        )

    def forward(self, x, lam=None, idx=None):
        if self.training:
            if random.random() > 0.8:
                i = random.randint(0, 14)
                x[:, [i, i+1]] = x[:, [i+1, i]]
            x = x + torch.randn_like(x) * 0.01
        x = self.proj(x) + self.pos_embed
        if lam is not None and self.training:
            x = lam * x + (1 - lam) * x[idx]
        x = self.transformer(x)
        pooled = self.pooling(x)
        logits = self.classifier(pooled)
        projs = F.normalize(self.scl_head(pooled), dim=1)
        return logits, projs

def margin_scl_loss(features, labels, temperature=0.07):
    margins = torch.zeros(7).to(features.device)
    margins[LID['Happiness']], margins[LID['Surprise']], margins[LID['Fear']] = 0.2, 0.15, 0.25
    sim = torch.matmul(features, features.T) / temperature
    labels = labels.unsqueeze(1)
    mask = torch.eq(labels, labels.T).float().to(features.device)
    batch_m = margins[labels].expand(-1, labels.size(0))
    sim = sim - (mask * batch_m)
    logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(labels.shape[0]).view(-1, 1).to(features.device), 0)
    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-6)
    return -(mask * logits_mask * log_prob).sum(1).mean() / (mask.sum(1).mean() + 1e-6)

class VideoDataset(Dataset):
    def __init__(self, df, feat_dir):
        self.df = df.reset_index(drop=True)
        self.feat_dir = feat_dir
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fid = row['sample_id'].replace("::", "__").replace("/", "_").replace(".mp4", "")
        c, d, r = [np.load(str(self.feat_dir / f"{fid}_{m}_seq.npy")) for m in ['clip', 'dinov2', 'resnet50']]
        return torch.tensor(np.concatenate([c, d, r], -1), dtype=torch.float32), torch.tensor(LID[row['emotion_final']], dtype=torch.long)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(r"d:\Thesis Project")
    feat_dir = root / "data" / "processed" / "features" / "video_sequences_v1"
    manifest_path = root / "video_manifest_trackA.csv"
    df = pd.read_csv(str(manifest_path))
    def feat_exists(sid):
        fid = sid.replace("::", "__").replace("/", "_").replace(".mp4", "")
        return (feat_dir / f"{fid}_clip_seq.npy").exists()
    df = df[(df['resolution_status'] == 'resolved') & (df['sample_id'].apply(feat_exists))]
    tr_df, va_df, te_df = df[df['split'] == 'train'], df[df['split'] == 'val'], df[df['split'] == 'test']
    tr_loader = DataLoader(VideoDataset(tr_df, feat_dir), batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(VideoDataset(va_df, feat_dir), batch_size=BATCH_SIZE, shuffle=False)
    te_loader = DataLoader(VideoDataset(te_df, feat_dir), batch_size=BATCH_SIZE, shuffle=False)
    
    ensemble_probs = []
    print(f"\n🚀 Running PRODUCTION BASELINE V1 (Stabilized DMC-SWA)")
    
    for i, seed in enumerate(ENSEMBLE_SEEDS):
        print(f"\n--- MODEL {i+1}/5 (SEED {seed}) ---")
        set_reproducibility(seed)
        model = StabilizedDMCModel().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR*1.2, steps_per_epoch=len(tr_loader), epochs=EPOCHS)
        swa_model = AveragedModel(model)
        swa_start, swa_scheduler = 15, SWALR(optimizer, swa_lr=LR/10)
        best_f1, model_path = 0, root / "models" / f"production_v1_{seed}.pt"
        os.makedirs(root / "models", exist_ok=True)
        for epoch in range(1, EPOCHS + 1):
            model.train()
            for x, y in tr_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                lam = np.random.beta(0.2, 0.2) if random.random() > 0.6 else None
                idx = torch.randperm(x.size(0)).to(device) if lam else None
                logits, projs = model(x, lam, idx)
                scl_loss = margin_scl_loss(projs, y)
                if lam: ce_loss = lam * F.cross_entropy(logits, y) + (1-lam)*F.cross_entropy(logits, y[idx])
                else: ce_loss = F.cross_entropy(logits, y, label_smoothing=0.1)
                (ce_loss + 0.4 * scl_loss).backward()
                optimizer.step()
                if epoch > swa_start: swa_model.update_parameters(model); swa_scheduler.step()
                else: scheduler.step()
            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for vx, vy in va_loader:
                    v_logits, _ = model(vx.to(device))
                    preds.extend(torch.argmax(v_logits, 1).cpu().numpy()); targets.extend(vy.numpy())
            v_f1 = f1_score(targets, preds, average='macro', zero_division=0)
            if v_f1 > best_f1: best_f1 = v_f1; torch.save(model.state_dict(), str(model_path))
            print(f"Epoch {epoch:02d} | Val F1: {v_f1:.4f}")
        model.load_state_dict(torch.load(str(model_path), weights_only=True))
        t_probs = []
        with torch.no_grad():
            for tx, ty in te_loader:
                t_logits, _ = model(tx.to(device))
                t_probs.append(F.softmax(t_logits, 1).cpu().numpy())
        ensemble_probs.append(np.vstack(t_probs))

    avg_probs = np.mean(ensemble_probs, axis=0)
    final_preds = np.argmax(avg_probs, 1)
    final_preds[avg_probs[:, LID['Surprise']] > 0.18] = LID['Surprise']
    final_preds[avg_probs[:, LID['Happiness']] > 0.22] = LID['Happiness']
    t_targets = te_df['emotion_final'].apply(lambda x: LID[x]).values
    print("\n" + "="*40 + "\nPRODUCTION BASELINE EVALUATION\n" + "="*40)
    print(f"Test Accuracy: {accuracy_score(t_targets, final_preds):.4f} | Test Macro F1: {f1_score(t_targets, final_preds, average='macro'):.4f}")

if __name__ == "__main__":
    main()
