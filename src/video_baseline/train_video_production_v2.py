import os
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.optim.swa_utils import AveragedModel, SWALR
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────────────────
# CONFIG & SEEDING (PRODUCTION BASELINE V2 - MULTI-SCALE)
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

EPOCHS = 25 
BATCH_SIZE = 32
LR = 7e-5
WEIGHT_DECAY = 5e-2
D_MODEL = 512
ENSEMBLE_SEEDS = [42, 1337, 2024, 777, 999]

class Lookahead:
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k, self.alpha = k, alpha
        self.param_groups = self.optimizer.param_groups
        self.slow_weights = [[p.data.clone().detach() for p in group['params']] for group in self.param_groups]
        self.counter = 0
    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self.counter += 1
        if self.counter % self.k == 0:
            for i, group in enumerate(self.param_groups):
                for j, p in enumerate(group['params']):
                    p.data.mul_(self.alpha).add_(self.slow_weights[i][j], alpha=1.0 - self.alpha)
                    self.slow_weights[i][j].copy_(p.data)
        return loss
    def zero_grad(self, set_to_none=True): self.optimizer.zero_grad(set_to_none=set_to_none)

class SEBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(c, c // 16), nn.ReLU(), nn.Linear(c // 16, c), nn.Sigmoid())
    def forward(self, x):
        b, n, c = x.size()
        y = self.avg_pool(x.transpose(1, 2)).view(b, c)
        y = self.fc(y).view(b, 1, c)
        return x * y

class MSWModel(nn.Module):
    def __init__(self, d_model=D_MODEL, dropout=0.5):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(3584, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.se = SEBlock(d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 16, d_model))
        layer = nn.TransformerEncoderLayer(d_model, 8, d_model*2, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, 2)
        self.attn = nn.Linear(d_model, 1)
        self.scale_fusion = nn.Linear(d_model * 3, d_model)
        self.classifier = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 7))
        self.scl_head = nn.Sequential(nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, 128))

    def forward(self, x):
        if self.training: x = x + torch.randn_like(x) * 0.01
        x = self.proj(x)
        x = self.se(x)
        x = x + self.pos_embed
        x = self.transformer(x)
        wide = x
        focused = x[:, 4:12, :]
        core = x[:, 6:10, :]
        def pool(feat):
            w = F.softmax(self.attn(feat), 1)
            return torch.sum(feat * w, 1)
        fused = self.scale_fusion(torch.cat([pool(wide), pool(focused), pool(core)], -1))
        return self.classifier(fused), F.normalize(self.scl_head(fused), 1)

def margin_scl_loss(features, labels, temperature=0.07):
    m = torch.zeros(7, device=features.device)
    m[LID['Happiness']], m[LID['Surprise']], m[LID['Fear']] = 0.2, 0.15, 0.25
    sim = torch.matmul(features, features.T) / temperature
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
    sim = sim - (mask * m[labels].unsqueeze(1))
    lmask = torch.scatter(torch.ones_like(mask), 1, torch.arange(labels.size(0), device=features.device).view(-1, 1), 0)
    return -(mask * lmask * (sim - torch.log(torch.exp(sim).sum(1, keepdim=True) + 1e-6))).sum(1).mean() / (mask.sum(1).mean() + 1e-6)

class VideoDataset(Dataset):
    def __init__(self, df, feat_dir): self.df = df.reset_index(drop=True); self.feat_dir = feat_dir
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
    df = pd.read_csv(str(root / "video_manifest_trackA.csv"))
    def feat_exists(sid): return (feat_dir / f"{sid.replace('::','__').replace('/','_').replace('.mp4','')}_clip_seq.npy").exists()
    df = df[(df['resolution_status'] == 'resolved') & (df['sample_id'].apply(feat_exists))]
    tr_df, va_df, te_df = df[df['split'] == 'train'], df[df['split'] == 'val'], df[df['split'] == 'test']
    tr_loader = DataLoader(VideoDataset(tr_df, feat_dir), batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(VideoDataset(va_df, feat_dir), batch_size=BATCH_SIZE, shuffle=False)
    te_loader = DataLoader(VideoDataset(te_df, feat_dir), batch_size=BATCH_SIZE, shuffle=False)
    ensemble_probs, ensemble_weights = [], []
    print(f"\n🚀 Running PRODUCTION BASELINE V2 (Multi-Scale Window)")
    for i, seed in enumerate(ENSEMBLE_SEEDS):
        print(f"\n--- MODEL {i+1}/5 (SEED {seed}) ---")
        set_reproducibility(seed)
        model = MSWModel().to(device)
        base_opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        optimizer = Lookahead(base_opt)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(base_opt, max_lr=LR*1.2, steps_per_epoch=len(tr_loader), epochs=EPOCHS)
        swa_model = AveragedModel(model)
        best_f1, model_path = 0, root / "models" / f"production_v2_{seed}.pt"
        os.makedirs(root / "models", exist_ok=True)
        for epoch in range(1, EPOCHS + 1):
            model.train()
            for x, y in tr_loader:
                x, y = x.to(device), y.to(device); optimizer.zero_grad()
                logits, projs = model(x); scl = margin_scl_loss(projs, y)
                ce = F.cross_entropy(logits, y, label_smoothing=0.1)
                (ce + 0.4*scl).backward(); optimizer.step()
                if epoch > 15: swa_model.update_parameters(model)
                scheduler.step()
            model.eval(); preds, targets = [], []
            with torch.no_grad():
                for vx, vy in va_loader:
                    vl, _ = model(vx.to(device))
                    preds.extend(vl.argmax(1).cpu().numpy()); targets.extend(vy.numpy())
            v_f1 = f1_score(targets, preds, average='macro', zero_division=0)
            if v_f1 > best_f1: best_f1 = v_f1; torch.save(model.state_dict(), str(model_path))
            print(f"Epoch {epoch:02d} | Val F1: {v_f1:.4f}")
        model.load_state_dict(torch.load(str(model_path), weights_only=True))
        t_probs = []
        with torch.no_grad():
            for tx, ty in te_loader:
                tl, _ = model(tx.to(device)); t_probs.append(F.softmax(tl, 1).cpu().numpy())
        ensemble_probs.append(np.vstack(t_probs)); ensemble_weights.append(best_f1)
    w = np.array(ensemble_weights); w = w / w.sum()
    final_preds = np.argmax(sum(p * weight for p, weight in zip(ensemble_probs, w)), 1)
    t_targets = te_df['emotion_final'].apply(lambda x: LID[x]).values
    print("\n" + "="*40 + "\nPRODUCTION V2 EVALUATION\n" + "="*40)
    print(f"Test Accuracy: {accuracy_score(t_targets, final_preds):.4f} | Test Macro F1: {f1_score(t_targets, final_preds, average='macro'):.4f}")

if __name__ == "__main__": main()
