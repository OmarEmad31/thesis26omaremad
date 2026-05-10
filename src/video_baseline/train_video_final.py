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

# ─────────────────────────────────────────────────────────
# CONFIG & SEEDING
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

EPOCHS = 35 
BATCH_SIZE = 32
LR = 9e-5
WEIGHT_DECAY = 1e-1 # Very high for small data
D_MODEL = 256        # Reduced complexity to fight overfitting
ENSEMBLE_SEEDS = [42, 1337, 2024, 777, 999]

# ─────────────────────────────────────────────────────────
# ESSENTIALIST ARCHITECTURE (PYRAMID + ORTHOGONAL)
# ─────────────────────────────────────────────────────────
class TemporalPyramidPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)
    def forward(self, x):
        # x: [B, 16, D]
        # Level 1: Global
        w1 = F.softmax(self.attn(x), 1)
        p1 = torch.sum(x * w1, 1)
        # Level 2: 2x8
        p2 = torch.cat([torch.mean(x[:, :8, :], 1), torch.mean(x[:, 8:, :], 1)], -1)
        return p1, p2

class EssentialistModel(nn.Module):
    def __init__(self, d_model=D_MODEL, dropout=0.5):
        super().__init__()
        self.proj_c = nn.Linear(768, d_model)
        self.proj_d = nn.Linear(768, d_model)
        self.proj_r = nn.Linear(2048, d_model)
        
        self.pos_embed = nn.Parameter(torch.randn(1, 16, d_model))
        layer = nn.TransformerEncoderLayer(d_model, 4, d_model*2, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, 2)
        
        self.tpp = TemporalPyramidPooling(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, d_model), # P1 (D) + P2 (2*D)
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 7)
        )
        self.scl_head = nn.Sequential(nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, 128))

    def forward(self, c, d, r):
        # 1. Projection
        c, d, r = self.proj_c(c), self.proj_d(d), self.proj_r(r)
        
        # Modality Dropout (Essential for generalization)
        if self.training:
            m_drop = random.random()
            if m_drop < 0.1: c = torch.zeros_like(c)
            elif m_drop < 0.2: d = torch.zeros_like(d)
            elif m_drop < 0.3: r = torch.zeros_like(r)
            
        x = (c + d + r) / 3.0 + self.pos_embed
        x = self.transformer(x)
        
        p1, p2 = self.tpp(x)
        fused = torch.cat([p1, p2], -1)
        
        logits = self.classifier(fused)
        return logits, F.normalize(self.scl_head(p1), 1), (c, d, r)

# ─────────────────────────────────────────────────────────
# LOSS & DATA
# ─────────────────────────────────────────────────────────
def orthogonal_loss(c, d, r):
    # Flatten temporal dimension for orthogonality check
    c_f, d_f, r_f = c.mean(1), d.mean(1), r.mean(1)
    sim_cd = torch.abs(F.cosine_similarity(c_f, d_f)).mean()
    sim_dr = torch.abs(F.cosine_similarity(d_f, r_f)).mean()
    sim_rc = torch.abs(F.cosine_similarity(r_f, c_f)).mean()
    return (sim_cd + sim_dr + sim_rc) / 3.0

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
        return torch.tensor(c, dtype=torch.float32), torch.tensor(d, dtype=torch.float32), torch.tensor(r, dtype=torch.float32), torch.tensor(LID[row['emotion_final']], dtype=torch.long)

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
    print(f"\n🚀 Starting TWEAK-v12: THE ESSENTIALIST (PYRAMID + ORTHOGONAL)")
    
    for i, seed in enumerate(ENSEMBLE_SEEDS):
        print(f"\n--- MODEL {i+1}/5 (SEED {seed}) ---")
        set_reproducibility(seed)
        model = EssentialistModel().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR*1.2, steps_per_epoch=len(tr_loader), epochs=EPOCHS)
        
        best_f1, model_path = 0, root / "models" / f"essential_{seed}.pt"
        os.makedirs(root / "models", exist_ok=True)
        
        for epoch in range(1, EPOCHS + 1):
            model.train(); total_loss = 0
            for c, d, r, y in tr_loader:
                c, d, r, y = c.to(device), d.to(device), r.to(device), y.to(device)
                optimizer.zero_grad()
                logits, projs, feats = model(c, d, r)
                
                scl = margin_scl_loss(projs, y)
                ce = F.cross_entropy(logits, y, label_smoothing=0.1)
                ortho = orthogonal_loss(*feats)
                
                loss = ce + 0.3*scl + 0.2*ortho
                loss.backward(); optimizer.step(); scheduler.step()
                total_loss += loss.item()
            
            model.eval(); preds, targets = [], []
            with torch.no_grad():
                for vc, vd, vr, vy in va_loader:
                    vl, _, _ = model(vc.to(device), vd.to(device), vr.to(device))
                    preds.extend(vl.argmax(1).cpu().numpy()); targets.extend(vy.numpy())
            v_f1 = f1_score(targets, preds, average='macro', zero_division=0)
            if v_f1 > best_f1: best_f1 = v_f1; torch.save(model.state_dict(), str(model_path))
            print(f"Epoch {epoch:02d} | Val F1: {v_f1:.4f}")

        model.load_state_dict(torch.load(str(model_path), weights_only=True))
        t_probs = []
        with torch.no_grad():
            for tc, td, tr, ty in te_loader:
                tl, _, _ = model(tc.to(device), td.to(device), tr.to(device))
                t_probs.append(F.softmax(tl, 1).cpu().numpy())
        ensemble_probs.append(np.vstack(t_probs)); ensemble_weights.append(best_f1)

    w = np.array(ensemble_weights); w = w / w.sum()
    weighted_probs = sum(p * weight for p, weight in zip(ensemble_probs, w))
    final_preds = weighted_probs.argmax(1)
    final_preds[weighted_probs[:, LID['Surprise']] > 0.18] = LID['Surprise']
    final_preds[weighted_probs[:, LID['Happiness']] > 0.22] = LID['Happiness']
    
    t_targets = te_df['emotion_final'].apply(lambda x: LID[x]).values
    print("\n" + "="*40 + "\nESSENTIALIST FINAL EVALUATION\n" + "="*40)
    print(f"Test Accuracy: {accuracy_score(t_targets, final_preds):.4f} | Test Macro F1: {f1_score(t_targets, final_preds, average='macro'):.4f}")
    print(classification_report(t_targets, final_preds, target_names=list(LID.keys()), zero_division=0))

if __name__ == "__main__":
    main()
