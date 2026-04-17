"""
Method 15: Contrastive ResNet-50 "The Hammer".
Architecture: ResNet-50 + SCL Projection Head.
Input: 3-Channel Spectrograms.
Optimization: Supervised Contrastive Learning (SCL) + Label Smoothing.
Goal: Reach 60% accuracy.

Run: python -m src.audio_baseline.train_resnet_spectrogram_ssl
"""

import os, sys, random, torch, librosa, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from pathlib import Path

project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.audio_baseline import config
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
BATCH_SIZE     = 16
NUM_EPOCHS     = 30
LR             = 5e-5 
SCL_TEMP       = 0.1
SCL_WEIGHT     = 0.3          # 30% Clustering, 70% Classification
LABEL_SMOOTH   = 0.1          # Prevent over-fitting
MAX_DURATION   = 5
SR             = 16000

# ─────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        sim = torch.matmul(features, features.T) / self.temperature
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        bs = labels.size(0)
        diag = torch.ones_like(mask) - torch.eye(bs, device=mask.device)
        mask = mask * diag
        max_s, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - max_s.detach()
        exp_s = torch.exp(sim) * diag
        log_p = sim - torch.log(exp_s.sum(1, keepdim=True) + 1e-8)
        valid = mask.sum(1) > 0
        if valid.any():
            return -(mask[valid] * log_p[valid]).sum(1).div(mask[valid].sum(1) + 1e-8).mean()
        return torch.tensor(0.0, device=features.device, requires_grad=True)

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class VisualSERDataset(Dataset):
    def __init__(self, df, label2id, audio_map, augment=False):
        self.df = df.reset_index(drop=True)
        self.label2id = label2id
        self.audio_map = audio_map
        self.augment = augment

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        basename = Path(str(row["audio_relpath"]).replace("\\", "/")).name
        path = self.audio_map.get(basename)
        try:
            audio, _ = librosa.load(path, sr=SR, mono=True)
        except:
            audio = np.zeros(SR * MAX_DURATION)
        
        target_len = SR * MAX_DURATION
        if len(audio) > target_len: audio = audio[:target_len]
        else: audio = np.pad(audio, (0, target_len - len(audio)))
        
        # 3-Channel: Static, Delta, Delta-Delta
        mel = librosa.feature.melspectrogram(y=audio.astype(np.float32), sr=SR, n_mels=128, n_fft=400, hop_length=160)
        static = librosa.power_to_db(mel, ref=np.max)
        d1 = librosa.feature.delta(static, order=1); d2 = librosa.feature.delta(static, order=2)
        img = np.stack([static, d1, d2], axis=0)
        
        if self.augment:
            if random.random() > 0.5:
                f = random.randint(0, 15); f0 = random.randint(0, 128-f); img[:, f0:f0+f, :] = img.min()
            if random.random() > 0.5:
                t = random.randint(0, 30); t0 = random.randint(0, img.shape[2]-t); img[:, :, t0:t0+t] = img.min()

        mn, mx = img.min(axis=(1, 2), keepdims=True), img.max(axis=(1, 2), keepdims=True)
        img = (img - mn) / (mx - mn + 1e-9)
        return {"image": torch.tensor(img, dtype=torch.float32), "label": torch.tensor(self.label2id[row["emotion_final"]], dtype=torch.long)}

# ─────────────────────────────────────────────
# MODEL (ResNet-50)
# ─────────────────────────────────────────────
class ContrastiveResNet(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        dim_in = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() # Remove default head
        
        # Projection head for SCL (Clustering)
        self.projection_head = nn.Sequential(
            nn.Linear(dim_in, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        
        # Classifier (Prediction)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(dim_in, num_labels)
        )

    def forward(self, x):
        feat = self.backbone(x)
        embeddings = self.projection_head(feat)
        logits = self.classifier(feat)
        return logits, embeddings

def build_model(num_labels, device):
    print("💎 Initializing Contrastive ResNet-50 (Method 15)...")
    model = ContrastiveResNet(num_labels)
    return model.to(device)

def main():
    torch.manual_seed(42); np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}\n🚀 METHOD 12 (IMPROVED): RESNET-50 VISUAL BASELINE\n{'='*70}\n")

    train_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv")
    val_df   = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv")
    test_df  = pd.read_csv(config.SPLIT_CSV_DIR / "test.csv")
    all_df   = pd.concat([train_df, val_df, test_df], ignore_index=True)

    label2id = {l: i for i, l in enumerate(sorted(all_df["emotion_final"].unique()))}
    num_labels = len(label2id)
    audio_map = {}
    src = Path("/content") if Path("/content").exists() else Path(config.DATA_ROOT).parent
    for p in src.rglob("*.wav"): audio_map[p.name] = p

    test_loader = DataLoader(VisualSERDataset(test_df, label2id, audio_map), batch_size=BATCH_SIZE)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X, y = np.arange(len(all_df)), np.array([label2id[l] for l in all_df["emotion_final"]])
    fold_results = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        print(f"\n🌀 FOLD {fold_idx+1}")
        tr_loader = DataLoader(VisualSERDataset(all_df.iloc[tr_idx], label2id, audio_map, augment=True), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        va_loader = DataLoader(VisualSERDataset(all_df.iloc[va_idx], label2id, audio_map), batch_size=BATCH_SIZE, drop_last=True)

        model = build_model(num_labels, device)
        cw = compute_class_weight("balanced", classes=np.unique(y[tr_idx]), y=y[tr_idx])
        ce_fn  = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device), label_smoothing=LABEL_SMOOTH)
        scl_fn = SupervisedContrastiveLoss(SCL_TEMP)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
        
        best_f1 = 0
        ckpt = config.CHECKPOINT_DIR / f"best_resnet50_fold_{fold_idx}.pt"

        for epoch in range(1, NUM_EPOCHS + 1):
            model.train(); t_loss = 0
            for batch in tqdm(tr_loader, desc=f"Ep{epoch}", leave=False):
                imgs, lbs = batch["image"].to(device), batch["label"].to(device)
                optimizer.zero_grad()
                logits, feat = model(imgs)
                
                loss_ce  = ce_fn(logits, lbs)
                loss_scl = scl_fn(feat, lbs)
                loss     = (1 - SCL_WEIGHT) * loss_ce + SCL_WEIGHT * loss_scl
                
                loss.backward()
                optimizer.step()
                t_loss += loss.item()
            
            scheduler.step()

            model.eval(); preds, truth = [], []
            with torch.no_grad():
                for b in va_loader:
                    logits, _ = model(b["image"].to(device))
                    preds.extend(torch.argmax(logits, 1).cpu().numpy())
                    truth.extend(b["label"].numpy())
            va_acc, va_f1 = accuracy_score(truth, preds), f1_score(truth, preds, average="macro")
            
            # Eval Test
            tp, tt = [], []
            with torch.no_grad():
                for b in test_loader:
                    ls, _ = model(b["image"].to(device))
                    tp.extend(torch.argmax(ls, 1).cpu().numpy()); tt.extend(b["label"].numpy())
            te_acc, te_f1 = accuracy_score(tt, tp), f1_score(tt, tp, average="macro")

            print(f"   Ep {epoch:2d}/{NUM_EPOCHS} | Loss {t_loss/len(tr_loader):.3f} | Val {va_acc:.3f}/{va_f1:.3f} | Test {te_acc:.3f}/{te_f1:.3f}")
            if va_f1 > best_f1: best_f1 = va_f1; best_acc = va_acc; torch.save(model.state_dict(), ckpt)
        
        fold_results.append({"acc": best_acc, "f1": best_f1})
        print(f"📊 FOLD {fold_idx+1} Result: Acc {best_acc*100:.1f}% | F1 {best_f1*100:.1f}%")

    print(f"\n🏆 FINAL PEAK PERFORMANCE: Mean Acc {np.mean([r['acc'] for r in fold_results])*100:.2f}%")

if __name__ == "__main__":
    main()
