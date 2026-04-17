"""
Method 12: Final Precision Supervised Fine-tuning.
Strategy: Load SSL-pretrained "Egyptian" weights + Weighted CrossEntropy + 3-Channel Spectrograms.
Goal: Hit the 58-60% target.

Run: python -m src.audio_baseline.train_resnet_spectrogram_ssl
"""

import os, sys, random, torch, librosa, numpy as np, pandas as pd
import torch.nn as nn
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
# CONFIG
# ─────────────────────────────────────────────
BATCH_SIZE = 32
NUM_EPOCHS = 25
LR = 5e-5      # Lower LR because we have powerful SSL weights
MAX_DURATION = 5
SR = 16000

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
        d1 = librosa.feature.delta(static, order=1)
        d2 = librosa.feature.delta(static, order=2)
        img = np.stack([static, d1, d2], axis=0)
        
        if self.augment:
            if random.random() > 0.5:
                f = random.randint(0, 10); f0 = random.randint(0, 128-f); img[:, f0:f0+f, :] = img.min()
            if random.random() > 0.5:
                t = random.randint(0, 20); t0 = random.randint(0, img.shape[2]-t); img[:, :, t0:t0+t] = img.min()

        img_min = img.min(axis=(1, 2), keepdims=True); img_max = img.max(axis=(1, 2), keepdims=True)
        img = (img - img_min) / (img_max - img_min + 1e-9)
        return {"image": torch.tensor(img, dtype=torch.float32), "label": torch.tensor(self.label2id[row["emotion_final"]], dtype=torch.long)}

# ─────────────────────────────────────────────
# MODEL (Load SSL Weights)
# ─────────────────────────────────────────────
def build_ssl_tuned_model(num_labels, device):
    model = models.resnet18(pretrained=False) # Load our own pre-training
    ssl_path = config.CHECKPOINT_DIR / "egyptian_ssl_resnet18.pt"
    
    if ssl_path.exists():
        print(f"💎 Loading Egyptian-Native SSL Weights from {ssl_path}...")
        state_dict = torch.load(ssl_path, map_location=device)
        # Handle the simclr projection head naming
        model.load_state_dict(state_dict, strict=False)
    else:
        print("⚠️ SSL Weights not found! Using standard ImageNet.")
        model = models.resnet18(pretrained=True)
        
    model.fc = nn.Linear(model.fc.in_features, num_labels)
    return model.to(device)

def main():
    torch.manual_seed(42); np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}\n🚀 METHOD 12: FINAL PRECISION (SSL-TUNED) RUN\n{'='*70}\n")

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
        tr_loader = DataLoader(VisualSERDataset(all_df.iloc[tr_idx], label2id, audio_map, augment=True), batch_size=BATCH_SIZE, shuffle=True)
        va_loader = DataLoader(VisualSERDataset(all_df.iloc[va_idx], label2id, audio_map), batch_size=BATCH_SIZE)

        model = build_ssl_tuned_model(num_labels, device)
        # 🧪 FIX: Weighted Loss to boost F1
        cw = compute_class_weight("balanced", classes=np.unique(y[tr_idx]), y=y[tr_idx])
        # 🧬 Precision Tuning: 
        # 1. Differential Learning Rates (Protect SSL weights, tune the head)
        optimizer = torch.optim.AdamW([
            {"params": [p for n, p in model.named_parameters() if "fc" not in n], "lr": LR / 10},
            {"params": model.fc.parameters(), "lr": LR},
        ], weight_decay=1e-4)

        # 2. Cosine Annealing Scheduler (Smooth descent)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        # 3. Label Smoothing (Prevents overconfidence)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device), label_smoothing=0.1)
        
        best_f1, best_acc, no_improve = 0, 0, 0
        ckpt = config.CHECKPOINT_DIR / f"best_ssl_fold_{fold_idx}.pt"

        for epoch in range(1, NUM_EPOCHS + 1):
            model.train(); t_loss = 0
            for batch in tqdm(tr_loader, desc=f"Ep{epoch}", leave=False):
                imgs, lbs = batch["image"].to(device), batch["label"].to(device)
                optimizer.zero_grad(); out = model(imgs); loss = criterion(out, lbs); loss.backward(); optimizer.step()
                t_loss += loss.item()
            
            scheduler.step() # Advance scheduler

            model.eval(); preds, truth = [], []
            with torch.no_grad():
                for b in va_loader:
                    out = model(b["image"].to(device)); preds.extend(torch.argmax(out, 1).cpu().numpy()); truth.extend(b["label"].numpy())
            
            va_acc, va_f1 = accuracy_score(truth, preds), f1_score(truth, preds, average="macro")
            
            # Eval Test
            tp, tt = [], []
            with torch.no_grad():
                for b in test_loader:
                    o = model(b["image"].to(device)); tp.extend(torch.argmax(o, 1).cpu().numpy()); tt.extend(b["label"].numpy())
            te_acc, te_f1 = accuracy_score(tt, tp), f1_score(tt, tp, average="macro")

            print(f"   Ep {epoch:2d}/{NUM_EPOCHS} | Val {va_acc:.3f}/{va_f1:.3f} | Test {te_acc:.3f}/{te_f1:.3f} | LR: {optimizer.param_groups[1]['lr']:.2e}")
            
            if va_f1 > best_f1: 
                best_f1 = va_f1; best_acc = va_acc; torch.save(model.state_dict(), ckpt)
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= 8: # Early Stopping
                print(f"   🛑 Early stopping at epoch {epoch}")
                break
        
        fold_results.append({"acc": best_acc, "f1": best_f1})
        print(f"📊 FOLD {fold_idx+1} Result: Acc {best_acc*100:.1f}% | F1 {best_f1*100:.1f}%")

    print(f"\n🏆 FINAL SSL-ENHANCED PERFORMANCE: Mean Acc {np.mean([r['acc'] for r in fold_results])*100:.2f}%")

if __name__ == "__main__":
    main()
