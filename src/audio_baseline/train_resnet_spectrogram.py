"""
Method 9: Visual Reset Baseline.
Architecture: ResNet-18 (Pre-trained) + Log-Mel Spectrograms.
Goal: Stable 35-40% baseline on small Egyptian SER data.

Run: python -m src.audio_baseline.train_resnet_spectrogram
"""

import os, sys, random, torch, librosa, numpy as np, pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
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
NUM_EPOCHS = 20
LR = 1e-4
MAX_DURATION = 5  # 5 seconds
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
        
        # 1. Load Audio
        try:
            audio, _ = librosa.load(path, sr=SR, mono=True)
        except:
            audio = np.zeros(SR * MAX_DURATION)
            
        # 2. Pad/Truncate
        target_len = SR * MAX_DURATION
        if len(audio) > target_len: audio = audio[:target_len]
        else: audio = np.pad(audio, (0, target_len - len(audio)))
        
        # 3. Generate Mel Spectrogram
        # Standard settings: 128 bins, 25ms window, 10ms hop
        mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=128, n_fft=400, hop_length=160)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # 4. Normalize to [0, 1] for CNN
        mel_scaled = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-9)
        
        # 5. Convert to 3-channel "image" for ResNet
        # Image shape: [3, 128, 501]
        img = np.stack([mel_scaled, mel_scaled, mel_scaled], axis=0)
        
        return {
            "image": torch.tensor(img, dtype=torch.float32),
            "label": torch.tensor(self.label2id[row["emotion_final"]], dtype=torch.long)
        }

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def build_resnet_model(num_labels):
    model = models.resnet18(pretrained=True)
    # Replace the final layer
    model.fc = nn.Linear(model.fc.in_features, num_labels)
    return model

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    torch.manual_seed(config.SEED); np.random.seed(config.SEED); random.seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print("🌅  METHOD 9: VISUAL RESET (RESNET-18 + SPECTROGRAMS)")
    print(f"{'='*70}\n")

    # 1. Load Data
    train_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv")
    val_df   = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv")
    test_df  = pd.read_csv(config.SPLIT_CSV_DIR / "test.csv")
    all_df   = pd.concat([train_df, val_df, test_df], ignore_index=True)

    label2id = {l: i for i, l in enumerate(sorted(all_df["emotion_final"].unique()))}
    id2label = {i: l for l, i in label2id.items()}
    num_labels = len(label2id)
    
    audio_map = {}
    src = Path("/content") if Path("/content").exists() else Path(config.DATA_ROOT).parent
    for p in src.rglob("*.wav"): audio_map[p.name] = p

    test_loader = DataLoader(VisualSERDataset(test_df, label2id, audio_map), batch_size=BATCH_SIZE)

    # 2. Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    X, y = np.arange(len(all_df)), np.array([label2id[l] for l in all_df["emotion_final"]])
    
    fold_results = []
    
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        print(f"\n🌀 FOLD {fold_idx+1}")
        tr_loader = DataLoader(VisualSERDataset(all_df.iloc[tr_idx], label2id, audio_map, augment=True), batch_size=BATCH_SIZE, shuffle=True)
        va_loader = DataLoader(VisualSERDataset(all_df.iloc[va_idx], label2id, audio_map), batch_size=BATCH_SIZE)

        model = build_resnet_model(num_labels).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        
        best_f1, best_acc = 0.0, 0.0
        
        for epoch in range(1, NUM_EPOCHS + 1):
            model.train(); train_loss = 0
            for batch in tqdm(tr_loader, desc=f"Ep{epoch}", leave=False):
                imgs, lbs = batch["image"].to(device), batch["label"].to(device)
                optimizer.zero_grad()
                out = model(imgs)
                loss = criterion(out, lbs)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Eval
            model.eval(); preds, truth = [], []
            with torch.no_grad():
                for batch in va_loader:
                    out = model(batch["image"].to(device))
                    preds.extend(torch.argmax(out, 1).cpu().numpy())
                    truth.extend(batch["label"].numpy())
            
            va_acc = accuracy_score(truth, preds)
            va_f1 = f1_score(truth, preds, average="macro")
            print(f"   Ep {epoch}/{NUM_EPOCHS} | Loss {train_loss/len(tr_loader):.4f} | Val Acc {va_acc:.3f} | Val F1 {va_f1:.3f}")
            
            if va_f1 > best_f1:
                best_f1 = va_f1; best_acc = va_acc
        
        fold_results.append({"acc": best_acc, "f1": best_f1})
        print(f"📊 FOLD {fold_idx+1} Best Result: Acc {best_acc*100:.1f}% | F1 {best_f1*100:.1f}%")

    print(f"\nFinal: Mean Val Acc {np.mean([r['acc'] for r in fold_results])*100:.1f}% | Mean Val F1 {np.mean([r['f1'] for r in fold_results])*100:.1f}%")

if __name__ == "__main__":
    main()
