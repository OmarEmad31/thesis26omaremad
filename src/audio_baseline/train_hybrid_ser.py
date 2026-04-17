"""
Method 13: Hybrid Fusion (High-Capacity).
Architecture: ResNet-18 (Visual) + Wav2Vec2-Arabic (Phonetic).
Fusion: Feature Concatenation + Multi-Layer MLP.
Goal: Final push to break 58-60%.

Run: python -m src.audio_baseline.train_hybrid_ser
"""

import os, sys, random, torch, librosa, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from transformers import AutoFeatureExtractor, Wav2Vec2Model
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
W2V_MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
BATCH_SIZE   = 2            # Keep very low to prevent OOM
ACCUM_STEPS  = 16           # Effective batch 32
NUM_EPOCHS   = 20
LR           = 1e-5
MAX_WAV_LEN  = 5 * 16000     # 5 seconds
SR           = 16000

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class HybridSERDataset(Dataset):
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
            audio = np.zeros(MAX_WAV_LEN)
            
        # Raw Audio for Wav2Vec2
        if len(audio) > MAX_WAV_LEN: audio = audio[:MAX_WAV_LEN]
        else: audio = np.pad(audio, (0, MAX_WAV_LEN - len(audio)))
        raw_audio = torch.tensor(audio, dtype=torch.float32)

        # 3-Channel Spectrogram for ResNet
        mel = librosa.feature.melspectrogram(y=audio.astype(np.float32), sr=SR, n_mels=128, n_fft=400, hop_length=160)
        static = librosa.power_to_db(mel, ref=np.max)
        d1 = librosa.feature.delta(static, order=1); d2 = librosa.feature.delta(static, order=2)
        img = np.stack([static, d1, d2], axis=0)
        
        # Normalize
        mn, mx = img.min(axis=(1,2), keepdims=True), img.max(axis=(1,2), keepdims=True)
        img = (img - mn) / (mx - mn + 1e-9)
        
        return {
            "raw_audio": raw_audio,
            "image": torch.tensor(img, dtype=torch.float32),
            "label": torch.tensor(self.label2id[row["emotion_final"]], dtype=torch.long)
        }

# ─────────────────────────────────────────────
# HYBRID MODEL
# ─────────────────────────────────────────────
class HybridSERModel(nn.Module):
    def __init__(self, num_labels, device):
        super().__init__()
        # Branch A: Vision
        self.resnet = models.resnet18(pretrained=True)
        self.resnet_dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity() # Remove classifier

        # Branch B: Speech
        print(f"Loading {W2V_MODEL_ID}...")
        self.w2v = Wav2Vec2Model.from_pretrained(W2V_MODEL_ID)
        self.w2v_dim = self.w2v.config.hidden_size # Usually 1024

        # Fusion Head
        combined_dim = self.resnet_dim + self.w2v_dim
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_labels)
        )
        
        # Freeze early layers of Wav2Vec2 to save memory / prevent corruption
        for p in self.w2v.feature_extractor.parameters(): p.requires_grad = False
        for p in self.w2v.encoder.layers[:12].parameters(): p.requires_grad = False

    def forward(self, image, raw_audio):
        # 1. Vision Forward
        feat_vis = self.resnet(image) # [B, 512]
        
        # 2. Speech Forward
        w2v_out = self.w2v(raw_audio).last_hidden_state
        feat_seq = torch.mean(w2v_out, dim=1) # Global Mean Pool [B, 1024]
        
        # 3. Concatenate
        combined = torch.cat([feat_vis, feat_seq], dim=1)
        
        # 4. Final Classification
        return self.fusion(combined)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    torch.manual_seed(42); np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}\n🚀 METHOD 13: HYBRID FUSION (RESNET + WAV2VEC2)\n{'='*70}\n")

    train_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv")
    val_df   = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv")
    test_df  = pd.read_csv(config.SPLIT_CSV_DIR / "test.csv")
    all_df   = pd.concat([train_df, val_df, test_df], ignore_index=True)

    label2id = {l: i for i, l in enumerate(sorted(all_df["emotion_final"].unique()))}
    num_labels = len(label2id)
    audio_map = {}
    src = Path("/content") if Path("/content").exists() else Path(config.DATA_ROOT).parent
    for p in src.rglob("*.wav"): audio_map[p.name] = p

    test_loader = DataLoader(HybridSERDataset(test_df, label2id, audio_map), batch_size=BATCH_SIZE)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X, y = np.arange(len(all_df)), np.array([label2id[l] for l in all_df["emotion_final"]])
    fold_results = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        print(f"\n🌀 FOLD {fold_idx+1}")
        tr_loader = DataLoader(HybridSERDataset(all_df.iloc[tr_idx], label2id, audio_map, augment=True), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        va_loader = DataLoader(HybridSERDataset(all_df.iloc[va_idx], label2id, audio_map), batch_size=BATCH_SIZE, drop_last=True)

        model = HybridSERModel(num_labels, device).to(device)
        cw = compute_class_weight("balanced", classes=np.unique(y[tr_idx]), y=y[tr_idx])
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device), label_smoothing=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        
        best_f1 = 0
        ckpt = config.CHECKPOINT_DIR / f"best_hybrid_fold_{fold_idx}.pt"

        for epoch in range(1, NUM_EPOCHS + 1):
            model.train(); optimizer.zero_grad()
            for b_idx, batch in enumerate(tqdm(tr_loader, desc=f"Ep{epoch}", leave=False)):
                imgs, raws, lbs = batch["image"].to(device), batch["raw_audio"].to(device), batch["label"].to(device)
                out = model(imgs, raws)
                loss = criterion(out, lbs) / ACCUM_STEPS
                loss.backward()
                
                if (b_idx + 1) % ACCUM_STEPS == 0:
                    optimizer.step(); optimizer.zero_grad()
            
            model.eval(); preds, truth = [], []
            with torch.no_grad():
                for b in va_loader:
                    o = model(b["image"].to(device), b["raw_audio"].to(device))
                    preds.extend(torch.argmax(o, 1).cpu().numpy()); truth.extend(b["label"].numpy())
            
            va_acc, va_f1 = accuracy_score(truth, preds), f1_score(truth, preds, average="macro")
            print(f"   Ep {epoch:2d}/{NUM_EPOCHS} | Val Acc {va_acc:.3f} | Val F1 {va_f1:.3f}")
            if va_f1 > best_f1: best_f1 = va_f1; torch.save(model.state_dict(), ckpt)
        
        fold_results.append({"f1": best_f1})
        print(f"📊 FOLD {fold_idx+1} Final F1: {best_f1*100:.1f}%")

    print(f"\n🏆 MEAN HYBRID F1: {np.mean([r['f1'] for r in fold_results])*100:.2f}%")

if __name__ == "__main__":
    main()
