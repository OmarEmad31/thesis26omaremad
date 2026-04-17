"""
Method 17: Contrastive Enrichment "The Final Push".
Strategy: 
1. Train a Teacher on 915 Gold Samples (SCL + ResNet-50).
2. Pseudo-label the other 4,800 Egyptian recordings.
3. Retrain on the Enriched Dataset (Gold + Silver).
Goal: 60% Individual Audio Accuracy.
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

# Add project root
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.audio_baseline import config

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BATCH_SIZE     = 16
ACCUM_STEPS    = 8
NUM_EPOCHS_T   = 15   # Teacher training
NUM_EPOCHS_S   = 20   # Student (Enriched) training
LR             = 5e-5
SCL_TEMP       = 0.1
SCL_WEIGHT     = 0.1
LABEL_SMOOTH   = 0.1
CONF_THRESHOLD = 0.85 # Confidence required for pseudo-labeling
SR             = 16000
MAX_DURATION   = 5

# ---------------------------------------------------------------------------
# DATASET & MODEL
# ---------------------------------------------------------------------------
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
        
        label = row["emotion_final"]
        label_id = self.label2id[label] if label in self.label2id else -1
        return {"image": torch.tensor(img, dtype=torch.float32), "label": torch.tensor(label_id, dtype=torch.long)}

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

class ContrastiveResNet(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        # Use modern weights API
        weights = models.ResNet50_Weights.DEFAULT
        self.backbone = models.resnet50(weights=weights)
        dim_in = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.projection_head = nn.Sequential(nn.Linear(dim_in, 512), nn.ReLU(), nn.Linear(512, 128))
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(dim_in, num_labels))
    def forward(self, x):
        feat = self.backbone(x)
        embeddings = self.projection_head(feat)
        logits = self.classifier(feat)
        return logits, embeddings

# ---------------------------------------------------------------------------
# TRAINING LOGIC
# ---------------------------------------------------------------------------
def train_loop(model, tr_loader, va_loader, ce_fn, scl_fn, optimizer, scheduler, device, epochs, ckpt_path):
    best_f1 = 0
    for epoch in range(1, epochs + 1):
        model.train(); t_loss = 0; optimizer.zero_grad()
        for b_idx, batch in enumerate(tqdm(tr_loader, desc=f"Ep{epoch}", leave=False)):
            imgs, lbs = batch["image"].to(device), batch["label"].to(device)
            logits, feat = model(imgs)
            loss = (1 - SCL_WEIGHT) * ce_fn(logits, lbs) + SCL_WEIGHT * scl_fn(feat, lbs)
            (loss / ACCUM_STEPS).backward()
            if (b_idx+1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad()
            t_loss += loss.item()
        scheduler.step()
        model.eval(); preds, truth = [], []
        with torch.no_grad():
            for b in va_loader:
                ls, _ = model(b["image"].to(device))
                preds.extend(torch.argmax(ls, 1).cpu().numpy()); truth.extend(b["label"].numpy())
        va_acc, va_f1 = accuracy_score(truth, preds), f1_score(truth, preds, average="macro")
        print(f"   Ep {epoch}/{epochs} | Loss {t_loss/len(tr_loader):.3f} | Val Acc {va_acc:.3f} | F1 {va_f1:.3f}")
        if va_f1 > best_f1: best_f1 = va_f1; torch.save(model.state_dict(), ckpt_path)
    return best_f1

def main():
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 PHASE 1: TEACHER TRAINING (GOLD ONLY)")
    
    # 1. Load Data
    train_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv")
    val_df   = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv")
    test_df  = pd.read_csv(config.SPLIT_CSV_DIR / "test.csv")
    audio_map = {}
    src = Path("/content") if Path("/content").exists() else Path(config.DATA_ROOT).parent
    for p in src.rglob("*.wav"): audio_map[p.name] = p
    
    label2id = {l: i for i, l in enumerate(sorted(train_df["emotion_final"].unique()))}
    id2label = {i: l for l, i in label2id.items()}
    num_labels = len(label2id)

    tr_loader = DataLoader(VisualSERDataset(train_df, label2id, audio_map, augment=True), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    va_loader = DataLoader(VisualSERDataset(val_df, label2id, audio_map), batch_size=BATCH_SIZE)
    test_loader = DataLoader(VisualSERDataset(test_df, label2id, audio_map), batch_size=BATCH_SIZE)

    model = ContrastiveResNet(num_labels).to(device)
    ce_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    scl_fn = SupervisedContrastiveLoss(SCL_TEMP)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS_T)
    
    teacher_ckpt = config.CHECKPOINT_DIR / "teacher_best.pt"
    train_loop(model, tr_loader, va_loader, ce_fn, scl_fn, optimizer, scheduler, device, NUM_EPOCHS_T, teacher_ckpt)

    print(f"\n🚀 PHASE 2: PSEUDO-LABELING (SILVER ENRICHMENT)")
    model.load_state_dict(torch.load(teacher_ckpt)); model.eval()
    
    # Load full metadata (this should exist from your earlier preprocessing)
    full_df = None
    search_paths = [
        Path(config.SPLIT_CSV_DIR).parent.parent / "manifest.csv",
        Path(config.SPLIT_CSV_DIR).parent.parent / "manifest_with_split.csv",
        Path("/content/dataset/data/processed/manifest.csv"),
    ]
    for p in search_paths:
        if p.exists():
            full_df = pd.read_csv(p)
            print(f"✅ Loaded master manifest from: {p}")
            break

    if full_df is None:
        print("⚠️ manifest.csv not found. Falling back to scanning audio files...")
        full_files = [str(Path(p).relative_to(config.DATA_ROOT)) for p in audio_map.values()]
        full_df = pd.DataFrame({"audio_relpath": full_files, "emotion_final": ["Unknown"] * len(full_files)})

    # Filter out files already in Gold
    gold_files = set(pd.concat([train_df, val_df, test_df])["audio_relpath"].apply(lambda x: Path(x).name))
    silver_df = full_df[~full_df["audio_relpath"].apply(lambda x: Path(x).name).isin(gold_files)].copy()
    
    silver_ds = VisualSERDataset(silver_df, label2id, audio_map, augment=False)
    silver_loader = DataLoader(silver_ds, batch_size=BATCH_SIZE)
    
    pseudo_labels = []; confidences = []
    with torch.no_grad():
        for b in tqdm(silver_loader, desc="Labeling Silver Data"):
            logits, _ = model(b["image"].to(device))
            probs = F.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)
            pseudo_labels.extend(pred.cpu().numpy())
            confidences.extend(conf.cpu().numpy())
    
    silver_df["emotion_final"] = [id2label[p] for p in pseudo_labels]
    silver_df["confidence"] = confidences
    enriched_df = silver_df[silver_df["confidence"] >= CONF_THRESHOLD].copy()
    
    print(f"📊 Enriched with {len(enriched_df)} high-confidence samples.")
    final_train_df = pd.concat([train_df, enriched_df], ignore_index=True)

    print(f"\n🚀 PHASE 3: FINAL PUSH (ENRICHED STUDENT)")
    final_tr_loader = DataLoader(VisualSERDataset(final_train_df, label2id, audio_map, augment=True), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    student_model = ContrastiveResNet(num_labels).to(device)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS_S)
    
    student_ckpt = config.CHECKPOINT_DIR / "final_60_audio_resnet.pt"
    train_loop(student_model, final_tr_loader, va_loader, ce_fn, scl_fn, optimizer, scheduler, device, NUM_EPOCHS_S, student_ckpt)

    print(f"\n🏆 TRAINING COMPLETE. MODEL SAVED TO: {student_ckpt}")

if __name__ == "__main__":
    main()
