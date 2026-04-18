"""
Method 17.3: ResNet-50 "The Last Stand".
Strategy: 
1. Revert to ResNet-50 Visual Baseline (Method 16).
2. Phase 1: Train 40% Teacher on ALL labeled data (Standard Shuffle + Weights).
3. Phase 2: Pseudo-label the 5,000 unlabelled files.
4. Phase 3: Train 60% Student on the massive Enriched set.
Goal: 60% Individual Audio Accuracy.
"""

import os, sys, random, torch, librosa, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
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
BATCH_SIZE     = 16           # Local batch
ACCUM_STEPS    = 8            # Virtual batch = 128
NUM_EPOCHS_T   = 12           # Fast Teacher
NUM_EPOCHS_S   = 20           # Growth Student
LR             = 1e-4
SCL_TEMP       = 0.1
SCL_WEIGHT     = 0.1          # SCL is a helper guide
LABEL_SMOOTH   = 0.1
CONF_THRESHOLD = 0.85
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
        bn = Path(str(row["audio_relpath"]).replace("\\", "/")).name
        path = self.audio_map.get(bn)
        try:
            audio, _ = librosa.load(path, sr=SR, mono=True)
        except:
            audio = np.zeros(SR * MAX_DURATION)
        
        target_len = SR * MAX_DURATION
        if len(audio) > target_len: audio = audio[:target_len]
        else: audio = np.pad(audio, (0, target_len - len(audio)))
        
        mel = librosa.feature.melspectrogram(y=audio.astype(np.float32), sr=SR, n_mels=128, n_fft=400, hop_length=160)
        static = librosa.power_to_db(mel, ref=np.max)
        d1 = librosa.feature.delta(static)
        d2 = librosa.feature.delta(static, order=2)
        img = np.stack([static, d1, d2], axis=0) # [3, 128, 501]
        
        if self.augment:
            if random.random() > 0.5:
                f = random.randint(0, 15); f0 = random.randint(0, 128-f); img[:, f0:f0+f, :] = img.min()
            if random.random() > 0.5:
                t = random.randint(0, 30); t0 = random.randint(0, img.shape[2]-t); img[:, :, t0:t0+t] = img.min()

        # Reliable Per-Channel Min-Max Scaling (Reverted to original)
        mn = img.min(axis=(1, 2), keepdims=True)
        mx = img.max(axis=(1, 2), keepdims=True)
        img = (img - mn) / (mx - mn + 1e-9)
        return {"image": torch.tensor(img, dtype=torch.float32), "label": torch.tensor(self.label2id.get(row["emotion_final"], -1), dtype=torch.long)}

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
        self.backbone = models.resnet50(pretrained=True)
        dim_in = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.projection_head = nn.Sequential(nn.Linear(dim_in, 512), nn.ReLU(), nn.Linear(512, 128))
        self.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(dim_in, num_labels))
    def forward(self, x):
        feat = self.backbone(x)
        embeddings = self.projection_head(feat)
        logits = self.classifier(feat)
        return logits, embeddings

def train_cycle(model, tr_loader, va_loader, device, epochs, ckpt_path, class_weights):
    ce_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)
    scl_fn = SupervisedContrastiveLoss(SCL_TEMP)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_acc = 0
    for epoch in range(1, epochs + 1):
        model.train(); t_loss = 0; optimizer.zero_grad()
        for b_idx, batch in enumerate(tqdm(tr_loader, desc=f"Ep{epoch}", leave=False)):
            imgs, lbs = batch["image"].to(device), batch["label"].to(device)
            logits, feat = model(imgs)
            loss = (1 - SCL_WEIGHT) * ce_fn(logits, lbs) + SCL_WEIGHT * scl_fn(feat, lbs)
            (loss / ACCUM_STEPS).backward()
            if (b_idx+1) % ACCUM_STEPS == 0:
                optimizer.step(); optimizer.zero_grad()
            t_loss += loss.item()
        scheduler.step()
        model.eval(); preds, truth = [], []
        with torch.no_grad():
            for b in va_loader:
                ls, _ = model(b["image"].to(device))
                preds.extend(torch.argmax(ls, 1).cpu().numpy()); truth.extend(b["label"].numpy())
        acc = accuracy_score(truth, preds); f1 = f1_score(truth, preds, average="macro")
        print(f"   Ep {epoch}/{epochs} | Loss {t_loss/len(tr_loader):.3f} | Acc {acc:.3f} | F1 {f1:.3f}")
        if acc > best_acc: best_acc = acc; torch.save(model.state_dict(), ckpt_path); print(f"   🌟 New High")

def main():
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load ALL available splits for maximum Teacher power
    train_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv")
    val_df   = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv")
    test_df  = pd.read_csv(config.SPLIT_CSV_DIR / "test.csv")
    
    audio_map = {}
    src = Path("/content/dataset") if Path("/content/dataset").exists() else Path(config.DATA_ROOT).parent
    for p in src.rglob("*.wav"): audio_map[p.name] = p
    
    classes = sorted(train_df["emotion_final"].unique())
    label2id = {l: i for i, l in enumerate(classes)}; id2label = {i: l for l, i in label2id.items()}; num_labels = len(label2id)
    
    y_tr = [label2id[l] for l in train_df["emotion_final"]]
    cw = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    class_weights = torch.tensor(cw, dtype=torch.float32).to(device)

    tr_loader = DataLoader(VisualSERDataset(train_df, label2id, audio_map, augment=True), batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(VisualSERDataset(val_df, label2id, audio_map), batch_size=BATCH_SIZE)

    print(f"\n🚀 PHASE 1: RESNET-50 TEACHER (CLASSIC SUCCESS CONFIG)")
    model = ContrastiveResNet(num_labels).to(device)
    t_ckpt = config.CHECKPOINT_DIR / "teacher_resnet_best.pt"
    train_cycle(model, tr_loader, va_loader, device, NUM_EPOCHS_T, t_ckpt, class_weights)

    print(f"\n🚀 PHASE 2: RAPID ENRICHMENT (5000 FILES)")
    model.load_state_dict(torch.load(t_ckpt)); model.eval()
    manifest_path = Path("/content/dataset/data/processed/manifest.csv")
    if not manifest_path.exists(): manifest_path = Path(config.SPLIT_CSV_DIR).parent.parent / "manifest.csv"
    full_df = pd.read_csv(manifest_path)
    gold_fns = set(pd.concat([train_df, val_df, test_df])["audio_relpath"].apply(lambda x: Path(x).name))
    silver_df = full_df[~full_df["audio_relpath"].apply(lambda x: Path(x).name).isin(gold_fns)].copy()
    
    silver_ds = VisualSERDataset(silver_df, label2id, audio_map); silver_loader = DataLoader(silver_ds, batch_size=BATCH_SIZE)
    l_list, c_list = [], []
    with torch.no_grad():
        for b in tqdm(silver_loader, desc="Labeling"):
            ls, _ = model(b["image"].to(device)); p = F.softmax(ls, dim=1); c, pred = torch.max(p, dim=1)
            l_list.extend(pred.cpu().numpy()); c_list.extend(c.cpu().numpy())
    
    silver_df["emotion_final"] = [id2label[p] for p in l_list]; silver_df["confidence"] = c_list
    enriched_df = silver_df[silver_df["confidence"] >= CONF_THRESHOLD].copy()
    print(f"📊 Enriched with {len(enriched_df)} samples.")
    final_tr_df = pd.concat([train_df, enriched_df], ignore_index=True)

    print(f"\n🚀 PHASE 3: FINAL GRADUATION PUSH (RESNET-60)")
    y_final = [label2id[l] for l in final_tr_df["emotion_final"]]
    cw_f = compute_class_weight("balanced", classes=np.unique(y_final), y=y_final)
    class_weights_f = torch.tensor(cw_f, dtype=torch.float32).to(device)

    final_loader = DataLoader(VisualSERDataset(final_tr_df, label2id, audio_map, augment=True), batch_size=BATCH_SIZE, shuffle=True)
    student_model = ContrastiveResNet(num_labels).to(device); s_ckpt = config.CHECKPOINT_DIR / "final_resnet50_60.pt"
    train_cycle(student_model, final_loader, va_loader, device, NUM_EPOCHS_S, s_ckpt, class_weights_f)

if __name__ == "__main__":
    main()
