"""
HArnESS-5Fold-Ensemble: Egyptian Arabic SER (The 50% Milestone Push).
5-Fold Cross-Validation + Heavy Online Waveform Augmentation.

Backbone: WavLM-Base-Plus (The Verified Winner).
Strategy: 5-Fold Ensemble (Voting).
Augmentation: Online Pitch, Speed, and Noise Injection.
Target: 50% Milestone.
"""

import os, sys, torch, librosa, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import WavLMModel, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from tqdm import tqdm
from pathlib import Path

# Add project root
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.audio_baseline import config

# ---------------------------------------------------------------------------
# LOSS: SUPCON (Clustering)
# ---------------------------------------------------------------------------
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        features = F.normalize(features, dim=1)
        logits = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)
        return -mean_log_prob_pos.mean()

# ---------------------------------------------------------------------------
# ARCHITECTURE: WAVLM-BASE-PLUS (ENCORE)
# ---------------------------------------------------------------------------
class EnsembleMember(nn.Module):
    def __init__(self, num_labels, model_name="microsoft/wavlm-base-plus"):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained(model_name)
        self.projector = nn.Sequential(nn.Linear(768, 512), nn.ReLU(), nn.Linear(512, 128))
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(768, 512), nn.ReLU(), nn.Linear(512, num_labels))

    def forward(self, wav, mask=None, mode="train"):
        wav = (wav - wav.mean(dim=-1, keepdim=True)) / (wav.std(dim=-1, keepdim=True) + 1e-6)
        outputs = self.wavlm(wav).last_hidden_state
        if mask is not None:
            down_mask = mask[:, ::320][:, :outputs.shape[1]]
            mask_exp = down_mask.unsqueeze(-1).expand(outputs.size()).float()
            pooled = torch.sum(outputs * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)
        else:
            pooled = outputs.mean(dim=1)
        if mode == "contrast": return self.projector(pooled)
        elif mode == "classify": return self.classifier(pooled)
        else: return self.projector(pooled), self.classifier(pooled)

# ---------------------------------------------------------------------------
# DATASET: HEAVY ONLINE AUGMENTATION
# ---------------------------------------------------------------------------
class HeavyAugDataset(Dataset):
    def __init__(self, df, audio_map, augment=False):
        self.df = df; self.audio_map = audio_map; self.max_len = 160000; self.augment = augment

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]; bn = Path(str(row["audio_relpath"]).replace("\\", "/")).name
        path = self.audio_map.get(bn); audio, _ = librosa.load(path, sr=16000, mono=True)
        
        # ONLINE AUGMENTATION (Prosody Variation)
        if self.augment:
            # 1. Pitch Shifting (-2 to +2 semitones)
            if np.random.rand() > 0.5:
                audio = librosa.effects.pitch_shift(audio, sr=16000, n_steps=np.random.uniform(-1.5, 1.5))
            # 2. Add White Noise
            if np.random.rand() > 0.5:
                noise = np.random.randn(len(audio))
                audio = audio + 0.005 * noise * np.max(audio)
        
        mask = np.zeros(self.max_len, dtype=np.float32)
        if len(audio) > self.max_len:
            audio = audio[:self.max_len]; mask[:] = 1.0
        else:
            mask[:len(audio)] = 1.0; audio = np.pad(audio, (0, self.max_len - len(audio)))
        return {"wav": torch.tensor(audio, dtype=torch.float32), 
                "mask": torch.tensor(mask, dtype=torch.float32), 
                "label": torch.tensor(row["label_id"], dtype=torch.long)}

# ---------------------------------------------------------------------------
# ORCHESTRATION: 5-FOLD LOOP
# ---------------------------------------------------------------------------
def train_one_fold(fold_idx, train_loader, val_loader, num_classes, device):
    print(f"\n🎧 TRAINING FOLD {fold_idx+1}/5...")
    model = EnsembleMember(num_classes).to(device)
    l_sup = SupConLoss(); l_ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Stage 1: Warmup
    for p in model.wavlm.parameters(): p.requires_grad = False
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    for _ in range(2):
        model.train()
        for b in train_loader:
            p, c = model(b["wav"].to(device), b["mask"].to(device), mode="both")
            loss = l_sup(p, b["label"].to(device)) + l_ce(c, b["label"].to(device))
            opt.zero_grad(); loss.backward(); opt.step()

    # Stage 2: Fine-Tuning
    for p in model.wavlm.parameters(): p.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    best_acc, ckpt = 0, None
    for epoch in range(1, 16): # Faster folds
        model.train()
        for b in train_loader:
            p, c = model(b["wav"].to(device), b["mask"].to(device), mode="both")
            loss = l_sup(p, b["label"].to(device)) + l_ce(c, b["label"].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        
        # Eval
        model.eval(); ps, ts = [], []
        with torch.no_grad():
            for b in val_loader:
                l = model(b["wav"].to(device), b["mask"].to(device), mode="classify")
                ps.extend(torch.argmax(l, 1).cpu().numpy()); ts.extend(b["label"].numpy())
        acc = accuracy_score(ts, ps)
        if acc > best_acc:
            best_acc = acc; ckpt = model.state_dict()
    
    save_path = config.CHECKPOINT_DIR / f"fold_{fold_idx+1}_best.pt"
    torch.save(ckpt, save_path)
    return save_path

def main():
    torch.manual_seed(42); np.random.seed(42); device = "cuda:0" if torch.cuda.is_available() else "cpu"
    merged_path = config.SPLIT_CSV_DIR / "train_val_merged.csv"
    if merged_path.exists():
        full_df = pd.read_csv(merged_path)
    else:
        tr = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv")
        vl = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv")
        full_df = pd.concat([tr, vl]).reset_index(drop=True)
    
    test_df = pd.read_csv(config.SPLIT_CSV_DIR / "test.csv")
    classes = sorted(full_df["emotion_final"].unique()); lid = {l: i for i, l in enumerate(classes)}
    for df in [full_df, test_df]: df["label_id"] = df["emotion_final"].map(lid)
    
    audio_map = {}
    data_search_path = Path("/content/dataset") if Path("/content/dataset").exists() else Path(config.DATA_ROOT).parent
    for ext in ["*.wav", "*.WAV", "*.Wav"]:
        for p in data_search_path.rglob(ext): audio_map[p.name] = p

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_paths = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(full_df)):
        tr_df = full_df.iloc[train_idx]; va_df = full_df.iloc[val_idx]
        tr_ds = HeavyAugDataset(tr_df, audio_map, augment=True) # Heavy Aug
        va_ds = HeavyAugDataset(va_df, audio_map, augment=False)
        tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True)
        va_loader = DataLoader(va_ds, batch_size=32)
        
        path = train_one_fold(fold_idx, tr_loader, va_loader, len(classes), device)
        fold_paths.append(path)

    print("\n🏆 CALCULATING ENSEMBLE TEST ACCURACY...")
    models = []
    for p in fold_paths:
        m = EnsembleMember(len(classes)).to(device)
        m.load_state_dict(torch.load(p)); m.eval(); models.append(m)
    
    te_loader = DataLoader(HeavyAugDataset(test_df, audio_map, augment=False), batch_size=32)
    all_logits, ts = [], []
    with torch.no_grad():
        for b in te_loader:
            fold_logits = []
            for m in models: fold_logits.append(F.softmax(m(b["wav"].to(device), b["mask"].to(device), mode="classify"), dim=1))
            mean_logits = torch.stack(fold_logits).mean(dim=0)
            all_logits.append(mean_logits.cpu().numpy()); ts.extend(b["label"].numpy())
    
    ps = np.argmax(np.vstack(all_logits), axis=1)
    print(f"🔥 FINAL ENSEMBLE TEST ACC: {accuracy_score(ts, ps):.3f} | F1: {f1_score(ts, ps, average='macro'):.3f}")

if __name__ == "__main__": main()
