"""
HArnESS-Augmented-Winner: Egyptian Arabic SER (The 50% Milestone Push).
Reverting to WavLM-Base-Plus + Heavy Online Augmentation.

Backbone: microsoft/wavlm-base-plus (The 36% Champion).
Augmentation: Temporal Jitter, Gaussian Noise, Fast Pitch-Warping.
Target: 50% Milestone.
"""

import os, sys, torch, librosa, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import WavLMModel, get_cosine_schedule_with_warmup
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
# LOSS: SUPCON (Geometric Resolution)
# ---------------------------------------------------------------------------
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]; labels = labels.contiguous().view(-1, 1)
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
# ARCHITECTURE: THE 36% CHAMPION
# ---------------------------------------------------------------------------
class AugmentedWavLM(nn.Module):
    def __init__(self, num_labels, model_name="microsoft/wavlm-base-plus"):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained(model_name)
        self.projector = nn.Sequential(nn.Linear(768, 512), nn.ReLU(), nn.Linear(512, 128))
        self.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(768, num_labels))

    def forward(self, wav, mask=None, mode="train"):
        wav = (wav - wav.mean(dim=-1, keepdim=True)) / (wav.std(dim=-1, keepdim=True) + 1e-6)
        outputs = self.wavlm(wav).last_hidden_state
        
        # Masked Pooling logic restored
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
class AugmentedDataset(Dataset):
    def __init__(self, df, audio_map, augment=False):
        self.df = df; self.audio_map = audio_map; self.max_len = 160000; self.augment = augment

    def __len__(self): return len(self.df)

    def apply_augment(self, audio):
        # 1. Random Noise
        if np.random.rand() < 0.4:
            noise = np.random.normal(0, np.random.uniform(0.001, 0.005), len(audio))
            audio = audio + noise
        
        # 2. Random Time Shift (Jitter)
        if np.random.rand() < 0.4:
            shift = np.random.randint(-16000, 16000)
            audio = np.roll(audio, shift)
            
        # 3. Fast Pitch Warping (via Resampling)
        if np.random.rand() < 0.3:
            orig_len = len(audio)
            step = np.random.uniform(0.9, 1.1)
            audio = librosa.resample(audio, orig_sr=16000, target_sr=int(16000 * step))
            if len(audio) > orig_len: audio = audio[:orig_len]
            else: audio = np.pad(audio, (0, orig_len - len(audio)))
        return audio

    def __getitem__(self, idx):
        row = self.df.iloc[idx]; bn = Path(str(row["audio_relpath"]).replace("\\", "/")).name
        path = self.audio_map.get(bn); audio, _ = librosa.load(path, sr=16000, mono=True)
        
        if self.augment: audio = self.apply_augment(audio)
            
        mask = np.zeros(self.max_len, dtype=np.float32)
        if len(audio) > self.max_len: audio = audio[:self.max_len]; mask[:] = 1.0
        else: mask[:len(audio)] = 1.0; audio = np.pad(audio, (0, self.max_len - len(audio)))
        return {"wav": torch.tensor(audio, dtype=torch.float32), 
                "mask": torch.tensor(mask, dtype=torch.float32), 
                "label": torch.tensor(row["label_id"], dtype=torch.long)}

def evaluate(model, loader, device):
    model.eval(); ps, ts = [], []
    with torch.no_grad():
        for b in loader:
            l = model(b["wav"].to(device), b["mask"].to(device), mode="classify")
            ps.extend(torch.argmax(l, 1).cpu().numpy()); ts.extend(b["label"].numpy())
    return accuracy_score(ts, ps), f1_score(ts, ps, average="macro")

def main():
    torch.manual_seed(42); np.random.seed(42); device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("🏗️ Initializing Augmented-Winner (WavLM 90M Champion)...")
    train_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv")
    val_df   = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv")
    test_df  = pd.read_csv(config.SPLIT_CSV_DIR / "test.csv")
    classes = sorted(train_df["emotion_final"].unique()); lid = {l: i for i, l in enumerate(classes)}
    for df in [train_df, val_df, test_df]: df["label_id"] = df["emotion_final"].map(lid)
    
    audio_map = {}
    data_search_path = Path("/content/dataset") if Path("/content/dataset").exists() else Path(config.DATA_ROOT).parent
    for ext in ["*.wav", "*.WAV", "*.Wav"]:
        for p in data_search_path.rglob(ext): audio_map[p.name] = p

    y_tr = train_df["label_id"].values
    weights = torch.tensor(compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr), dtype=torch.float32).to(device)

    model = AugmentedWavLM(len(classes)).to(device)
    l_sup = SupConLoss(temperature=0.07); l_ce = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    
    tr_loader = DataLoader(AugmentedDataset(train_df, audio_map, augment=True), batch_size=32, shuffle=True)
    va_loader = DataLoader(AugmentedDataset(val_df, audio_map, augment=False), batch_size=32)
    te_loader = DataLoader(AugmentedDataset(test_df, audio_map, augment=False), batch_size=32)

    # DLR: Stable backbone, Accelerated Head
    optimizer = torch.optim.AdamW([
        {"params": model.wavlm.parameters(), "lr": 1e-5},
        {"params": model.projector.parameters(), "lr": 2e-4},
        {"params": model.classifier.parameters(), "lr": 5e-4},
    ], weight_decay=0.01)
    
    sch = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(tr_loader), num_training_steps=len(tr_loader)*60)
    
    best_va = 0
    for epoch in range(1, 61):
        model.train(); pbar = tqdm(tr_loader, desc=f"Sprint Ep{epoch}")
        for b in pbar:
            w, m, l = b["wav"].to(device), b["mask"].to(device), b["label"].to(device)
            p, c = model(w, m, mode="both"); loss = 1.0 * l_sup(p, l) + 1.0 * l_ce(c, l)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); sch.step()
        
        va, vf = evaluate(model, va_loader, device); ta, tf = evaluate(model, te_loader, device)
        print(f"🏆 Ep {epoch} Results: VAL Acc: {va:.3f} | TEST Acc: {ta:.3f} (F1: {tf:.3f})")
        if va > best_va:
            best_va = va; torch.save(model.state_dict(), config.CHECKPOINT_DIR / "augmented_winner_best.pt")

if __name__ == "__main__": main()
