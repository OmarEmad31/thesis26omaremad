"""
HArnESS-VAD-Mixup-Stats: Egyptian Arabic SER (The 50% Individual Push).
Information-Density Maximization via Trimming, Stats, and Interpolation.

Backbone: WavLM-Base-Plus.
Preprocessing: Online VAD (librosa.effects.trim).
Pooling: Stats-Pooling (Mean + StdDev).
Augmentation: Acoustic Mixup (alpha=0.4).
Loss: Hybrid SupCon + Label Smoothed CE.
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
# LOSS: SUPCON + LABEL SMOOTHED CE
# ---------------------------------------------------------------------------
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: [BS, Dim], labels: [BS] (hard labels for SupCon)
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
# ARCHITECTURE: STATS POOLING + DUAL HEAD
# ---------------------------------------------------------------------------
class StatsPooling(nn.Module):
    def forward(self, x, mask=None):
        # x: [BS, Seq, Hidden]
        if mask is not None:
            mask_exp = mask.unsqueeze(-1).expand(x.size()).float()
            # Mean
            sum_x = torch.sum(x * mask_exp, 1)
            counts = torch.clamp(mask_exp.sum(1), min=1e-9)
            mean = sum_x / counts
            # StdDev
            sq_diff = (x - mean.unsqueeze(1))**2 * mask_exp
            std = torch.sqrt(torch.sum(sq_diff, 1) / counts + 1e-6)
        else:
            mean = x.mean(dim=1)
            std = x.std(dim=1)
        return torch.cat([mean, std], dim=-1) # [BS, 768 * 2 = 1536]

class VADMixupWavLM(nn.Module):
    def __init__(self, num_labels, model_name="microsoft/wavlm-base-plus"):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained(model_name)
        self.pooling = StatsPooling()
        
        # 1536 dim due to Stats Pooling (Mean + StdDev)
        self.projector = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Linear(512, num_labels)
        )

    def forward(self, wav, mask=None, mode="train"):
        wav = (wav - wav.mean(dim=-1, keepdim=True)) / (wav.std(dim=-1, keepdim=True) + 1e-6)
        outputs = self.wavlm(wav).last_hidden_state
        
        if mask is not None:
            down_mask = mask[:, ::320][:, :outputs.shape[1]]
        else:
            down_mask = None
            
        pooled = self.pooling(outputs, mask=down_mask)
        
        if mode == "contrast":
            return self.projector(pooled)
        elif mode == "classify":
            return self.classifier(pooled)
        else:
            return self.projector(pooled), self.classifier(pooled)

# ---------------------------------------------------------------------------
# DATASET: ONLINE VAD TRIMMING
# ---------------------------------------------------------------------------
class VADEgyptianDataset(Dataset):
    def __init__(self, df, audio_map):
        self.df = df; self.audio_map = audio_map
        self.max_len = 80000 # 5 seconds of ACTIVE speech is plenty

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]; bn = Path(str(row["audio_relpath"]).replace("\\", "/")).name
        path = self.audio_map.get(bn); audio, _ = librosa.load(path, sr=16000, mono=True)
        
        # ONLINE VAD: Trim silence
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=25)
        
        mask = np.zeros(self.max_len, dtype=np.float32)
        if len(audio_trimmed) > self.max_len:
            audio_trimmed = audio_trimmed[:self.max_len]; mask[:] = 1.0
        else:
            mask[:len(audio_trimmed)] = 1.0; audio_trimmed = np.pad(audio_trimmed, (0, self.max_len - len(audio_trimmed)))
            
        return {"wav": torch.tensor(audio_trimmed, dtype=torch.float32), 
                "mask": torch.tensor(mask, dtype=torch.float32), 
                "label": torch.tensor(row["label_id"], dtype=torch.long)}

# ---------------------------------------------------------------------------
# TRAINING: MIXUP PROTOCOL
# ---------------------------------------------------------------------------
def apply_mixup(wav, lbl, alpha=0.4):
    if alpha <= 0: return wav, lbl, None, None, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(wav.size(0))
    mixed_wav = lam * wav + (1 - lam) * wav[idx, :]
    lbl_a, lbl_b = lbl, lbl[idx]
    return mixed_wav, lbl_a, lbl_b, lam

def evaluate(model, loader, device):
    model.eval(); ps, ts = [], []
    with torch.no_grad():
        for b in loader:
            l = model(b["wav"].to(device), b["mask"].to(device), mode="classify")
            ps.extend(torch.argmax(l, 1).cpu().numpy()); ts.extend(b["label"].numpy())
    return accuracy_score(ts, ps), f1_score(ts, ps, average="macro")

def main():
    torch.manual_seed(42); np.random.seed(42); device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print("🏗️ Initializing VAD-Mixup-Stats Engine...")
    train_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv")
    val_df   = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv")
    test_df  = pd.read_csv(config.SPLIT_CSV_DIR / "test.csv")
    classes = sorted(train_df["emotion_final"].unique())
    lid = {l: i for i, l in enumerate(classes)}
    for df in [train_df, val_df, test_df]: df["label_id"] = df["emotion_final"].map(lid)
    
    audio_map = {}
    data_search_path = Path("/content/dataset") if Path("/content/dataset").exists() else Path(config.DATA_ROOT).parent
    for ext in ["*.wav", "*.WAV", "*.Wav"]:
        for p in data_search_path.rglob(ext): audio_map[p.name] = p

    weights = torch.tensor(compute_class_weight("balanced", classes=np.unique(train_df["label_id"].values), 
                                               y=train_df["label_id"].values), dtype=torch.float32).to(device)

    model = VADMixupWavLM(len(classes)).to(device)
    l_sup = SupConLoss(temperature=0.1); l_ce = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    
    tr_loader = DataLoader(VADEgyptianDataset(train_df, audio_map), batch_size=32, shuffle=True)
    va_loader = DataLoader(VADEgyptianDataset(val_df, audio_map), batch_size=32)
    te_loader = DataLoader(VADEgyptianDataset(test_df, audio_map), batch_size=32)

    print("\n🔥 STAGE 1: DENSITY WARMUP (Stats Pooling)")
    for param in model.wavlm.parameters(): param.requires_grad = False
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    for epoch in range(1, 6):
        model.train(); pbar = tqdm(tr_loader, desc=f"Warmup Ep{epoch}")
        for b in pbar:
            w = b["wav"].to(device); l = b["label"].to(device); m = b["mask"].to(device)
            # No Mixup in Warmup to stabilize head
            p, c = model(w, m, mode="both")
            loss = 1.0 * l_sup(p, l) + 1.0 * l_ce(c, l)
            opt.zero_grad(); loss.backward(); opt.step()
        va, vf = evaluate(model, va_loader, device)
        print(f"📈 Warmup Epoch {epoch} | Val Acc: {va:.3f} | Val F1: {vf:.3f}")

    print("\n🚀 STAGE 2: THE 50% MIXUP PUSH (50 Epochs Milestone)")
    for param in model.wavlm.parameters(): param.requires_grad = True
    opt = torch.optim.AdamW([{"params": model.wavlm.parameters(), "lr": 1e-5},
                            {"params": model.projector.parameters(), "lr": 1e-4},
                            {"params": model.classifier.parameters(), "lr": 1e-4}], weight_decay=0.01)
    sch = get_cosine_schedule_with_warmup(opt, num_warmup_steps=len(tr_loader), num_training_steps=len(tr_loader)*50)
    
    best_va = 0
    for epoch in range(1, 51):
        model.train(); pbar = tqdm(tr_loader, desc=f"Mixup Ep{epoch}")
        for b in pbar:
            w = b["wav"].to(device); l = b["label"].to(device); m = b["mask"].to(device)
            
            # APPLY MIXUP (Waveform Domain)
            w_mix, l_a, l_b, lam = apply_mixup(w, l, alpha=0.4)
            
            # SupCon uses hard labels (original labels), Classifier uses mixed labels
            p, _ = model(w, m, mode="both") # SupCon on raw
            _, c_mix = model(w_mix, m, mode="both") # Classify on mixed
            
            loss = 1.0 * l_sup(p, l) + 1.0 * (lam * l_ce(c_mix, l_a) + (1 - lam) * l_ce(c_mix, l_b))
            
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            
        va, vf = evaluate(model, va_loader, device)
        ta, tf = evaluate(model, te_loader, device)
        print(f"🏆 Epoch {epoch} Results: VAL Acc: {va:.3f} | TEST Acc: {ta:.3f} F1: {tf:.3f}")
        if va > best_va:
            best_va = va; torch.save(model.state_dict(), config.CHECKPOINT_DIR / "vad_mixup_best.pt")

if __name__ == "__main__": main()
