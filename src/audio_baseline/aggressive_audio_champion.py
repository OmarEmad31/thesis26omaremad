"""
HArnESS-Super-Turbo-Champion (Memory-Safe V12).
Egyptian Arabic SER (Target: 57%+ Val Acc).
FIXES: CUDA OOM (4GB), Multi-Worker Echoing, Slow Runtime.
"""

import os, sys, random
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import WavLMModel, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch
from torch.amp import GradScaler, autocast

# ---------------------------------------------------------------------------
# CHAMPION ARCHITECTURE: WLP + SUPCON HEAD + GRADIENT CHECKPOINTING
# ---------------------------------------------------------------------------
class AggressiveChampionSER(nn.Module):
    def __init__(self, num_labels, model_name="microsoft/wavlm-base-plus"):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained(model_name, output_hidden_states=True)
        # CRITICAL FOR 4GB GPUs
        self.wavlm.gradient_checkpointing_enable()
        
        self.layer_weights = nn.Parameter(torch.ones(13))
        self.projector = nn.Sequential(nn.Linear(768, 512), nn.ReLU(), nn.Linear(512, 256))
        self.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(768, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(512, num_labels)
        )

    def forward(self, wav, mask):
        wav = (wav - wav.mean(dim=-1, keepdim=True)) / (wav.std(dim=-1, keepdim=True) + 1e-6)
        outputs = self.wavlm(wav, attention_mask=mask)
        hidden_states = torch.stack(outputs.hidden_states, dim=0)
        w = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        weighted_hidden = (hidden_states * w).sum(dim=0)
        down_mask = mask[:, ::320][:, :weighted_hidden.shape[1]]
        mask_exp = down_mask.unsqueeze(-1).expand(weighted_hidden.size()).float()
        pooled = torch.sum(weighted_hidden * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)
        z = F.normalize(self.projector(pooled), p=2, dim=1)
        logits = self.classifier(pooled)
        return logits, z

# ---------------------------------------------------------------------------
# SOTA LOSS: SUPERVISED CONTRASTIVE LOSS
# ---------------------------------------------------------------------------
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        return -mean_log_prob_pos.mean()

# ---------------------------------------------------------------------------
# DATASET (REDUCED TO 7 SECONDS)
# ---------------------------------------------------------------------------
class ChampionDataset(Dataset):
    def __init__(self, df, audio_map, augment=False):
        self.df = df; self.audio_map = audio_map
        self.max_len = 112000 # 7 SECONDS @ 16kHz (SAFE SPOT FOR 4GB)
        self.augment = augment
        self.aug_pipe = Compose([
            AddGaussianNoise(p=0.3),
            PitchShift(min_semitones=-1.5, max_semitones=1.5, p=0.4),
            TimeStretch(min_rate=0.8, max_rate=1.2, p=0.2)
        ])
        
    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]; bn = Path(str(row["audio_relpath"]).replace("\\", "/")).name
        path = self.audio_map.get(bn)
        if not path: return self.__getitem__((idx + 1) % len(self.df))
        
        audio, _ = librosa.load(path, sr=16000, mono=True)
        if self.augment: audio = self.aug_pipe(samples=audio, sample_rate=16000)
        
        mask = np.zeros(self.max_len, dtype=np.float32)
        if len(audio) > self.max_len: 
            audio = audio[:self.max_len]; mask[:] = 1.0
        else: 
            mask[:len(audio)] = 1.0; audio = np.pad(audio, (0, self.max_len - len(audio)))
            
        return {
            "wav": torch.tensor(audio, dtype=torch.float32), 
            "mask": torch.tensor(mask, dtype=torch.float32), 
            "label": torch.tensor(row["lid"], dtype=torch.long)
        }

def evaluate(model, loader, device):
    model.eval(); ps, ts = [], []
    with torch.no_grad():
        for b in loader:
            l, _ = model(b["wav"].to(device), b["mask"].to(device))
            ps.extend(torch.argmax(l, 1).cpu().numpy()); ts.extend(b["label"].numpy())
    return accuracy_score(ts, ps), f1_score(ts, ps, average="macro")

def main():
    # 1. Safe Load
    print("[DEBUG] Phase 1: Safe Data Loading...", flush=True)
    csv_dir = Path("D:/Thesis Project/data/processed/splits/audio_eligible")
    read_args = {'engine': 'python', 'encoding': 'utf-8'}
    tr_df = pd.read_csv(csv_dir / "train.csv", **read_args)
    va_df = pd.read_csv(csv_dir / "val.csv", **read_args)
    te_df = pd.read_csv(csv_dir / "test.csv", **read_args)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INIT] LAUNCHING MEMORY-SAFE CHAMPION | Device: {device}", flush=True)

    # Project setup
    project_root = str(Path(__file__).parent.parent.parent.absolute())
    if project_root not in sys.path: sys.path.insert(0, project_root)
    from src.audio_baseline import config

    classes = config.EMOTIONS
    lid = {l: i for i, l in enumerate(classes)}
    for df in [tr_df, va_df, te_df]: df["lid"] = df["emotion_final"].map(lid)
    
    audio_map = {}
    p_root = Path(config.DATA_ROOT).parent
    for ext in ["*.wav", "*.WAV", "*.Wav"]:
        for f in p_root.rglob(ext): audio_map[f.name] = f

    y_tr = tr_df["lid"].values
    class_counts = np.bincount(y_tr)
    weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    samples_weight = torch.tensor([weights[t] for t in y_tr])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    # Memory-Safe Settings
    BATCH_SIZE = 2; ACCUM_STEPS = 8; EPOCHS = 15; ALPHA = 0.3
    loader_args = {"batch_size": BATCH_SIZE, "num_workers": 2, "pin_memory": True} 
    tr_loader = DataLoader(ChampionDataset(tr_df, audio_map, augment=True), sampler=sampler, **loader_args)
    va_loader = DataLoader(ChampionDataset(va_df, audio_map), **loader_args)
    te_loader = DataLoader(ChampionDataset(te_df, audio_map), **loader_args)

    # Model & Optimization
    os.environ["HF_HOME"] = "D:/HuggingFaceCache"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    model = AggressiveChampionSER(len(classes)).to(device)
    l_ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    l_sc = SupConLoss(temperature=0.1)
    scaler = GradScaler('cuda')

    optimizer = torch.optim.AdamW([
        {"params": model.wavlm.parameters(), "lr": 3e-5}, 
        {"params": [p for n, p in model.named_parameters() if "wavlm" not in n], "lr": 1e-3}
    ], weight_decay=0.01)
    
    warmup_steps = int(0.1 * len(tr_loader) * EPOCHS)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, len(tr_loader) * EPOCHS)

    # Training Loop
    best_acc = 0
    print(f"\n[STEP] Starting Memory-Safe Loop. Target: 57% Acc.", flush=True)
    for epoch in range(1, EPOCHS + 1):
        model.train(); t_loss = 0; optimizer.zero_grad()
        pbar = tqdm(tr_loader, desc=f"Ep {epoch}/{EPOCHS}")
        for i, b in enumerate(pbar):
            w, m, l = b["wav"].to(device), b["mask"].to(device), b["label"].to(device)
            with autocast('cuda'):
                logits, z = model(w, m)
                loss = ((1 - ALPHA) * l_ce(logits, l) + ALPHA * l_sc(z, l)) / ACCUM_STEPS
            scaler.scale(loss).backward()
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(); scheduler.step()
            t_loss += loss.item() * ACCUM_STEPS
            pbar.set_postfix({"loss": f"{t_loss/(i+1):.4f}", "lr": f"{optimizer.param_groups[1]['lr']:.6f}"})
        
        v_acc, v_f1 = evaluate(model, va_loader, device)
        t_acc, t_f1 = evaluate(model, te_loader, device)
        print(f"Epoch {epoch} | Val: {v_acc:.3f} | Test: {t_acc:.3f}", flush=True)
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), config.CHECKPOINT_DIR / "aggressive_champion.pt")
            print("New Best Saved", flush=True)

if __name__ == "__main__":
    main()
