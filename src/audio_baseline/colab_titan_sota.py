"""
HArnESS-Colab-Titan-SOTA.
Implementation: WavLM-Large-Robust + WLP + SupCon.
This script is designed to be LOCAL in your project but 100% compatible with Google Colab.
"""

import os, sys, random, subprocess
from pathlib import Path

# --- STEP 1: ENVIRONMENT & AUTO-INSTALL ---
IS_COLAB = 'google.colab' in sys.modules or os.path.exists('/content')

if IS_COLAB:
    print("[INIT] Colab Detected. Installing SOTA dependencies...")
    subprocess.run(["pip", "install", "-q", "transformers==4.48.3", "audiomentations", "librosa", "accelerate", "pyloudnorm", "peft"])
    DRIVE_ZIP = "/content/drive/MyDrive/Thesis Project/Thesis_Audio_Full.zip"
    if os.path.exists(DRIVE_ZIP) and not os.path.exists("/content/dataset"):
        print(f"[INIT] Extracting {DRIVE_ZIP}...")
        import zipfile
        with zipfile.ZipFile(DRIVE_ZIP, 'r') as z: z.extractall('/content/')

# --- STEP 2: HEAVY IMPORTS ---
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

# --- STEP 3: TITAN ARCHITECTURE ---
from peft import LoraConfig, get_peft_model

class TitanAudioSER(nn.Module):
    def __init__(self, num_labels, model_name="microsoft/wavlm-large"):
        super().__init__()
        base_model = WavLMModel.from_pretrained(model_name, output_hidden_states=True)
        
        # LoRA Configuration for SER
        peft_config = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["query", "value"],
            lora_dropout=0.05, bias="none"
        )
        self.wavlm = get_peft_model(base_model, peft_config)
        self.wavlm.gradient_checkpointing_enable() 
        self.layer_weights = nn.Parameter(torch.ones(25)) 
        self.classifier = nn.Sequential(
            nn.Linear(1024, 768), nn.BatchNorm1d(768), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(768, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, num_labels)
        )
        self.projector = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 256))

    def forward(self, wav, mask):
        wav = (wav - wav.mean(dim=-1, keepdim=True)) / (wav.std(dim=-1, keepdim=True) + 1e-6)
        outputs = self.wavlm(wav, attention_mask=mask)
        hidden_states = torch.stack(outputs.hidden_states, dim=0)
        w = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        weighted_hidden = (hidden_states * w).sum(dim=0)
        mask_exp = mask[:, ::320][:, :weighted_hidden.shape[1]].unsqueeze(-1).expand(weighted_hidden.size()).float()
        pooled = torch.sum(weighted_hidden * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)
        return self.classifier(pooled), F.normalize(self.projector(pooled), p=2, dim=1)

class SupConLoss(nn.Module):
    def __init__(self, t=0.07):
        super().__init__(); self.t = t
    def forward(self, z, l):
        mask = torch.eq(l.view(-1,1), l.view(-1,1).T).float().to(z.device)
        logits = torch.div(torch.matmul(z, z.T), self.t)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        exp = torch.exp(logits) * (1 - torch.eye(z.shape[0]).to(z.device))
        log_prob = logits - torch.log(exp.sum(1, keepdim=True) + 1e-6)
        return -(mask * log_prob).sum(1).mean()

class TitanDataset(Dataset):
    def __init__(self, df, audio_dir, augment=False):
        self.df = df
        self.audio_dir = Path(audio_dir)
        self.max_len = 160000 
        self.aug = Compose([AddGaussianNoise(p=0.3), PitchShift(p=0.4), TimeStretch(p=0.2)]) if augment else None
        
        # --- NEW: CACHE ACTUAL PATHS ---
        print(f"[INIT] Scanning {audio_dir} for audio files...")
        self.path_map = {f.name: f for f in self.audio_dir.rglob("*.wav")}
        print(f"[INIT] Found {len(self.path_map)} valid audio files.")

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        # Safe loop instead of recursion to prevent RecursionError
        for _ in range(len(self.df)):
            row = self.df.iloc[idx]
            fname = Path(row["audio_relpath"]).name
            
            if fname in self.path_map:
                p = self.path_map[fname]
                try:
                    wav, _ = librosa.load(p, sr=16000)
                    if self.aug: wav = self.aug(samples=wav, sample_rate=16000)
                    mask = np.zeros(self.max_len, dtype=np.float32)
                    if len(wav) > self.max_len: 
                        wav = wav[:self.max_len]; mask[:] = 1.0
                    else: 
                        mask[:len(wav)] = 1.0; wav = np.pad(wav, (0, self.max_len - len(wav)))
                    return {"wav": torch.tensor(wav), "mask": torch.tensor(mask), "label": torch.tensor(row["lid"])}
                except Exception as e:
                    print(f"[WARN] Failed to load {p}: {e}")
            
            # If file missing or load failed, try next index
            idx = (idx + 1) % len(self.df)
            
        raise FileNotFoundError(f"Could not find any valid audio files in {self.audio_dir}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- AUTOMATIC PATH RESOLUTION ---
    if IS_COLAB:
        # 1. Search for CSVs
        default_csv = Path("/content/drive/MyDrive/Thesis Project/data/processed/splits/text_hc")
        if (default_csv / "train.csv").exists():
            csv_dir = default_csv
        else:
            print("[DEBUG] Searching Drive for train.csv...")
            csv_dir = None
            for root, dirs, files in os.walk('/content/drive/MyDrive'):
                if "train.csv" in files and "text_hc" in root:
                    csv_dir = Path(root); break
            if not csv_dir: csv_dir = Path("/content")

        # 2. Search for Audio Dir (Handles nested zip extractions)
        print("[DEBUG] Locating audio dataset directory root...")
        if os.path.exists("/content/dataset"):
            audio_dir = "/content/dataset"
        elif os.path.exists("/content/Thesis Project/dataset"):
            audio_dir = "/content/Thesis Project/dataset"
        else:
            audio_dir = "/content"
            for item in os.listdir('/content'):
                item_path = os.path.join('/content', item)
                if os.path.isdir(item_path) and item != 'drive' and item != 'sample_data':
                    # If this top-level folder contains any WAVs, it is our root
                    contains_wavs = False
                    for root, dirs, files in os.walk(item_path):
                        if any(f.endswith('.wav') for f in files):
                            contains_wavs = True; break
                    if contains_wavs:
                        audio_dir = item_path
                        print(f"[DEBUG] Found dataset root at: {audio_dir}")
                        break
    else:
        csv_dir = Path("D:/Thesis Project/data/processed/splits/audio_eligible")
        audio_dir = "D:/Thesis Project/dataset"

    if not (csv_dir / "train.csv").exists():
        print(f"[FATAL] Could not find train.csv in {csv_dir}")
        return

    print(f"[INIT] Reading CSVs from {csv_dir}...")
    tr_df = pd.read_csv(csv_dir / "train.csv")
    va_df = pd.read_csv(csv_dir / "val.csv")
    te_df = pd.read_csv(csv_dir / "test.csv")
    
    classes = sorted(tr_df["emotion_final"].unique())
    lid = {l: i for i, l in enumerate(classes)}
    for df in [tr_df, va_df, te_df]: df["lid"] = df["emotion_final"].map(lid)
    
    BATCH = 2 if IS_COLAB else 2
    ACCUM = 8 if IS_COLAB else 8
    EPOCHS = 15; ALPHA = 0.3
    
    loader_args = {"batch_size": BATCH, "num_workers": 2 if IS_COLAB else 0, "pin_memory": True}
    tr_loader = DataLoader(TitanDataset(tr_df, audio_dir, True), shuffle=True, **loader_args)
    va_loader = DataLoader(TitanDataset(va_df, audio_dir), **loader_args)
    te_loader = DataLoader(TitanDataset(te_df, audio_dir), **loader_args)
    
    model = TitanAudioSER(len(classes)).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.wavlm.parameters(), "lr": 2e-5},
        {"params": [p for n, p in model.named_parameters() if "wavlm" not in n], "lr": 1e-3}
    ], weight_decay=0.01)
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, len(tr_loader), len(tr_loader)*EPOCHS)
    l_ce = nn.CrossEntropyLoss(label_smoothing=0.1); l_sc = SupConLoss(); scaler = GradScaler('cuda')

    print(f"[START] Training Titan SOTA. Environment: {'Colab' if IS_COLAB else 'Local'}")
    best_acc = 0
    for ep in range(1, EPOCHS + 1):
        model.train(); t_loss = 0
        optimizer.zero_grad()
        pbar = tqdm(tr_loader, desc=f"Epoch {ep}")
        for b in pbar:
            w, m, l = b["wav"].to(device), b["mask"].to(device), b["label"].to(device)
            with autocast('cuda'):
                logits, z = model(w, m)
                loss = ((1-ALPHA)*l_ce(logits, l) + ALPHA*l_sc(z, l)) / ACCUM
            
            scaler.scale(loss).backward()
            
            if (pbar.n + 1) % ACCUM == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            t_loss += loss.item() * ACCUM
            pbar.set_postfix({"loss": f"{t_loss/(pbar.n+1):.4f}"})
        
        # Eval
        val_acc, _ = evaluate(model, va_loader, device)
        test_acc, _ = evaluate(model, te_loader, device)
        print(f"Epoch {ep} | Val: {val_acc:.3f} | Test: {test_acc:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "titan_sota.pt")
            print("New Best Saved.")

def evaluate(model, loader, device):
    model.eval(); ps, ts = [], []
    with torch.no_grad():
        for b in loader:
            l, _ = model(b["wav"].to(device), b["mask"].to(device))
            ps.extend(torch.argmax(l, 1).cpu().numpy()); ts.extend(b["label"].numpy())
    return accuracy_score(ts, ps), f1_score(ts, ps, average="macro")

if __name__ == "__main__":
    main()
