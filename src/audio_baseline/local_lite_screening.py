"""
HArnESS-Lite-Screening: The "Potential Check" Sprint.
Methodology:
- Run 3-5 epochs ONLY to check learning potential.
- Use 20% stratified subset to save thermal/time resources.
- Optimized for GTX 1650 (4GB) + Local Windows.
"""

import os, sys, torch, librosa, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import WavLMModel, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pathlib import Path
from torch.cuda.amp import GradScaler, autocast

# CRITICAL: Re-route the heavy models to the D: drive
os.environ["HF_HOME"] = "D:/HuggingFaceCache"

# Add project root
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.audio_baseline import config

class OmegaWLPWavLM(nn.Module):
    def __init__(self, num_labels, model_name="microsoft/wavlm-base-plus"):
        super().__init__()
        # Load WavLM from D: drive cache
        self.wavlm = WavLMModel.from_pretrained(model_name, output_hidden_states=True)
        self.layer_weights = nn.Parameter(torch.ones(13))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_labels)
        )

    def forward(self, wav, mask):
        wav = (wav - wav.mean(dim=-1, keepdim=True)) / (wav.std(dim=-1, keepdim=True) + 1e-6)
        outputs = self.wavlm(wav)
        hidden_states = torch.stack(outputs.hidden_states, dim=0)
        w = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        weighted_hidden = (hidden_states * w).sum(dim=0)
        
        down_mask = mask[:, ::320][:, :weighted_hidden.shape[1]]
        mask_exp = down_mask.unsqueeze(-1).expand(weighted_hidden.size()).float()
        pooled = torch.sum(weighted_hidden * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)
        return self.classifier(pooled)

class MasterPoolDataset(Dataset):
    def __init__(self, df, audio_map, augment=False):
        self.df = df; self.audio_map = audio_map; self.max_len = 160000; self.augment = augment
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]; bn = Path(str(row["audio_relpath"]).replace("\\", "/")).name
        path = self.audio_map.get(bn); audio, _ = librosa.load(path, sr=16000, mono=True)
        if self.augment and np.random.random() < 0.5:
            steps = np.random.uniform(-1, 1)
            audio = librosa.effects.pitch_shift(audio, sr=16000, n_steps=steps)
        mask = np.zeros(self.max_len, dtype=np.float32)
        if len(audio) > self.max_len: audio = audio[:self.max_len]; mask[:] = 1.0
        else: mask[:len(audio)] = 1.0; audio = np.pad(audio, (0, self.max_len - len(audio)))
        return {"wav": torch.tensor(audio, dtype=torch.float32), 
                "mask": torch.tensor(mask, dtype=torch.float32), 
                "label": torch.tensor(row["lid"], dtype=torch.long)}

def main():
    torch.manual_seed(42); np.random.seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INIT] Laptop Lite Screening Strategy (Target: {device})")
    
    # 1. LOAD SPLITS
    tr_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv")
    va_df = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv")
    
    classes = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
    lid = {l: i for i, l in enumerate(classes)}
    for df in [tr_df, va_df]: df["lid"] = df["emotion_final"].map(lid)
    
    audio_map = {}
    p_root = Path(config.DATA_ROOT).parent
    for ext in ["*.wav", "*.WAV", "*.Wav"]:
        for f in p_root.rglob(ext): audio_map[f.name] = f

    # 2. SUBSET SAMPLING (USER REQUEST: 20% ONLY)
    print(f"[LITE] Original Train Size: {len(tr_df)}")
    _, tr_lite_df = train_test_split(tr_df, test_size=0.2, stratify=tr_df["lid"], random_state=42)
    print(f"[LITE] Screening Subset Size: {len(tr_lite_df)}")

    # 3. SETTINGS FOR LOCAL 1650 (4GB)
    BATCH_SIZE = 2             # Conservative
    ACCUMULATION_STEPS = 8    # Effective Batch Size = 16
    MAX_EPOCHS = 3             # User requested "less epochs" for potential check
    
    tr_loader = DataLoader(MasterPoolDataset(tr_lite_df, audio_map, augment=True), batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(MasterPoolDataset(va_df, audio_map), batch_size=BATCH_SIZE)
    
    model = OmegaWLPWavLM(len(classes)).to(device)
    scaler = GradScaler()
    total_steps = len(tr_loader) * MAX_EPOCHS
    opt = torch.optim.AdamW([
        {"params": model.wavlm.parameters(), "lr": 1e-5},
        {"params": [p for n, p in model.named_parameters() if "wavlm" not in n], "lr": 5e-4},
    ], weight_decay=0.01)
    sch = get_cosine_schedule_with_warmup(opt, len(tr_loader), total_steps)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    print(f"[SCREENING] Starting {MAX_EPOCHS}-epoch fast sprint. Model cached on D: drive.")
    
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        opt.zero_grad()
        for i, b in enumerate(tr_loader):
            w, m, l = b["wav"].to(device), b["mask"].to(device), b["label"].to(device)
            with autocast():
                logits = model(w, m)
                loss = criterion(logits, l) / ACCUMULATION_STEPS
            scaler.scale(loss).backward()
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                sch.step()
        
        # Fast Validation
        model.eval(); ps, ts = [], []
        with torch.no_grad():
            for b in va_loader:
                out = model(b["wav"].to(device), b["mask"].to(device))
                ps.extend(torch.argmax(out, 1).cpu().numpy()); ts.extend(b["label"].numpy())
        acc = accuracy_score(ts, ps)
        print(f"📊 Sprint Epoch {epoch}/{MAX_EPOCHS} | Val Acc: {acc:.3f}")

    print("\n[DONE] Screening complete. Verify accuracy trend to assess potential.")

if __name__ == "__main__": main()
