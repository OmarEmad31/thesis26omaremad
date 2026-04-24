"""
HArnESS-Local-Rescue: 4GB GPU Efficiency Champion.
Features: 
- Gradient Accumulation (Simulates Batch 16 on Batch 2 hardware).
- Automatic Mixed Precision (AMP) for 50% memory savings.
- Single-Fold high-intensity training.
"""

import os, sys, torch, librosa, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import WavLMModel, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
from torch.cuda.amp import GradScaler, autocast

# Add project root
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.audio_baseline import config

class OmegaWLPWavLM(nn.Module):
    def __init__(self, num_labels, model_name="microsoft/wavlm-base-plus"):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained(model_name, output_hidden_states=True)
        self.layer_weights = nn.Parameter(torch.ones(13))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.AvgPool1d(1), # Placeholder for future expansion
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
    torch.manual_seed(42); np.random.seed(42); device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("[INIT] Local 4GB Rescue Mission Started...")
    
    tr_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv")
    va_df = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv")
    
    classes = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
    lid = {l: i for i, l in enumerate(classes)}
    for df in [tr_df, va_df]: df["lid"] = df["emotion_final"].map(lid)
    
    audio_map = {}
    p_root = Path(config.DATA_ROOT).parent
    for ext in ["*.wav", "*.WAV", "*.Wav"]:
        for f in p_root.rglob(ext): audio_map[f.name] = f

    # SETTINGS FOR 4GB GPU
    BATCH_SIZE = 2             # Fit in memory
    ACCUMULATION_STEPS = 8    # 2 * 8 = 16 (Effective Batch Size)
    
    tr_loader = DataLoader(MasterPoolDataset(tr_df, audio_map, augment=True), batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(MasterPoolDataset(va_df, audio_map), batch_size=BATCH_SIZE)
    
    model = OmegaWLPWavLM(len(classes)).to(device)
    scaler = GradScaler()
    total_steps = len(tr_loader) * 20
    opt = torch.optim.AdamW([
        {"params": model.wavlm.parameters(), "lr": 1e-5},
        {"params": [p for n, p in model.named_parameters() if "wavlm" not in n], "lr": 5e-4},
    ], weight_decay=0.01)
    sch = get_cosine_schedule_with_warmup(opt, len(tr_loader), total_steps)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_acc = 0
    print(f"[STATUS] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print(f"[STATUS] Effective Batch Size: {BATCH_SIZE * ACCUMULATION_STEPS}")
    
    for epoch in range(1, 21):
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
        
        # Validation
        model.eval(); ps, ts = [], []
        with torch.no_grad():
            for b in va_loader:
                out = model(b["wav"].to(device), b["mask"].to(device))
                ps.extend(torch.argmax(out, 1).cpu().numpy()); ts.extend(b["label"].numpy())
        acc = accuracy_score(ts, ps)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), config.CHECKPOINT_DIR / "local_rescue_best.pt")
        
        print(f"Epoch {epoch}/20 | Val Acc: {acc:.3f} | Best: {best_acc:.3f}")

    print("\n[DONE] Local rescue training complete.")

if __name__ == "__main__": main()
