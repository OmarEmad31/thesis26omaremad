"""
HArnESS-High-Density: Egyptian Arabic SER (3s Cropping Overhaul).
Maximizing emotional signal by eliminating padded silence.

Backbone: WavLM-Base-Plus (768 dim).
Pooling: Masked Mean Pooling (High-Density).
Protocol: 3s Random Cropping (Online Augmentation).
Balance: Full WeightedCrossEntropy.
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
# ARCHITECTURE: STABLE MASKED MEAN POOLING
# ---------------------------------------------------------------------------
class MaskedMeanPooling(nn.Module):
    def forward(self, x, mask):
        if mask is None: return x.mean(dim=1)
        mask_expanded = mask.unsqueeze(-1).expand(x.size()).float()
        sum_embeddings = torch.sum(x * mask_expanded, 1)
        sum_mask = mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_embeddings / sum_mask

class WavLMHighDensity(nn.Module):
    def __init__(self, num_labels, wavlm_name="microsoft/wavlm-base-plus"):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained(wavlm_name)
        self.pooling = MaskedMeanPooling()
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, num_labels)
        )

    def forward(self, wav, mask=None):
        wav = (wav - wav.mean(dim=-1, keepdim=True)) / (wav.std(dim=-1, keepdim=True) + 1e-6)
        outputs = self.wavlm(wav).last_hidden_state
        if mask is not None:
            down_mask = mask[:, ::320][:, :outputs.shape[1]]
        else:
            down_mask = None
        pooled = self.pooling(outputs, mask=down_mask)
        logits = self.classifier(pooled)
        return logits

# ---------------------------------------------------------------------------
# DATASET: RANDOM CROPPING ENGINE (3 SECONDS)
# ---------------------------------------------------------------------------
class CropEgyptianDataset(Dataset):
    def __init__(self, df, audio_map, is_train=False):
        self.df = df
        self.audio_map = audio_map
        self.crop_len = 48000 # 3s @ 16kHz
        self.is_train = is_train

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        bn = Path(str(row["audio_relpath"]).replace("\\", "/")).name
        path = self.audio_map.get(bn)
        if path is None: raise FileNotFoundError(f"Missing: {bn}")
        
        audio, _ = librosa.load(path, sr=16000, mono=True)
        
        # 3-Second Cropping Logic
        if len(audio) > self.crop_len:
            if self.is_train:
                start = np.random.randint(0, len(audio) - self.crop_len)
            else:
                start = (len(audio) - self.crop_len) // 2 # Center crop for Eval
            audio = audio[start:start+self.crop_len]
            mask = np.ones(self.crop_len, dtype=np.float32)
        else:
            mask = np.zeros(self.crop_len, dtype=np.float32)
            mask[:len(audio)] = 1.0
            audio = np.pad(audio, (0, self.crop_len - len(audio)))
        
        return {
            "wav": torch.tensor(audio, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "label": torch.tensor(row["label_id"], dtype=torch.long)
        }

# ---------------------------------------------------------------------------
# EVAL & TRAINING
# ---------------------------------------------------------------------------
def evaluate(model, loader, device):
    model.eval()
    ps, ts = [], []
    with torch.no_grad():
        for b in loader:
            l = model(b["wav"].to(device), b["mask"].to(device))
            ps.extend(torch.argmax(l, 1).cpu().numpy()); ts.extend(b["label"].numpy())
    return accuracy_score(ts, ps), f1_score(ts, ps, average="macro")

def main():
    torch.manual_seed(42); np.random.seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print("🏗️ Initializing High-Density Crop Engine...")
    train_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv")
    val_df   = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv")
    test_df  = pd.read_csv(config.SPLIT_CSV_DIR / "test.csv")
    
    classes = sorted(train_df["emotion_final"].unique())
    label2id = {l: i for i, l in enumerate(classes)}
    for df in [train_df, val_df, test_df]: df["label_id"] = df["emotion_final"].map(label2id)
    
    audio_map = {}
    data_search_path = Path("/content/dataset") if Path("/content/dataset").exists() else Path(config.DATA_ROOT).parent
    for ext in ["*.wav", "*.WAV", "*.Wav"]:
        for p in data_search_path.rglob(ext): audio_map[p.name] = p

    # FULL BALANCED WEIGHTS
    y_tr = train_df["label_id"].values
    weights = torch.tensor(compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr), dtype=torch.float32).to(device)

    model = WavLMHighDensity(len(classes)).to(device)
    l_ce = nn.CrossEntropyLoss(weight=weights) 
    
    tr_loader = DataLoader(CropEgyptianDataset(train_df, audio_map, is_train=True), batch_size=32, shuffle=True)
    va_loader = DataLoader(CropEgyptianDataset(val_df, audio_map), batch_size=32)
    te_loader = DataLoader(CropEgyptianDataset(test_df, audio_map), batch_size=32)

    print("\n🔥 STAGE 1: HEAD WARMUP (3s CROPPING)")
    for param in model.wavlm.parameters(): param.requires_grad = False
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    for epoch in range(1, 4):
        model.train()
        pbar = tqdm(tr_loader, desc=f"Warmup Ep{epoch}")
        for b in pbar:
            wav = b["wav"].to(device); l = b["label"].to(device); m = b["mask"].to(device)
            logits = model(wav, m)
            loss = l_ce(logits, l)
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        va, vf = evaluate(model, va_loader, device)
        print(f"📈 Warmup Epoch {epoch} | Val Acc: {va:.3f} | Val F1: {vf:.3f}")

    print("\n🚀 STAGE 2: DISCRIMINATIVE FINE-TUNING")
    for param in model.wavlm.parameters(): param.requires_grad = True
    opt = torch.optim.AdamW([
        {"params": model.wavlm.parameters(), "lr": 1e-6},
        {"params": model.classifier.parameters(), "lr": 1e-4}
    ], weight_decay=0.01)
    
    sch = get_cosine_schedule_with_warmup(opt, num_warmup_steps=len(tr_loader), num_training_steps=len(tr_loader)*15)
    
    best_va = 0
    for epoch in range(1, 16):
        model.train()
        pbar = tqdm(tr_loader, desc=f"Fine-tune Ep{epoch}")
        for b in pbar:
            wav = b["wav"].to(device); l = b["label"].to(device); m = b["mask"].to(device)
            logits = model(wav, m)
            loss = l_ce(logits, l)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        va, vf = evaluate(model, va_loader, device)
        ta, tf = evaluate(model, te_loader, device)
        print(f"🏆 Epoch {epoch} Results:")
        print(f"   VALIDATION -> Acc: {va:.3f} | Macro-F1: {vf:.3f}")
        print(f"   TESTING    -> Acc: {ta:.3f} | Macro-F1: {tf:.3f}")
        
        if va > best_va:
            best_va = va
            torch.save(model.state_dict(), config.CHECKPOINT_DIR / "wavlm_crop_sota.pt")
            print("   🌟 New Best SOTA Saved")

if __name__ == "__main__": main()
