"""
HArnESS-SOTA-Flagship: Egyptian Arabic SER (The 57% Milestone Push).
Final high-capacity supervised implementation.

Backbone: WavLM-Large (315M parameters).
Pooling: Attentive Masked Pooling (Focus on emotional spikes).
Balance: WeightedRandomSampler + WeightedCrossEntropyLoss.
Stability: Input Normalization + Discriminative Fine-Tuning.
"""

import os, sys, torch, librosa, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import WavLMModel, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from pathlib import Path
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

# Add project root
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.audio_baseline import config

# ---------------------------------------------------------------------------
# ARCHITECTURE: ATTENTIVE MASKED POOLING
# ---------------------------------------------------------------------------
class AttentivePooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask=None):
        # x: [BS, Seq, Hidden], mask: [BS, Seq]
        attn_logits = self.attention(x) # [BS, Seq, 1]
        
        if mask is not None:
            mask = mask.unsqueeze(-1) # [BS, Seq, 1]
            attn_logits = attn_logits.masked_fill(mask == 0, -1e9)
            
        weights = torch.softmax(attn_logits, dim=1) # [BS, Seq, 1]
        pooled = torch.sum(x * weights, dim=1) # [BS, Hidden]
        return pooled

class WavLMSOTAFlagship(nn.Module):
    def __init__(self, num_labels, wavlm_name="microsoft/wavlm-large"):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained(wavlm_name)
        
        # Stability: Multi-layer Head for High-Param model
        self.pooling = AttentivePooling(1024)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        )

    def forward(self, wav, mask=None):
        # 1. Normalization (Crucial for attention stability)
        wav = (wav - wav.mean(dim=-1, keepdim=True)) / (wav.std(dim=-1, keepdim=True) + 1e-6)
        
        # 2. Backbone
        outputs = self.wavlm(wav).last_hidden_state
        
        # 3. Mask Alignment (Factor 320 for WavLM)
        if mask is not None:
            down_mask = mask[:, ::320][:, :outputs.shape[1]]
        else:
            down_mask = None
            
        # 4. Pooling & Classification
        pooled = self.pooling(outputs, mask=down_mask)
        logits = self.classifier(pooled)
        return logits

# ---------------------------------------------------------------------------
# DATASET: ONLINE AUGMENTATION PIPELINE
# ---------------------------------------------------------------------------
class SOTAAugmentedDataset(Dataset):
    def __init__(self, df, audio_map, augment=False):
        self.df = df
        self.audio_map = audio_map
        self.max_len = 160000 
        self.augment = augment
        
        if self.augment:
            self.augmentor = Compose([
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.4),
                TimeStretch(min_rate=0.8, max_rate=1.2, p=0.3),
                PitchShift(min_semitones=-4, max_semitones=4, p=0.4),
                Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
            ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        bn = Path(str(row["audio_relpath"]).replace("\\", "/")).name
        path = self.audio_map.get(bn)
        
        if path is None: raise FileNotFoundError(f"Missing: {bn}")
        audio, _ = librosa.load(path, sr=16000, mono=True)
        
        if self.augment:
            audio = self.augmentor(samples=audio, sample_rate=16000)

        mask = np.zeros(self.max_len, dtype=np.float32)
        if len(audio) > self.max_len:
            audio = audio[:self.max_len]
            mask[:] = 1.0
        else:
            mask[:len(audio)] = 1.0
            audio = np.pad(audio, (0, self.max_len - len(audio)))
        
        return {
            "wav": torch.tensor(audio, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "label": torch.tensor(row["label_id"], dtype=torch.long)
        }

# ---------------------------------------------------------------------------
# TRAINING SCRIPT
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
    W_NAME = "microsoft/wavlm-large"

    print("🏗️ Initializing SOTA Flagship Engine...")
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

    # BALANCE CALCULATION
    from sklearn.utils.class_weight import compute_class_weight
    y_tr = train_df["label_id"].values
    weights = torch.tensor(compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr), dtype=torch.float32).to(device)
    
    from torch.utils.data import WeightedRandomSampler
    class_sample_count = np.bincount(y_tr)
    weight = 1. / class_sample_count
    samples_weight = torch.from_numpy(np.array([weight[t] for t in y_tr]))
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

    model = WavLMSOTAFlagship(len(classes), W_NAME).to(device)
    l_ce = nn.CrossEntropyLoss(weight=weights) 
    
    # Batch Size 8 + 4-step Gradient Accumulation = Effective BS 32 for stability
    # BS 8 fits easily on any Colab GPU with the Large model
    tr_loader = DataLoader(SOTAAugmentedDataset(train_df, audio_map, augment=True), batch_size=8, sampler=sampler)
    va_loader = DataLoader(SOTAAugmentedDataset(val_df, audio_map), batch_size=8)
    te_loader = DataLoader(SOTAAugmentedDataset(test_df, audio_map), batch_size=8)
    accum_steps = 4

    print("\n🔥 STAGE 1: HEAD WARMUP (2 Epochs)")
    for param in model.wavlm.parameters(): param.requires_grad = False
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    for epoch in range(1, 3):
        model.train()
        pbar = tqdm(tr_loader, desc=f"Warmup Ep{epoch}")
        for i, b in enumerate(pbar):
            wav = b["wav"].to(device); l = b["label"].to(device); m = b["mask"].to(device)
            logits = model(wav, m)
            loss = l_ce(logits, l) / accum_steps
            loss.backward()
            if (i+1) % accum_steps == 0:
                opt.step(); opt.zero_grad()
            pbar.set_postfix({"loss": f"{loss.item()*accum_steps:.4f}"})
        va, vf = evaluate(model, va_loader, device)
        print(f"📈 Warmup Epoch {epoch} | Val Acc: {va:.3f} | Val F1: {vf:.3f}")

    print("\n🚀 STAGE 2: SOTA FINE-TUNING (15 Epochs)")
    for param in model.wavlm.parameters(): param.requires_grad = True
    # Discriminative LR for Backbone Preservation
    opt = torch.optim.AdamW([
        {"params": model.wavlm.parameters(), "lr": 2e-6},
        {"params": model.classifier.parameters(), "lr": 1e-4}
    ], weight_decay=0.01)
    
    sch = get_cosine_schedule_with_warmup(opt, num_warmup_steps=len(tr_loader), num_training_steps=len(tr_loader)*15)
    
    best_va = 0
    for epoch in range(1, 16):
        model.train()
        pbar = tqdm(tr_loader, desc=f"SOTA Ep{epoch}")
        for i, b in enumerate(pbar):
            wav = b["wav"].to(device); l = b["label"].to(device); m = b["mask"].to(device)
            logits = model(wav, m)
            loss = l_ce(logits, l) / accum_steps
            loss.backward()
            if (i+1) % accum_steps == 0:
                opt.step(); opt.zero_grad(); sch.step()
            pbar.set_postfix({"loss": f"{loss.item()*accum_steps:.4f}"})
            
        va, vf = evaluate(model, va_loader, device)
        ta, tf = evaluate(model, te_loader, device)
        print(f"🏆 Epoch {epoch} Results:")
        print(f"   VALIDATION -> Acc: {va:.3f} | Macro-F1: {vf:.3f}")
        print(f"   TESTING    -> Acc: {ta:.3f} | Macro-F1: {tf:.3f}")
        
        if va > best_va:
            best_va = va
            torch.save(model.state_dict(), config.CHECKPOINT_DIR / "wavlm_flagship_best.pt")
            print("   🌟 SOTA Best Model Saved")

if __name__ == "__main__": main()
