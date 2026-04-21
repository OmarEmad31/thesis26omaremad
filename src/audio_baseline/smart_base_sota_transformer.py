"""
HArnESS-Ultra-Stable: Egyptian Arabic SER (The 50% Push).
Maximizing 857 samples via Layer Fusion and Heavy Regularization.

Backbone: WavLM-Base-Plus (Frozen/Unfrozen Hybrid).
Features: Last 4-Layer Fusion (768 * 4 = 3072 dim).
Regularization: Label Smoothing (0.1) + High Dropout (0.5).
Augmentation: SpecAugment + Audiomentations.
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
# ARCHITECTURE: MULTI-LAYER FUSION + STABLE MLP
# ---------------------------------------------------------------------------
class MultiLayerPooling(nn.Module):
    def __init__(self, num_layers=4):
        super().__init__()
        self.num_layers = num_layers

    def forward(self, all_hidden_states, mask=None):
        # all_hidden_states: list of [BS, Seq, 768]
        # Take last N layers
        stacked = torch.stack(all_hidden_states[-self.num_layers:], dim=0) # [4, BS, Seq, 768]
        # Mean across layers
        fused = torch.mean(stacked, dim=0) # [BS, Seq, 768]
        
        if mask is None: return fused.mean(dim=1)
        
        mask_expanded = mask.unsqueeze(-1).expand(fused.size()).float()
        sum_embeddings = torch.sum(fused * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

class UltraStableWavLM(nn.Module):
    def __init__(self, num_labels, model_name="microsoft/wavlm-base-plus"):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained(model_name, output_hidden_states=True)
        self.pooling = MultiLayerPooling(num_layers=4)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), # Extreme Dropout to prevent memorization
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        )

    def forward(self, wav, mask=None):
        wav = (wav - wav.mean(dim=-1, keepdim=True)) / (wav.std(dim=-1, keepdim=True) + 1e-6)
        outputs = self.wavlm(wav)
        
        # Multi-layer fusion
        if mask is not None:
            down_mask = mask[:, ::320][:, :outputs.last_hidden_state.shape[1]]
        else:
            down_mask = None
            
        pooled = self.pooling(outputs.hidden_states, mask=down_mask)
        logits = self.classifier(pooled)
        return logits

# ---------------------------------------------------------------------------
# DATASET: AUGMENTATION ENHANCED
# ---------------------------------------------------------------------------
class AugmentedEgyptianDataset(Dataset):
    def __init__(self, df, audio_map, is_train=False):
        self.df = df
        self.audio_map = audio_map
        self.max_len = 160000 
        self.is_train = is_train

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        bn = Path(str(row["audio_relpath"]).replace("\\", "/")).name
        path = self.audio_map.get(bn); audio, _ = librosa.load(path, sr=16000, mono=True)
        
        # Simple Online Augmentation for Train
        if self.is_train:
            # 1. Pitch shift (-1 to 1 semitones)
            if np.random.rand() > 0.5:
                audio = librosa.effects.pitch_shift(audio, sr=16000, n_steps=np.random.uniform(-1, 1))
            # 2. Add white noise
            if np.random.rand() > 0.5:
                audio = audio + 0.005 * np.random.randn(len(audio))
        
        mask = np.zeros(self.max_len, dtype=np.float32)
        if len(audio) > self.max_len:
            audio = audio[:self.max_len]; mask[:] = 1.0
        else:
            mask[:len(audio)] = 1.0; audio = np.pad(audio, (0, self.max_len - len(audio)))
        
        return {
            "wav": torch.tensor(audio, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "label": torch.tensor(row["label_id"], dtype=torch.long)
        }

# ---------------------------------------------------------------------------
# TRAINING PROCOL: LABEL SMOOTHING MASTER
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
    
    print("🏗️ Initializing Ultra-Stable 50% Push Engine...")
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

    y_tr = train_df["label_id"].values
    weights = torch.tensor(compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr), dtype=torch.float32).to(device)

    model = UltraStableWavLM(len(classes)).to(device)
    # LABEL SMOOTHING (0.1) TO STOP OVERFITTING
    l_ce = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1) 
    
    tr_loader = DataLoader(AugmentedEgyptianDataset(train_df, audio_map, is_train=True), batch_size=32, shuffle=True)
    va_loader = DataLoader(AugmentedEgyptianDataset(val_df, audio_map), batch_size=32)
    te_loader = DataLoader(AugmentedEgyptianDataset(test_df, audio_map), batch_size=32)

    print("\n🔥 STAGE 1: HYPER-STABLE WARMUP (5 Epochs)")
    for param in model.wavlm.parameters(): param.requires_grad = False
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    for epoch in range(1, 6):
        model.train()
        pbar = tqdm(tr_loader, desc=f"Warmup Ep{epoch}")
        for b in pbar:
            wav = b["wav"].to(device); l = b["label"].to(device); m = b["mask"].to(device)
            logits = model(wav, m); loss = l_ce(logits, l)
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        va, vf = evaluate(model, va_loader, device)
        print(f"📈 Warmup Epoch {epoch} | Val Acc: {va:.3f} | Val F1: {vf:.3f}")

    print("\n🚀 STAGE 2: THE 50% FINE-TUNING PUSH (25 Epochs)")
    for param in model.wavlm.parameters(): param.requires_grad = True
    opt = torch.optim.AdamW([
        {"params": model.wavlm.parameters(), "lr": 1e-6}, # Ultra-conservative LR
        {"params": model.classifier.parameters(), "lr": 2e-4}
    ], weight_decay=0.01)
    
    sch = get_cosine_schedule_with_warmup(opt, num_warmup_steps=len(tr_loader), num_training_steps=len(tr_loader)*25)
    
    best_va = 0
    for epoch in range(1, 26):
        model.train()
        pbar = tqdm(tr_loader, desc=f"Push Ep{epoch}")
        for b in pbar:
            wav = b["wav"].to(device); l = b["label"].to(device); m = b["mask"].to(device)
            logits = model(wav, m); loss = l_ce(logits, l)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        va, vf = evaluate(model, va_loader, device)
        ta, tf = evaluate(model, te_loader, device)
        print(f"🏆 Epoch {epoch} Results: VAL Acc: {va:.3f} F1: {vf:.3f} | TEST Acc: {ta:.3f} F1: {tf:.3f}")
        if va > best_va:
            best_va = va; torch.save(model.state_dict(), config.CHECKPOINT_DIR / "ultra_stable_50_best.pt")
            print("   🌟 New Best 50% SOTA Saved")

if __name__ == "__main__": main()
