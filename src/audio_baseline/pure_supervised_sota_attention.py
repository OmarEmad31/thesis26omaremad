"""
HArnESS-Pure-Supervised-SOTA: Egyptian Arabic SER (57% Accuracy Push).
Maximum Information Density implementation for small datasets.

Backbone: WavLM-Large (315M parameters, Prosody-Native).
Pooling: Attentive Pooling (Global Emotion Spike Detection).
Head: Transformer Encoder Layer (Rhythmic Dependency Modeling).
Augmentation: Audiomentations (Online Diversity Pipeline).
"""

import os, sys, torch, librosa, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import WavLMModel, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from pathlib import Path
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

# Add project root
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.audio_baseline import config

# ---------------------------------------------------------------------------
# ARCHITECTURE: ATTENTIVE POOLING & TRANSFORMER HEAD
# ---------------------------------------------------------------------------
class AttentivePooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [BS, Seq, Hidden]
        weights = torch.softmax(self.attention(x), dim=1) # [BS, Seq, 1]
        pooled = torch.sum(x * weights, dim=1) # [BS, Hidden]
        return pooled

class WavLMAttentionSOTA(nn.Module):
    def __init__(self, num_labels, wavlm_name="microsoft/wavlm-large"):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained(wavlm_name)
        self.wavlm.config.output_hidden_states = True
        
        # Stability: LayerNorm for the backbone output
        self.backbone_norm = nn.LayerNorm(1024)
        
        # Temporal Mastery: Self-Attention Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=8, dim_feedforward=2048, batch_first=True, dropout=0.2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # Global Emotion Selection
        self.pooling = AttentivePooling(1024)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_labels)
        )

    def forward(self, wav, mask):
        # WavLM-Large output: [BS, Seq, 1024]
        outputs = self.wavlm(wav, attention_mask=mask).last_hidden_state
        outputs = self.backbone_norm(outputs)
        
        # Transformer-based temporal modeling
        encoded = self.transformer_encoder(outputs) # [BS, Seq, 1024]
        
        # Select the most emotionally salient frames
        pooled = self.pooling(encoded) # [BS, 1024]
        
        # Final Classification
        logits = self.classifier(pooled)
        return logits

# ---------------------------------------------------------------------------
# AUGMENTED DATASET: SYNTHETIC VOLUME EXPANSION
# ---------------------------------------------------------------------------
class AugmentedEgyptianDataset(Dataset):
    def __init__(self, df, audio_map, augment=False):
        self.df = df
        self.audio_map = audio_map
        self.max_len = 160000 # 10s @ 16kHz
        self.augment = augment
        
        # Precision Augmentation Pipeline
        if self.augment:
            self.augmentor = Compose([
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                TimeStretch(min_rate=0.8, max_rate=1.2, p=0.4),
                PitchShift(min_semitones=-4, max_semitones=4, p=0.4),
                Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
            ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        bn = Path(str(row["audio_relpath"]).replace("\\", "/")).name
        path = self.audio_map.get(bn)
        
        if path is None:
            raise FileNotFoundError(f"Missing: {bn}")
            
        audio, _ = librosa.load(path, sr=16000, mono=True)
        
        # Apply Augmentation if enabled
        if self.augment:
            audio = self.augmentor(samples=audio, sample_rate=16000)

        # Padding/Clipping
        if len(audio) > self.max_len: audio = audio[:self.max_len]
        else: audio = np.pad(audio, (0, self.max_len - len(audio)))
        
        return {
            "wav": torch.tensor(audio, dtype=torch.float32),
            "label": torch.tensor(row["label_id"], dtype=torch.long)
        }

# ---------------------------------------------------------------------------
# SOTA TRAINING: THE 57% PUSH
# ---------------------------------------------------------------------------
def evaluate(model, loader, device):
    model.eval()
    ps, ts = [], []
    with torch.no_grad():
        for b in loader:
            l = model(b["wav"].to(device), None)
            ps.extend(torch.argmax(l, 1).cpu().numpy()); ts.extend(b["label"].numpy())
    return accuracy_score(ts, ps), f1_score(ts, ps, average="macro")

def main():
    torch.manual_seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    W_NAME = "microsoft/wavlm-large"

    print("🏗️ Initializing SOTA Attention Engine...")
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
    samples_weight = torch.from_numpy(np.array([1.0/np.bincount(y_tr)[t] for t in y_tr]))
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

    model = WavLMAttentionSOTA(len(classes), W_NAME).to(device)
    
    # Label Smoothing (0.1) for better generalization on small data
    l_ce = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    
    # Batch Size: 8 for WavLM-Large (Colab Pro can handle it safely)
    tr_loader = DataLoader(AugmentedEgyptianDataset(train_df, audio_map, augment=True), batch_size=8, sampler=sampler)
    va_loader = DataLoader(AugmentedEgyptianDataset(val_df, audio_map), batch_size=16)
    te_loader = DataLoader(AugmentedEgyptianDataset(test_df, audio_map), batch_size=16)

    # STAGE 1: HEAD WARMUP (2 Epochs)
    print("\n🔥 STAGE 1: HEAD WARMUP (BACKBONE FROZEN)")
    for param in model.wavlm.parameters(): param.requires_grad = False
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    for epoch in range(1, 3):
        model.train()
        for b in tqdm(tr_loader, desc=f"Warmup Ep{epoch}"):
            wav = b["wav"].to(device); l = b["label"].to(device)
            logits = model(wav, None)
            loss = l_ce(logits, l)
            opt.zero_grad(); loss.backward(); opt.step()
        va, vf = evaluate(model, va_loader, device)
        print(f"📈 Warmup Epoch {epoch} | Val Acc: {va:.3f} | Val F1: {vf:.3f}")

    # STAGE 2: DEEP FINE-TUNING (15 Epochs)
    print("\n🚀 STAGE 2: SOTA FINE-TUNING (UNFROZEN)")
    for param in model.wavlm.parameters(): param.requires_grad = True
    
    # Discriminative LR: Ultralow for Backbone, Standard for Head
    opt = torch.optim.AdamW([
        {"params": model.wavlm.parameters(), "lr": 2e-6},
        {"params": model.transformer_encoder.parameters(), "lr": 1e-4},
        {"params": model.pooling.parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 1e-4}
    ], weight_decay=0.01)
    
    sch = get_cosine_schedule_with_warmup(opt, num_warmup_steps=len(tr_loader)*2, num_training_steps=len(tr_loader)*15)
    
    best_va = 0
    for epoch in range(1, 16):
        model.train()
        for b in tqdm(tr_loader, desc=f"SOTA Ep{epoch}"):
            wav = b["wav"].to(device); l = b["label"].to(device)
            logits = model(wav, None)
            loss = l_ce(logits, l)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            
        va, vf = evaluate(model, va_loader, device)
        ta, tf = evaluate(model, te_loader, device)
        print(f"🏆 Epoch {epoch} Results:")
        print(f"   VALIDATION -> Acc: {va:.3f} | Macro-F1: {vf:.3f}")
        print(f"   TESTING    -> Acc: {ta:.3f} | Macro-F1: {tf:.3f}")
        
        if va > best_va:
            best_va = va
            torch.save(model.state_dict(), config.CHECKPOINT_DIR / "wavlm_sota_attention.pt")
            print("   🌟 New Best Validation Accuracy Saved")

if __name__ == "__main__": main()
