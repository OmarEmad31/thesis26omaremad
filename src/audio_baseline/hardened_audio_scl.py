"""
HArnESS-Hardened-SOTA: Egyptian Arabic Audio Emotion Recognition.
Senior Research Scientist Implementation (Final Stabilization).

Backbone: WavLM-Base-Plus (12-layer Denoising Speech Transformer).
Temporal Head: 2-layer Bidirectional GRU (Temporal Energy Modeling).
Fusion Strategy: Triple-Feature (Mel + MFCC + Chroma) with BatchNorm Scaling.
Geometry: Supervised Contrastive Learning (SupCon) for Latent Separability.
"""

import os, sys, random, torch, librosa, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoProcessor, WavLMModel, get_cosine_schedule_with_warmup
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
# RESEARCH UNIT: SUPERVISED CONTRASTIVE LOSS
# ---------------------------------------------------------------------------
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: [BS, Dim] (Must be L2-normalized)
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Compute similarity
        logits = torch.div(torch.matmul(features, features.T), self.temperature)
        
        # Stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # Mask self-contrast
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        # Log prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # Mean likelihood over positives
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        return -mean_log_prob_pos.mean()

# ---------------------------------------------------------------------------
# CORE SOTA ENGINE: THE HARDENED MODEL
# ---------------------------------------------------------------------------
class HardenedAudioSOTA(nn.Module):
    def __init__(self, num_labels, wavlm_name):
        super().__init__()
        # 1. Backbone: WavLM-Base-Plus (12 layers)
        self.wavlm = WavLMModel.from_pretrained(wavlm_name)
        self.wavlm.config.output_hidden_states = True
        
        # 2. Feature Normalizers (To fix the 5.8% Scaling Collapse)
        self.mel_norm = nn.BatchNorm1d(128)
        self.mfcc_norm = nn.BatchNorm1d(40)
        self.chroma_norm = nn.BatchNorm1d(12)
        
        # 3. Temporal Mastery: Bidirectional GRU
        # Combined Feature Dim = 768 (WavLM) + 128 (Mel) + 40 (MFCC) + 12 (Chroma) = 948
        self.feature_proj = nn.Linear(948, 512)
        self.gru = nn.GRU(input_size=512, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)
        
        # 4. SCL Projector (Geometric Alignment)
        self.scl_projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        
        # 5. Classifier (Final Logic)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_labels)
        )

    def forward(self, wav, mask, mel, mfcc, chroma):
        # Audio Backbone
        w_outs = self.wavlm(wav, attention_mask=mask).last_hidden_state # [BS, SeqA, 768]
        
        # Normalize Handcrafted Features [BS, SeqA, Dim]
        # Transpose for BatchNorm1d [BS, Dim, SeqA]
        mel = self.mel_norm(mel.transpose(1, 2)).transpose(1, 2)
        mfcc = self.mfcc_norm(mfcc.transpose(1, 2)).transpose(1, 2)
        chroma = self.chroma_norm(chroma.transpose(1, 2)).transpose(1, 2)
        
        # Concatenate and Project
        fused = torch.cat([w_outs, mel, mfcc, chroma], dim=-1) # [BS, SeqA, 948]
        projected = self.feature_proj(fused)
        
        # Temporal Processing
        gru_out, _ = self.gru(projected) # [BS, SeqA, 512]
        
        # Global Average Pooling across time
        pooled = gru_out.mean(dim=1)
        
        # SCL Latent Space (L2-Normalized)
        z = F.normalize(self.scl_projector(pooled), p=2, dim=1)
        
        # Class Logits
        logits = self.classifier(pooled)
        return logits, z

# ---------------------------------------------------------------------------
# DATASET & STABILIZED LOADER
# ---------------------------------------------------------------------------
class HardenedDataset(Dataset):
    def __init__(self, df, audio_map, augment=False):
        self.df = df
        self.audio_map = audio_map
        self.max_len = 160000 
        self.augment = augment

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        bn = Path(str(row["audio_relpath"]).replace("\\", "/")).name
        path = self.audio_map.get(bn)
        
        audio, _ = librosa.load(path, sr=16000, mono=True)
        if len(audio) > self.max_len: audio = audio[:self.max_len]
        else: audio = np.pad(audio, (0, self.max_len - len(audio)))

        # Handcrafted Features (Aligned to WavLM SeqLength ~499)
        hop = 320; n_fft = 1024
        mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=128, n_fft=n_fft, hop_length=hop)
        mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40, n_fft=n_fft, hop_length=hop)
        chroma = librosa.feature.chroma_stft(y=audio, sr=16000, n_fft=n_fft, hop_length=hop)
        
        mel = torch.tensor(librosa.power_to_db(mel).T[:499], dtype=torch.float32)
        mfcc = torch.tensor(mfcc.T[:499], dtype=torch.float32)
        chroma = torch.tensor(chroma.T[:499], dtype=torch.float32)
        
        return {
            "wav": torch.tensor(audio, dtype=torch.float32),
            "mel": mel, "mfcc": mfcc, "chroma": chroma,
            "label": torch.tensor(row["label_id"], dtype=torch.long)
        }

# ---------------------------------------------------------------------------
# HARDENED TRAINING PROTOCOL
# ---------------------------------------------------------------------------
def evaluate(model, loader, device):
    model.eval()
    ps, ts = [], []
    with torch.no_grad():
        for b in loader:
            l, _ = model(b["wav"].to(device), None, b["mel"].to(device), b["mfcc"].to(device), b["chroma"].to(device))
            ps.extend(torch.argmax(l, 1).cpu().numpy()); ts.extend(b["label"].numpy())
    return accuracy_score(ts, ps), f1_score(ts, ps, average="macro")

def train_epoch(model, loader, device, opt, sch, epoch, l_ce, l_sc, alpha):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f"Ep{epoch}", leave=False)
    for b in pbar:
        w = b["wav"].to(device); l = b["label"].to(device)
        me, mf, ch = b["mel"].to(device), b["mfcc"].to(device), b["chroma"].to(device)
        
        logits, z = model(w, None, me, mf, ch)
        loss = (1-alpha) * l_ce(logits, l) + alpha * l_sc(z, l)
        
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def main():
    torch.manual_seed(42); np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W_NAME = "microsoft/wavlm-base-plus"

    print("📂 Initializing Pristine Lab Environment...")
    train_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv")
    val_df   = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv")
    test_df  = pd.read_csv(config.SPLIT_CSV_DIR / "test.csv")
    
    classes = sorted(train_df["emotion_final"].unique())
    label2id = {l: i for i, l in enumerate(classes)}
    for df in [train_df, val_df, test_df]: df["label_id"] = df["emotion_final"].map(label2id)
    
    audio_map = {}
    src = Path("/content") if Path("/content").exists() else Path(config.DATA_ROOT).parent
    for p in src.rglob("*.wav"): audio_map[p.name] = p

    y_tr = train_df["label_id"].values
    weights = torch.tensor(compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr), dtype=torch.float32).to(device)
    samples_weight = torch.from_numpy(np.array([1.0/np.bincount(y_tr)[t] for t in y_tr]))
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

    model = HardenedAudioSOTA(len(classes), W_NAME).to(device)
    
    l_ce = nn.CrossEntropyLoss(weight=weights)
    l_sc = SupConLoss(temperature=0.1)
    
    tr_loader = DataLoader(HardenedDataset(train_df, audio_map), batch_size=4, sampler=sampler) # BS=4 for Base model
    va_loader = DataLoader(HardenedDataset(val_df, audio_map), batch_size=4)
    te_loader = DataLoader(HardenedDataset(test_df, audio_map), batch_size=4)

    # TWO-STAGE HARDENING
    print("\n🔥 STAGE 1: COMPONENT STABILIZATION (FROZEN BACKBONE)")
    for param in model.wavlm.parameters(): param.requires_grad = False
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    sch = get_cosine_schedule_with_warmup(opt, 0, len(tr_loader)*3)
    for epoch in range(1, 4):
        loss = train_epoch(model, tr_loader, device, opt, sch, epoch, l_ce, l_sc, alpha=0.5)
        v_a, v_f = evaluate(model, va_loader, device)
        print(f"   > Stabilize Ep {epoch}: Val Acc {v_a:.3f} | F1 {v_f:.3f}")

    print("\n🚀 STAGE 2: DEEP SOTA FINE-TUNING (UNFROZEN)")
    for param in model.wavlm.parameters(): param.requires_grad = True
    opt = torch.optim.AdamW([{"params": model.wavlm.parameters(), "lr": 2e-5}, {"params": model.classifier.parameters(), "lr": 1e-4}], lr=1e-4)
    sch = get_cosine_schedule_with_warmup(opt, 100, len(tr_loader)*15)
    
    best_f1 = 0
    for epoch in range(1, 16):
        loss = train_epoch(model, tr_loader, device, opt, sch, epoch, l_ce, l_sc, alpha=0.8)
        va, vf = evaluate(model, va_loader, device)
        ta, tf = evaluate(model, te_loader, device)
        print(f"🏆 Epoch {epoch} | Val F1: {vf:.3f} | Test F1: {tf:.3f} | loss: {loss:.3f}")
        if vf > best_f1:
            best_f1 = vf
            torch.save(model.state_dict(), config.CHECKPOINT_DIR / "hardened_sota_best.pt")
            print("   🌟 New Best Checkpoint Saved")

if __name__ == "__main__": main()
