"""
HArnESS-Perfect-SOTA: Egyptian Arabic Audio Emotion Recognition.
Senior Research Scientist Specialized Implementation.

Backbone: WavLM-Large (Denoising-centric).
Fusion: Mel + MFCC-40 + Chroma (Acoustic Prism).
Head: Hybrid 2D-CNN + Transformer Encoder (Temporal Brain).
Loss: SupCon (L2, Tau 0.1) + Weighted Cross-Entropy.
"""

import os, sys, random, torch, librosa, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoProcessor, Wav2Vec2Model, WavLMModel, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from pathlib import Path

# Try importing audiomentations for SOTA augmentation
try:
    from audiomentations import Compose, AddGaussianNoise, PitchShift, SpecAugment
    HAS_AUDIOMENTATIONS = True
except ImportError:
    HAS_AUDIOMENTATIONS = False

# Add project root
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.audio_baseline import config

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
# SOTA ARCHITECTURE: PERFECT EGYPTIAN SER
# ---------------------------------------------------------------------------
class EgyptianAudioSOTA(nn.Module):
    def __init__(self, num_labels, wavlm_name):
        super().__init__()
        # 1. WavLM-Large Backbone (315M params)
        self.wavlm = WavLMModel.from_pretrained(wavlm_name)
        self.wavlm.config.output_hidden_states = True
        
        # 2. Hybrid Head Architecture
        # Input to CNN will be [BS, 1, Seq_A, 1024 + 128 (mel) + 40 (mfcc) + 12 (chroma)]
        # Total Features = 1024 + 180 = 1204
        self.cnn_head = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4)), # Reduce spectral dim
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4))
        )
        
        # Dimensionality Check: 1204 / 4 / 4 = 75
        self.transformer_input_dim = 64 * 75
        
        # 3. Transformer Encoder (Long-range Egyptian Rhythms)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.transformer_input_dim, nhead=10, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # 4. SupCon Head
        self.supcon_head = nn.Sequential(
            nn.Linear(self.transformer_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # 5. Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.transformer_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_values, audio_mask, mel, mfcc, chroma):
        # Audio Backbone
        w_outs = self.wavlm(input_values, attention_mask=audio_mask).last_hidden_state # [BS, SeqA, 1024]
        
        # Triple-Feature Fusion (aligned frame-by-frame)
        # Mel: [BS, SeqA, 128], MFCC: [BS, SeqA, 40], Chroma: [BS, SeqA, 12]
        fused = torch.cat([w_outs, mel, mfcc, chroma], dim=-1) # [BS, SeqA, 1204]
        
        # 2D-CNN spectral texture extraction
        # Current SeqA for 10s is ~499 frames
        x = fused.unsqueeze(1) # [BS, 1, SeqA, 1204]
        x = self.cnn_head(x)   # [BS, 64, SeqA, 75]
        
        # Transformer Temporal Modeling
        # Reshape to [BS, SeqA, 64*75]
        bs, ch, seq, feat = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(bs, seq, ch * feat)
        
        x = self.transformer_encoder(x)
        
        # Mean Pooling across time
        pooled = x.mean(dim=1)
        
        # Embedding & Logits
        z = F.normalize(self.supcon_head(pooled), p=2, dim=1)
        logits = self.classifier(pooled)
        return logits, z

# ---------------------------------------------------------------------------
# DATASET & ACOUSTIC PRISM EXTRATOR
# ---------------------------------------------------------------------------
class EgyptianPerfectDataset(Dataset):
    def __init__(self, df, audio_map, augment=False):
        self.df = df
        self.audio_map = audio_map
        self.augment = augment
        self.max_len = 160000 # 10s
        if augment and HAS_AUDIOMENTATIONS:
            self.aug = Compose([
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3),
                PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
                SpecAugment(num_mask_holes=1, max_hole_width=20, p=0.5)
            ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        bn = Path(str(row["audio_relpath"]).replace("\\", "/")).name
        path = self.audio_map.get(bn)
        
        audio, _ = librosa.load(path, sr=16000, mono=True)
        if len(audio) > self.max_len: audio = audio[:self.max_len]
        else: audio = np.pad(audio, (0, self.max_len - len(audio)))

        if self.augment:
            if HAS_AUDIOMENTATIONS: audio = self.aug(samples=audio, sample_rate=16000)
            elif random.random() > 0.5: audio += 0.002 * np.random.randn(len(audio))

        # Acoustic Prism Extraction (aligned to WavLM 320 hop_length)
        # 160000 / 320 = 500 frames minus 1 for matching WavLM sequence output
        hop = 320; n_fft = 1024
        mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=128, n_fft=n_fft, hop_length=hop)
        mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40, n_fft=n_fft, hop_length=hop)
        chroma = librosa.feature.chroma_stft(y=audio, sr=16000, n_fft=n_fft, hop_length=hop)
        
        # Transpose to [Seq, Dim]
        mel = torch.tensor(librosa.power_to_db(mel).T[:499], dtype=torch.float32)
        mfcc = torch.tensor(mfcc.T[:499], dtype=torch.float32)
        chroma = torch.tensor(chroma.T[:499], dtype=torch.float32)
        
        return {
            "wav": torch.tensor(audio, dtype=torch.float32),
            "mel": mel, "mfcc": mfcc, "chroma": chroma,
            "label": torch.tensor(row["label_id"], dtype=torch.long)
        }

# ---------------------------------------------------------------------------
# SOTA TRAINING SYSTEM
# ---------------------------------------------------------------------------
def train_epoch(model, loader, device, opt, scheduler, l_sc, l_ce, alpha):
    model.train()
    t_loss, preds, truth = 0, [], []
    pbar = tqdm(loader, desc="Training", leave=False)
    for b in pbar:
        wav = b["wav"].to(device); l = b["label"].to(device)
        mel = b["mel"].to(device); mfcc = b["mfcc"].to(device); chroma = b["chroma"].to(device)
        
        logits, z = model(wav, None, mel, mfcc, chroma)
        loss = (1-alpha)*l_ce(logits, l) + alpha*l_sc(z, l)
        
        opt.zero_grad(); loss.backward(); opt.step(); scheduler.step()
        t_loss += loss.item()
        preds.extend(torch.argmax(logits, 1).cpu().numpy()); truth.extend(l.cpu().numpy())
    return f1_score(truth, preds, average="macro")

def main():
    torch.manual_seed(42); np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W_NAME = "microsoft/wavlm-large"

    train_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv")
    val_df   = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv")
    
    classes = sorted(train_df["emotion_final"].unique())
    label2id = {l: i for i, l in enumerate(classes)}
    train_df["label_id"] = train_df["emotion_final"].map(label2id)
    val_df["label_id"] = val_df["emotion_final"].map(label2id)
    
    audio_map = {}
    src = Path("/content") if Path("/content").exists() else Path(config.DATA_ROOT).parent
    for p in src.rglob("*.wav"): audio_map[p.name] = p

    y_tr = train_df["label_id"].values
    weights = torch.tensor(compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr), dtype=torch.float32).to(device)
    samples_weight = torch.from_numpy(np.array([1.0/np.bincount(y_tr)[t] for t in y_tr]))
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

    model = EgyptianAudioSOTA(len(classes), W_NAME).to(device)
    
    # LOSS & DATA
    l_ce = nn.CrossEntropyLoss(weight=weights)
    l_sc = SupConLoss(temperature=0.1)
    
    tr_loader = DataLoader(EgyptianPerfectDataset(train_df, audio_map, augment=True), batch_size=2, sampler=sampler)
    va_loader = DataLoader(EgyptianPerfectDataset(val_df, audio_map), batch_size=2)

    # TWO-STAGE PROTOCOL
    print("🔥 STAGE 1: WARMUP HEADS (FROZEN WAVLM)")
    for param in model.wavlm.parameters(): param.requires_grad = False
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    sch = get_cosine_schedule_with_warmup(opt, 0, len(tr_loader)*3)
    for _ in range(3): train_epoch(model, tr_loader, device, opt, sch, l_sc, l_ce, alpha=0.5)

    print("🚀 STAGE 2: FINE-TUNING (UNFROZEN)")
    for param in model.wavlm.parameters(): param.requires_grad = True
    opt = torch.optim.AdamW([{"params": model.wavlm.parameters(), "lr": 1e-6}, {"params": model.classifier.parameters(), "lr": 1e-4}], lr=1e-4)
    sch = get_cosine_schedule_with_warmup(opt, 0, len(tr_loader)*12)
    for epoch in range(1, 13):
        f1 = train_epoch(model, tr_loader, device, opt, sch, l_sc, l_ce, alpha=0.7)
        # Eval
        model.eval(); vp, vt = [], []
        with torch.no_grad():
            for b in va_loader:
                ls, _ = model(b["wav"].to(device), None, b["mel"].to(device), b["mfcc"].to(device), b["chroma"].to(device))
                vp.extend(torch.argmax(ls, 1).cpu().numpy()); vt.extend(b["label"].numpy())
        print(f"🏆 Ep {epoch} | Val Mac-F1: {f1_score(vt, vp, average='macro'):.3f}")

if __name__ == "__main__": main()
