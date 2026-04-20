"""
HArnESS-ECAPA-STABILITY: Egyptian Arabic Audio Emotion Recognition.
Leveraging SpeechBrain Pre-trained ECAPA-TDNN for robust Affective Computing.

Backbone: ECAPA-TDNN (1024 channels, Squeeze-and-Excitation).
Pre-training: VoxCeleb (Millions of speaker identities).
Loss: CrossEntropy (90%) + SupCon (10%) for Geometry.
"""

import os, sys, torch, librosa, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from pathlib import Path

# SpeechBrain integration
try:
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
    from speechbrain.pretrained import EncoderClassifier
    HAS_SB = True
except ImportError:
    HAS_SB = False

# Add project root
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.audio_baseline import config

# ---------------------------------------------------------------------------
# SCL GEOMETRY: SUPERVISED CONTRASTIVE LOSS
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
# ECAPA-TDNN ARCHITECTURE
# ---------------------------------------------------------------------------
class ECAPAForSER(nn.Module):
    def __init__(self, num_labels, device="cuda"):
        super().__init__()
        # Load Pre-trained ECAPA-TDNN from SpeechBrain
        print("📥 Fetching VoxCeleb Pre-trained ECAPA-TDNN...")
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device}
        )
        self.backbone = classifier.mods.embedding_model
        
        # 192 is the default embedding size for standard ECAPA-TDNN in speechbrain
        input_dim = 192 
        
        # SCL Projector
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Emotive Classifier
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_labels)
        )

    def forward(self, x):
        # x expected as 80-dim Filterbanks: [BS, Time, 80]
        # ECAPA expects [BS, Time, Freq] but we check dimensions
        if len(x.shape) == 3:
            # SpeechBrain ECAPA-TDNN expects [BS, Time, Freq]
            pass 
        
        embeddings = self.backbone(x) # [BS, 1, 192]
        embeddings = embeddings.squeeze(1) # [BS, 192]
        
        z = F.normalize(self.projector(embeddings), p=2, dim=1)
        logits = self.classifier(embeddings)
        return logits, z

# ---------------------------------------------------------------------------
# ACOUSTIC PRISM LOADER (FBANK)
# ---------------------------------------------------------------------------
class EgyptianFbankDataset(Dataset):
    def __init__(self, df, audio_map):
        self.df = df
        self.audio_map = audio_map
        self.max_len = 160000 # 10s

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Clean basename extraction
        bn = Path(str(row["audio_relpath"]).replace("\\", "/")).name
        path = self.audio_map.get(bn)
        
        if path is None:
            # Fallback: try case-insensitive or common naming variations if needed
            raise FileNotFoundError(f"❌ CRITICAL ERROR: Could not find audio file '{bn}' anywhere in /content. Please ensure the dataset is unzipped correctly.")

        try:
            audio, _ = librosa.load(path, sr=16000, mono=True)
        except Exception as e:
            raise RuntimeError(f"❌ ERROR loading {path}: {str(e)}")
            
        if len(audio) > self.max_len: audio = audio[:self.max_len]
        else: audio = np.pad(audio, (0, self.max_len - len(audio)))

        # Extract 80-dim log-mel filterbanks
        fbank = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=80, n_fft=400, hop_length=160)
        fbank = librosa.power_to_db(fbank).T # [Time, 80]
        
        return {
            "fbank": torch.tensor(fbank, dtype=torch.float32),
            "label": torch.tensor(row["label_id"], dtype=torch.long)
        }

# ---------------------------------------------------------------------------
# STABILITY TRAINING
# ---------------------------------------------------------------------------
def evaluate(model, loader, device):
    model.eval()
    ps, ts = [], []
    with torch.no_grad():
        for b in loader:
            l, _ = model(b["fbank"].to(device))
            ps.extend(torch.argmax(l, 1).cpu().numpy()); ts.extend(b["label"].numpy())
    return accuracy_score(ts, ps), f1_score(ts, ps, average="macro")

def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not HAS_SB:
        print("❌ ERROR: speechbrain not installed. Run !pip install speechbrain")
        return

    train_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv")
    val_df   = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv")
    test_df  = pd.read_csv(config.SPLIT_CSV_DIR / "test.csv")
    
    classes = sorted(train_df["emotion_final"].unique())
    label2id = {l: i for i, l in enumerate(classes)}
    for df in [train_df, val_df, test_df]: df["label_id"] = df["emotion_final"].map(label2id)
    
    audio_map = {}
    # Prioritize specialized dataset folder, fallback to scanning /content
    data_search_path = Path("/content/dataset") if Path("/content/dataset").exists() else Path("/content")
    print(f"🔍 Scanning {data_search_path} for audio files...")
    
    for p in data_search_path.rglob("*.wav"):
        audio_map[p.name] = p
    
    if len(audio_map) == 0:
        # Emergency secondary scan
        print("⚠️ Warning: No .wav files found in primary path. Scanning entire /content disk...")
        for p in Path("/content").rglob("*.wav"):
            audio_map[p.name] = p
            
    print(f"✅ Found {len(audio_map)} audio files in mapping.")

    y_tr = train_df["label_id"].values
    weights = torch.tensor(compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr), dtype=torch.float32).to(device)
    samples_weight = torch.from_numpy(np.array([1.0/np.bincount(y_tr)[t] for t in y_tr]))
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

    model = ECAPAForSER(len(classes), device=str(device)).to(device)
    
    l_ce = nn.CrossEntropyLoss(weight=weights)
    l_sc = SupConLoss(temperature=0.1)
    
    tr_loader = DataLoader(EgyptianFbankDataset(train_df, audio_map), batch_size=32, sampler=sampler)
    va_loader = DataLoader(EgyptianFbankDataset(val_df, audio_map), batch_size=32)
    te_loader = DataLoader(EgyptianFbankDataset(test_df, audio_map), batch_size=32)

    print("\n🔥 STAGE 1: HEAD TUNING (FROZEN ECAPA)")
    for param in model.backbone.parameters(): param.requires_grad = False
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    for epoch in range(1, 6):
        model.train()
        for b in tqdm(tr_loader, desc=f"Ep{epoch}", leave=False):
            fb = b["fbank"].to(device); l = b["label"].to(device)
            logits, z = model(fb)
            loss = 0.9 * l_ce(logits, l) + 0.1 * l_sc(z, l)
            opt.zero_grad(); loss.backward(); opt.step()
        
        va, vf = evaluate(model, va_loader, device)
        print(f"📊 Epoch {epoch} | Val Acc: {va:.3f} | Val F1: {vf:.3f}")

    print("\n🚀 STAGE 2: END-TO-END FINE-TUNING")
    for param in model.backbone.parameters(): param.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    best_f1 = 0
    for epoch in range(1, 16):
        model.train()
        for b in tqdm(tr_loader, desc=f"Fine-tune Ep{epoch}", leave=False):
            fb = b["fbank"].to(device); l = b["label"].to(device)
            logits, z = model(fb)
            loss = 0.8 * l_ce(logits, l) + 0.2 * l_sc(z, l)
            opt.zero_grad(); loss.backward(); opt.step()
            
        va, vf = evaluate(model, va_loader, device)
        ta, tf = evaluate(model, te_loader, device)
        print(f"🏆 Epoch {epoch} | Val F1: {vf:.3f} | Test F1: {tf:.3f}")
        
        if vf > best_f1:
            best_f1 = vf
            torch.save(model.state_dict(), config.CHECKPOINT_DIR / "ecapa_ser_best.pt")
            print("🌟 New Best F1 Saved")

if __name__ == "__main__": main()
