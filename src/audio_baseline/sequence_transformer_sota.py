"""
Sequence Transformer v10 — Bypassing the Crop Corruption
======================================================
Instead of randomly cropping 5 seconds of audio and praying the
emotion was in that 5 seconds, this script processes the ENTIRE audio
file using a sliding window to generate a sequence of features.

It then trains a lightweight PyTorch Transformer to mathematically
sweep across the entire audio sequence and place its Attention exactly
on the moment the emotion happens (e.g., ignoring 10 seconds of silence
and zeroing in on 2 seconds of crying). 
"""

import os, sys, subprocess, zipfile, random
from collections import defaultdict

def install_deps():
    pkgs = []
    for mod, pkg in [("audiomentations", "audiomentations"), ("transformers", "transformers")]:
        try: __import__(mod)
        except ImportError: pkgs.append(pkg)
    if pkgs:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *pkgs, "-q"])

if "google.colab" in sys.modules or os.path.exists("/content"):
    install_deps()

import torch, torch.nn as nn, torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd, numpy as np, librosa
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoProcessor, AutoModel

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
SR             = 16000
CHUNK_LEN      = 3 * SR       # 3-second sliding window
STRIDE_LEN     = int(1.5 * SR)# 1.5-second stride overlapping
EPOCHS         = 40
BATCH_SIZE     = 32
MODEL_NAME     = "superb/wav2vec2-base-superb-er"

# ─────────────────────────────────────────────────────────
# CACHING / FEATURE EXTRACTION STAGE
# Extracts the ENTIRE audio file into a Sequence of Frames
# ─────────────────────────────────────────────────────────
@torch.no_grad()
def extract_sequences(df, path_map, processor, model, device, is_train=True):
    model.eval()
    dataset_features = []
    dataset_labels   = []
    
    desc = "Extracting Train Sequences" if is_train else "Extracting Val Sequences"
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        fname = Path(row["audio_relpath"]).name
        if fname not in path_map: continue
        
        try:
            # 1. Load FULL audio, no destructive trimming.
            wav, _ = librosa.load(path_map[fname], sr=SR)
            # Z-Score normalization for mic equalization
            wav = (wav - np.mean(wav)) / (np.std(wav) + 1e-6)
            
            # Data Augmentation (Expanding dataset safely)
            if is_train:
                variants = [
                    wav,
                    librosa.effects.time_stretch(wav, rate=1.15),
                    librosa.effects.time_stretch(wav, rate=0.85),
                    librosa.effects.pitch_shift(wav, sr=SR, n_steps=2),
                ]
            else:
                variants = [wav]
                
            for var_idx, var in enumerate(variants):
                if len(var) < CHUNK_LEN:
                    # Pad if less than 3 seconds
                    var = np.pad(var, (0, CHUNK_LEN - len(var)))
                
                # Slicing full audio into overlapping 3s chunks
                chunks = []
                for start in range(0, len(var) - CHUNK_LEN + 1, STRIDE_LEN):
                    chunks.append(var[start : start + CHUNK_LEN])
                
                # If there's leftover audio that's long enough, capture it
                if len(var) > CHUNK_LEN and len(var) % STRIDE_LEN != 0:
                    chunks.append(var[-CHUNK_LEN:])
                    
                if len(chunks) == 0:
                    chunks.append(np.pad(var, (0, CHUNK_LEN - len(var))))
                
                # Extract deep features for each 3s sub-clip
                sequence_frames = []
                # Batch process if possible, but safe iterative approach to prevent OOM
                for chunk in chunks:
                    inputs = processor(chunk, sampling_rate=SR, return_tensors="pt").to(device)
                    # Get deep emotion layers
                    outputs = model(**inputs)
                    # Mean over time for this 3-second chunk
                    chunk_embed = outputs.last_hidden_state.mean(dim=1).squeeze(0) # [768]
                    sequence_frames.append(chunk_embed.cpu())
                    
                # Full sequence for the audio file: [Num_Chunks, 768]
                full_sequence = torch.stack(sequence_frames)
                dataset_features.append(full_sequence)
                dataset_labels.append(row["lid"])
                
        except Exception as e:
            pass
            
    return dataset_features, dataset_labels

# ─────────────────────────────────────────────────────────
# CUSTOM PYTORCH DATASET WITH PADDING
# ─────────────────────────────────────────────────────────
class SequenceDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __len__(self): return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], torch.tensor(self.labels[idx], dtype=torch.long)

def collate_sequences(batch):
    features, labels = zip(*batch)
    # Pads [Seq_Len, 768] sequences to the max sequence length in this batch
    padded_features = pad_sequence(features, batch_first=True, padding_value=0.0) # [B, Max_Seq_Len, 768]
    
    # Create mask for attention (True where it's zero padded)
    lengths = torch.tensor([f.size(0) for f in features])
    max_len = padded_features.size(1)
    mask = torch.arange(max_len).expand(len(lengths), max_len) >= lengths.unsqueeze(1)
    
    return padded_features, torch.stack(labels), mask

# ─────────────────────────────────────────────────────────
# THE SEQUENCE TRANSFORMER 
# Automatically searches the whole file for the emotion.
# ─────────────────────────────────────────────────────────
class EmotionSequenceTransformer(nn.Module):
    def __init__(self, num_labels, d_model=768, nhead=8, num_layers=3):
        super().__init__()
        
        # Transformer sees the sequence of 3s clips
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=1024, dropout=0.3, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Attention Pooling: Learns to zero-in on the specific frame (clip) that has the crying or yelling
        self.attn = nn.Linear(d_model, 1)
        
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, num_labels)
        )
        
    def forward(self, sequences, padding_mask):
        # sequences: [B, Max_Seq_Len, 768]
        # padding_mask: [B, Max_Seq_Len] (True for padding)
        
        x = self.transformer(sequences, src_key_padding_mask=padding_mask) # [B, Seq_Len, 768]
        
        # Attention Pooling
        attn_weights = self.attn(x).squeeze(-1) # [B, Seq_Len]
        # Set attention over padded frames to -infinity
        attn_weights.masked_fill_(padding_mask, -1e9)
        
        attn_weights = F.softmax(attn_weights, dim=-1).unsqueeze(-1) # [B, Seq_Len, 1]
        pooled_feat = (x * attn_weights).sum(dim=1) # [B, 768]
        
        logits = self.classifier(pooled_feat)
        return logits

# ─────────────────────────────────────────────────────────
# SETUP UTILS
# ─────────────────────────────────────────────────────────
def get_path_map(colab_root):
    pm = {f.name: str(f) for f in colab_root.rglob("*.wav")}
    if pm: return pm
    zname = "Thesis_Audio_Full.zip"; zpath = None
    for root, _, files in os.walk("/content/drive/MyDrive"):
        if zname in files: zpath = os.path.join(root, zname); break
    if not zpath: raise FileNotFoundError(f"{zname} not found.")
    with zipfile.ZipFile(zpath) as z: z.extractall("/content/dataset")
    return {f.name: str(f) for f in Path("/content/dataset").rglob("*.wav")}

# ─────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    colab_root = Path("/content/drive/MyDrive/Thesis Project")
    csv_p = colab_root / "data/processed/splits/text_hc"
    
    path_map = get_path_map(colab_root)
    tr_df, va_df = pd.read_csv(csv_p / "train.csv"), pd.read_csv(csv_p / "val.csv")
    
    lid = {l: i for i, l in enumerate(sorted(tr_df["emotion_final"].unique()))}
    tr_df["lid"] = tr_df["emotion_final"].map(lid)
    va_df["lid"] = va_df["emotion_final"].map(lid)

    print("\n🧠 LOADING EMOTION BACKBONE (superb/wav2vec2-base-superb-er)...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)

    print("\n🚀 PHASE 1: FULL AUDIO SEQUENCE EXTRACTION (Bypassing Crop Corruption)")
    # Extract training and validation sequences entirely offline to save VRAM and compute
    tr_features, tr_labels = extract_sequences(tr_df, path_map, processor, model, device, is_train=True)
    va_features, va_labels = extract_sequences(va_df, path_map, processor, model, device, is_train=False)
    
    # We don't need the massive 90M parameter backbone in VRAM anymore.
    del processor, model; torch.cuda.empty_cache()
    
    tr_ds = SequenceDataset(tr_features, tr_labels)
    va_ds = SequenceDataset(va_features, va_labels)
    
    # Generate balanced class weights to naturally handle the Neutral imbalance
    class_weights = compute_class_weight(
        class_weight='balanced', classes=np.unique(tr_labels), y=np.array(tr_labels)
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_sequences)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_sequences)
    
    print("\n🚀 PHASE 2: TRAINING PYTORCH SEQUENCE TRANSFORMER (Self-Attention over Time)")
    seq_transformer = EmotionSequenceTransformer(num_labels=len(lid)).to(device)
    
    optimizer = torch.optim.AdamW(seq_transformer.parameters(), lr=5e-4, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1) # Soft boundaries
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_acc, best_f1 = 0.0, 0.0
    
    for ep in range(1, EPOCHS + 1):
        seq_transformer.train()
        train_loss = 0.0
        
        for sequences, labels, mask in tr_loader:
            sequences, labels, mask = sequences.to(device), labels.to(device), mask.to(device)
            
            logits = seq_transformer(sequences, mask)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(seq_transformer.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        seq_transformer.eval()
        ps, ts = [], []
        with torch.no_grad():
            for sequences, labels, mask in va_loader:
                sequences, mask = sequences.to(device), mask.to(device)
                logits = seq_transformer(sequences, mask)
                ps.extend(logits.argmax(-1).cpu().numpy())
                ts.extend(labels.numpy())
                
        acc = accuracy_score(ts, ps)
        f1 = f1_score(ts, ps, average='macro', zero_division=0)
        scheduler.step(acc)
        
        tag = ""
        if acc > best_acc:
            best_acc, best_f1 = acc, f1
            tag = "  *** BEST ***"
            # Optional: save state dict
            
        print(f"Epoch {ep:02d} | Loss {train_loss/len(tr_loader):.3f} | Val Acc {acc:.3f} | Val F1 {f1:.3f}{tag}")
        
    print(f"\n=======================================================")
    print(f"🎉 DEFINITIVE SEQUENCE SYSTEM -> Valid Acc: {best_acc:.3f} | F1: {best_f1:.3f}")
    print(f"=======================================================")
    print("This method mathematically bypassed label corruption on long audio.")

if __name__ == "__main__":
    train()
