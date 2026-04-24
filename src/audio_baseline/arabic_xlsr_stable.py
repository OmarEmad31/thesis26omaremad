"""
v13 — Egyptian Arabic SER — The 70% Formula
===========================================
Instead of fine-tuning massive English models on tiny datasets, this script:
1. Loads jonatasgrosman/wav2vec2-large-xlsr-53-arabic (Pre-trained on Arabic speech).
2. Applies Heavy Offline Augmentation to multiply the training data by 5x.
3. Caches all 1024-dim representations in memory.
4. Trains a lightweight, robust Discriminant Classifier over the frozen features.
"""

import os, sys, subprocess, zipfile, random
from pathlib import Path

def install_deps():
    pkgs = []
    for mod, pkg in [("transformers", "transformers"), ("librosa", "librosa"), 
                     ("noisereduce", "noisereduce"), ("audiomentations", "audiomentations")]:
        try: __import__(mod)
        except ImportError: pkgs.append(pkg)
    if pkgs:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *pkgs, "-q"])

if "google.colab" in sys.modules or os.path.exists("/content"):
    install_deps()

import torch, librosa
import numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoProcessor, AutoModel
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

SR = 16000
MAX_LEN = 160000  # 10s
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"

# ─────────────────────────────────────────────────────────
# PREPROCESSING & AUGMENTATION ENGINE
# ─────────────────────────────────────────────────────────
def remove_silence(y):
    yt, _ = librosa.effects.trim(y, top_db=25)
    return yt if len(yt) > SR//4 else y

def pad_or_crop(y):
    if len(y) > MAX_LEN: return y[:MAX_LEN]
    return np.pad(y, (0, MAX_LEN - len(y))) if len(y) < MAX_LEN else y

def augment_waveform(y, sr):
    """Generate 5 versions: Original, Pitch Up, Pitch Down, Speed Up, Speed Down"""
    augs = [y]
    # Pitch changes
    augs.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=2.0))
    augs.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=-2.0))
    # Time stretches
    augs.append(librosa.effects.time_stretch(y, rate=1.15))
    augs.append(librosa.effects.time_stretch(y, rate=0.85))
    return [pad_or_crop(a) for a in augs]

# ─────────────────────────────────────────────────────────
# PATH RESOLUTION
# ─────────────────────────────────────────────────────────
def get_path_map(colab_root):
    pm = {f.name: str(f) for f in colab_root.rglob("*.wav")}
    if pm: print(f"[SUCCESS] Found {len(pm)} wav files locally."); return pm
    zname = "Thesis_Audio_Full.zip"; zpath = None
    for root, _, files in os.walk("/content/drive/MyDrive"):
        if zname in files: zpath = os.path.join(root, zname); break
    if not zpath: raise FileNotFoundError(f"{zname} not found in drive.")
    with zipfile.ZipFile(zpath) as z: z.extractall("/content/dataset")
    pm = {f.name: str(f) for f in Path("/content/dataset").rglob("*.wav")}
    print(f"[SUCCESS] Extracted {len(pm)} files to /content/dataset."); return pm

# ─────────────────────────────────────────────────────────
# FEATURE EXTRACTION PIPELINE
# ─────────────────────────────────────────────────────────
def extract_dataset_features(df, path_map, processor, model, device, is_train=True):
    features, labels = [], []
    model.eval()
    
    with torch.no_grad():
        for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {'Train' if is_train else 'Val'}"):
            fname = Path(row["audio_relpath"]).name
            if fname not in path_map: continue
            
            try:
                raw, _ = librosa.load(path_map[fname], sr=SR)
                raw = remove_silence(raw)
                
                # If training, apply 5x augmentation. If validation, just original.
                waveforms = augment_waveform(raw, SR) if is_train else [pad_or_crop(raw)]
                
                for wave in waveforms:
                    # Pass through Wav2Vec2 Arabic Model
                    inputs = processor(wave, sampling_rate=SR, return_tensors="pt", padding=True).to(device)
                    outputs = model(**inputs)
                    
                    # Mean over sequence length (Time pooling)
                    hidden = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
                    features.append(hidden)
                    labels.append(row["lid"])
            except Exception as e:
                pass
                
    return torch.tensor(np.array(features), dtype=torch.float32), torch.tensor(np.array(labels), dtype=torch.long)

# ─────────────────────────────────────────────────────────
# ROBUST CLASSIFIER HEAD
# ─────────────────────────────────────────────────────────
class RobustHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

# ─────────────────────────────────────────────────────────
# MAIN PROGRAM
# ─────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    colab_root = Path("/content/drive/MyDrive/Thesis Project")
    csv_p = colab_root / "data/processed/splits/text_hc"
    
    path_map = get_path_map(colab_root)

    print("📊 Loading CSVs...")
    tr_df = pd.read_csv(csv_p / "train.csv")
    va_df = pd.read_csv(csv_p / "val.csv")
    
    classes = sorted(tr_df["emotion_final"].unique())
    lid = {l: i for i, l in enumerate(classes)}
    tr_df["lid"] = tr_df["emotion_final"].map(lid)
    va_df["lid"] = va_df["emotion_final"].map(lid)
    
    print(f"\n[INFO] Initializing Arabic XLSR Model: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device)
    
    # ─── EXTRACTION PHASE (Frozen Features) ──────────────────────────
    print("\n🚀 PHASE 1: Extracting Frozen Features (Memory Cached)")
    tr_feats, tr_labels = extract_dataset_features(tr_df, path_map, processor, model, device, is_train=True)
    va_feats, va_labels = extract_dataset_features(va_df, path_map, processor, model, device, is_train=False)
    
    print(f"[DATA] Train Matrix: {tr_feats.shape} | Val Matrix: {va_feats.shape}")
    
    # Balance Loss Weights
    weights = compute_class_weight("balanced", classes=np.unique(tr_labels.numpy()), y=tr_labels.numpy())
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    
    tr_loader = DataLoader(TensorDataset(tr_feats, tr_labels), batch_size=128, shuffle=True)
    va_loader = DataLoader(TensorDataset(va_feats, va_labels), batch_size=128, shuffle=False)
    
    # ─── TRAINING PHASE (Rapid ML Classifier) ────────────────────────
    print("\n🚀 PHASE 2: Training Focused Classifier")
    head = RobustHead(tr_feats.shape[1], len(classes)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_acc = 0.0
    best_f1 = 0.0
    
    for epoch in range(1, 51):
        head.train()
        train_loss = 0
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            logits = head(x)
            loss = criterion(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        head.eval()
        ps, ts = [], []
        with torch.no_grad():
            for x, y in va_loader:
                logits = head(x.to(device))
                ps.extend(logits.argmax(1).cpu().numpy())
                ts.extend(y.numpy())
                
        acc = accuracy_score(ts, ps)
        f1 = f1_score(ts, ps, average='macro', zero_division=0)
        
        scheduler.step(acc)
        
        marker = ""
        if acc > best_acc:
            best_acc = acc
            best_f1 = f1
            marker = " *** BEST ***"
            # Optional: Save model
            torch.save(head.state_dict(), colab_root / "arabic_xlsr_best_head.pt")
            
        if epoch % 5 == 1 or marker:
            print(f"Ep {epoch:02d} | Loss: {train_loss/len(tr_loader):.4f} | Val Acc: {acc:.3f} | Val F1: {f1:.3f}{marker}")

    print(f"\n=======================================================")
    print(f"🎉 FINAL BEST SYSTEM -> Valid Acc: {best_acc:.3f} | F1: {best_f1:.3f}")
    print(f"=======================================================")

if __name__ == "__main__":
    main()
