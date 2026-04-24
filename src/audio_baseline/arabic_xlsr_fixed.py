"""
v14 — EGY-MER Audio SOTA Target (~54%) 
===========================================
Fixes "ASR Layer Collapse" by extracting the intermediate hidden states
of the XLSR model where acoustic/emotional prosody is encoded, before it 
gets deleted by the final text-prediction layers.
"""

import os, sys, subprocess, zipfile, random
from pathlib import Path

def install_deps():
    pkgs = []
    for mod, pkg in [("transformers", "transformers"), ("librosa", "librosa"), 
                     ("audiomentations", "audiomentations")]:
        try: __import__(mod)
        except ImportError: pkgs.append(pkg)
    if pkgs:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *pkgs, "-q"])

if "google.colab" in sys.modules or os.path.exists("/content"):
    install_deps()

import torch, librosa
import numpy as np, pandas as pd
import torch.nn as nn
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
    augs = [y]
    augs.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=2.0))
    augs.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=-2.0))
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
    return pm

# ─────────────────────────────────────────────────────────
# MULTI-LAYER FEATURE EXTRACTION (THE FIX)
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
                
                waveforms = augment_waveform(raw, SR) if is_train else [pad_or_crop(raw)]
                
                for wave in waveforms:
                    inputs = processor(wave, sampling_rate=SR, return_tensors="pt", padding=True).to(device)
                    # Enable hidden states extraction
                    outputs = model(**inputs, output_hidden_states=True)
                    
                    # outputs.hidden_states is a tuple of 25 layers (Embedding + 24 Transformer block outputs)
                    all_layers = torch.stack(outputs.hidden_states) # [25, 1, seq_len, 1024]
                    
                    # Extract from acoustic middle layers (Layers 10 to 18)
                    # We average across these middle layers to capture prosody, pitch, and energy.
                    middle_layers = all_layers[10:19] # [9, 1, seq_len, 1024]
                    
                    # Mean over the depth
                    depth_pooled = middle_layers.mean(dim=0) # [1, seq_len, 1024]
                    
                    # Mean over time
                    time_pooled = depth_pooled.mean(dim=1).cpu().numpy()[0] # [1024]
                    
                    features.append(time_pooled)
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
    
    print(f"\n[INFO] Initializing Arabic XLSR Model (Intermediate Layer Extraction)")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device)
    
    print("\n🚀 PHASE 1: Extracting Frozen Multi-Layer Features")
    tr_feats, tr_labels = extract_dataset_features(tr_df, path_map, processor, model, device, is_train=True)
    va_feats, va_labels = extract_dataset_features(va_df, path_map, processor, model, device, is_train=False)
    
    print(f"[DATA] Train Matrix: {tr_feats.shape} | Val Matrix: {va_feats.shape}")
    
    weights = compute_class_weight("balanced", classes=np.unique(tr_labels.numpy()), y=tr_labels.numpy())
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    
    tr_loader = DataLoader(TensorDataset(tr_feats, tr_labels), batch_size=128, shuffle=True)
    va_loader = DataLoader(TensorDataset(va_feats, va_labels), batch_size=128, shuffle=False)
    
    print("\n🚀 PHASE 2: Training Focused Classifier")
    head = RobustHead(tr_feats.shape[1], len(classes)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_acc, best_f1 = 0.0, 0.0
    
    for epoch in range(1, 101):
        head.train()
        train_loss = 0
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(head(x), y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss += loss.item()
            
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
            best_acc, best_f1 = acc, f1
            marker = " *** BEST ***"
            torch.save(head.state_dict(), colab_root / "arabic_xlsr_fixed_head.pt")
            
        if epoch % 5 == 1 or marker:
            print(f"Ep {epoch:03d} | Loss: {train_loss/len(tr_loader):.4f} | Val Acc: {acc:.3f} | Val F1: {f1:.3f}{marker}")

    print(f"\n=======================================================")
    print(f"🎉 FINAL BEST SYSTEM -> Valid Acc: {best_acc:.3f} | F1: {best_f1:.3f}")
    print(f"=======================================================")

if __name__ == "__main__":
    main()
