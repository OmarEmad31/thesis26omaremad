"""
HArnESS Final Run — WavLM + BiLSTM + Acoustic Fusion
======================================================
This is the completely rebuilt, scientifically grounded pipeline.
1. Elite Preprocessing: Pre-emphasis filtering + Strict 5s Padding.
2. Temporal Modeling: BiLSTM over WavLM hidden sequences.
3. Attention Pooling: Selects the temporal peak of emotion.
4. Acoustic Fusion: MFCC(13) + Pitch + Energy explicitly fed to head.
5. Imbalance Control: Weighted Sampler + Class Weights + Focal Loss.
6. Evaluation: Tracks Macro F1 and Matrix to debug majority guessing.
"""

import os, sys, subprocess, zipfile, random
from collections import defaultdict

def install_deps():
    pkgs = []
    for mod, pkg in [("peft", "peft"), ("transformers", "transformers")]:
        try: __import__(mod)
        except ImportError: pkgs.append(pkg)
    if pkgs:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *pkgs, "-q"])

if "google.colab" in sys.modules or os.path.exists("/content"):
    install_deps()

import torch, torch.nn as nn, torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, Sampler
import pandas as pd, numpy as np, librosa
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from transformers import WavLMModel, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────
SR             = 16000
MAX_LEN        = 80000        # Exactly 5 seconds
K_PER_CLASS    = 2            # Batch size = 14
NUM_CLASSES    = 7
BATCH_SIZE     = K_PER_CLASS * NUM_CLASSES   
EPOCHS         = 30
MODEL_NAME     = "microsoft/wavlm-base-plus"

# ─────────────────────────────────────────────────────────
# IMALANCE SAMPLER
# ─────────────────────────────────────────────────────────
class BalancedBatchSampler(Sampler):
    def __init__(self, labels, k=K_PER_CLASS):
        self.k = k
        self.class_indices = defaultdict(list)
        for i, lbl in enumerate(labels):
            self.class_indices[int(lbl)].append(i)
        self.classes   = sorted(self.class_indices.keys())
        self.n_batches = min(len(v) for v in self.class_indices.values()) // k

    def __iter__(self):
        pools = {c: random.sample(idxs, len(idxs)) for c, idxs in self.class_indices.items()}
        ptrs  = {c: 0 for c in self.classes}
        for _ in range(self.n_batches):
            batch = []
            for c in self.classes:
                batch.extend(pools[c][ptrs[c]: ptrs[c] + self.k])
                ptrs[c] += self.k
            random.shuffle(batch)
            yield batch
    def __len__(self): return self.n_batches

# ─────────────────────────────────────────────────────────
# FOCAL LOSS WITH EXPLICIT CLASS WEIGHTS
# ─────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight # [Num_Classes] tensor

    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(logits, labels, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# ─────────────────────────────────────────────────────────
# ATTENTION POOLING
# ─────────────────────────────────────────────────────────
class AttentionPool(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.attn = nn.Linear(d, 1)
        nn.init.zeros_(self.attn.weight)
        nn.init.zeros_(self.attn.bias)

    def forward(self, x):
        # x is [B, T, D]
        w = F.softmax(self.attn(x), dim=1) # [B, T, 1]
        return (x * w).sum(dim=1)          # [B, D]

# ─────────────────────────────────────────────────────────
# ARCHITECTURE: WavLM -> BiLSTM -> Attention + Acoustic
# ─────────────────────────────────────────────────────────
class WavLM_BiLSTM_Acoustic(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        
        # 1. WavLM Backbone with LoRA
        base = WavLMModel.from_pretrained(MODEL_NAME)
        cfg  = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
        self.wavlm = get_peft_model(base, cfg)
        
        # 2. Temporal BiLSTM (Capturing the sequence of emotion)
        self.bilstm = nn.LSTM(
            input_size=768, 
            hidden_size=256, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        
        # 3. Attention Pooling
        self.attn_pool = AttentionPool(d=512)
        
        # 4. Final MLP Classifier (512 Deep + 15 Acoustic = 527)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512 + 15, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )

    def freeze_backbone(self):
        for n, p in self.wavlm.named_parameters(): p.requires_grad = "lora_" in n
    def unfreeze_lora(self):
        for n, p in self.wavlm.named_parameters(): p.requires_grad = "lora_" in n

    def forward(self, wav, acoustic_features):
        # WavLM Pass
        outputs = self.wavlm(wav).last_hidden_state # [B, T, 768]
        
        # BiLSTM Pass
        lstm_out, _ = self.bilstm(outputs) # [B, T, 512]
        
        # Attention Pooling
        deep_features = self.attn_pool(lstm_out) # [B, 512]
        
        # Acoustic Fusion
        fused_features = torch.cat([deep_features, acoustic_features], dim=-1) # [B, 527]
        
        logits = self.classifier(fused_features)
        return logits

# ─────────────────────────────────────────────────────────
# FEATURE EXTRACTION (Elite Preprocessing)
# ─────────────────────────────────────────────────────────
def compute_acoustic_features(y: np.ndarray) -> np.ndarray:
    # 1. MFCCs (13 bands)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1) # [13]
    
    # 2. RMS Energy
    rms = np.mean(librosa.feature.rms(y=y)) # [1]
    
    # 3. Pitch (F0)
    f0 = librosa.yin(y, fmin=65, fmax=2093)
    f0 = np.nan_to_num(f0)
    pitch_mean = np.mean(f0) / 500.0 # Normalizing pitch roughly [1]
    
    features = np.concatenate([mfcc_mean, [rms, pitch_mean]]) # 15 dimensions total
    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
    return features.astype(np.float32)

class AdvancedAudioDataset(Dataset):
    def __init__(self, df, path_map, augment=False):
        self.df = df.reset_index(drop=True)
        self.path_map = path_map
        self.augment = augment

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]; fname = Path(row["audio_relpath"]).name
        path = self.path_map.get(fname)
        
        # 1. Load Audio
        wav, _ = librosa.load(path, sr=SR)
        
        # 2. Elite Preprocessing: Pre-emphasis Filter to boost high frequencies (emotion)
        wav = librosa.effects.preemphasis(wav)
        
        # 3. Data Augmentation
        if self.augment:
            if random.random() < 0.3: wav = librosa.effects.pitch_shift(wav, sr=SR, n_steps=random.uniform(-1, 1))
            if random.random() < 0.3: wav = librosa.effects.time_stretch(wav, rate=random.uniform(0.9, 1.1))
        
        # 4. Padding / Truncation strictly to 5 seconds
        if len(wav) > MAX_LEN:
            s = random.randint(0, len(wav) - MAX_LEN) if self.augment else 0
            wav = wav[s:s + MAX_LEN]
        else:
            wav = np.pad(wav, (0, MAX_LEN - len(wav)))
            
        # 5. Z-Score Amplitude Normalization
        wav = (wav - np.mean(wav)) / (np.std(wav) + 1e-6)
        
        acoustic = compute_acoustic_features(wav)
        
        return {
            "wav": torch.tensor(wav, dtype=torch.float32),
            "acoustic": torch.tensor(acoustic, dtype=torch.float32),
            "label": torch.tensor(int(row["lid"]), dtype=torch.long)
        }

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
# EVALUATION WITH MATRICES
# ─────────────────────────────────────────────────────────
def evaluate_model(model, loader, device, classes_list):
    model.eval()
    ps, ts = [], []
    with torch.no_grad(), autocast("cuda"):
        for b in loader:
            logits = model(b["wav"].to(device), b["acoustic"].to(device))
            ps.extend(logits.argmax(1).cpu().numpy())
            ts.extend(b["label"].numpy())
            
    acc = accuracy_score(ts, ps)
    f1 = f1_score(ts, ps, average="macro", zero_division=0)
    cm = confusion_matrix(ts, ps)
    return acc, f1, cm

def print_confusion_matrix(cm, classes):
    print("\n[CONFUSION MATRIX] rows = True, cols = Pred")
    # Define column widths based on target class name length
    col_w = max(len(c) for c in classes) + 2 
    header = "".rjust(col_w) + "".join([c[:3].rjust(5) for c in classes])
    print(header)
    for i, row in enumerate(cm):
        print(classes[i].rjust(col_w) + "".join([str(val).rjust(5) for val in row]))

# ─────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────
def train():
    device = "cuda"
    colab_root = Path("/content/drive/MyDrive/Thesis Project")
    csv_p = colab_root / "data/processed/splits/text_hc"
    
    path_map = get_path_map(colab_root)
    tr_df, va_df = pd.read_csv(csv_p / "train.csv"), pd.read_csv(csv_p / "val.csv")
    
    classes_list = sorted(tr_df["emotion_final"].unique())
    lid = {l: i for i, l in enumerate(classes_list)}
    tr_df["lid"] = tr_df["emotion_final"].map(lid)
    va_df["lid"] = va_df["emotion_final"].map(lid)
    
    print("\n🧠 INITIALIZING WIPED ARCHITECTURE (WavLM -> BiLSTM -> Attention)...")
    model = WavLM_BiLSTM_Acoustic(num_labels=len(lid)).to(device)
    model.freeze_backbone()
    
    # 1. Compute Statistical Class Weights dynamically
    train_labels = tr_df["lid"].values
    weights_np = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(weights_np, dtype=torch.float32).to(device)
    print(f"📊 Class Weights Generated: {weights_np}")

    # 2. Initialize Datasets & Dataloaders
    tr_ds = AdvancedAudioDataset(tr_df, path_map, augment=True)
    va_ds = AdvancedAudioDataset(va_df, path_map, augment=False)
    
    # Validation sees 16 at a time.
    va_loader = DataLoader(va_ds, batch_size=16, shuffle=False, num_workers=0)
    
    # Tr frozen is for Phase 1 Head Training
    tr_frozen = DataLoader(tr_ds, batch_size=32, shuffle=True, drop_last=True, num_workers=0)

    # 3. Loss & Optimizers
    focal_criterion = FocalLoss(weight=class_weights, gamma=2.0)
    scaler = GradScaler("cuda")
    
    # PHASE 1
    p1_params = [p for p in model.parameters() if p.requires_grad]
    opt1 = torch.optim.AdamW(p1_params, lr=1e-3, weight_decay=0.01)
    
    print("\n🚀 PHASE 1: WARMING UP BILSTM & HEAD")
    for ep in range(1, 4):
        model.train(); ep_loss = 0.0
        for b in tqdm(tr_frozen, desc=f"Ph1 Ep{ep}", leave=False):
            with autocast("cuda"):
                logits = model(b["wav"].to(device), b["acoustic"].to(device))
                loss = focal_criterion(logits, b["label"].to(device))
            opt1.zero_grad(); scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(p1_params, 1.0)
            scaler.step(opt1); scaler.update(); ep_loss += loss.item()
            
        acc, f1, _ = evaluate_model(model, va_loader, device, classes_list)
        print(f"Ph1 Ep {ep:02d} | Loss {ep_loss/len(tr_frozen):.3f} | Acc {acc:.3f} | Macro F1 {f1:.3f}")

    # PHASE 2
    model.unfreeze_lora()
    opt2 = torch.optim.AdamW([
        {"params": [p for n, p in model.wavlm.named_parameters() if "lora_" in n], "lr": 1e-4},
        {"params": model.bilstm.parameters(), "lr": 1e-3},
        {"params": model.attn_pool.parameters(), "lr": 1e-3},
        {"params": model.classifier.parameters(), "lr": 1e-3},
    ], weight_decay=0.01)
    
    def make_sampler(): return BalancedBatchSampler(train_labels, k=K_PER_CLASS)
    sch2 = get_cosine_schedule_with_warmup(opt2, len(make_sampler()) * 2, len(make_sampler()) * EPOCHS)

    print("\n🚀 PHASE 2: END-TO-END TEMPORAL FINE-TUNING")
    best_f1, best_acc = 0.0, 0.0
    best_cm = None

    for ep in range(1, EPOCHS + 1):
        tr_loader = DataLoader(tr_ds, batch_sampler=make_sampler(), num_workers=0)
        model.train(); ep_loss = 0.0
        
        for b in tqdm(tr_loader, desc=f"Ph2 Ep{ep:02d}", leave=False):
            with autocast("cuda"):
                logits = model(b["wav"].to(device), b["acoustic"].to(device))
                loss = focal_criterion(logits, b["label"].to(device))
            opt2.zero_grad(); scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt2); scaler.update(); sch2.step()
            ep_loss += loss.item()
            
        acc, f1, cm = evaluate_model(model, va_loader, device, classes_list)
        
        tag = ""
        # We explicitly track F1 as our Best model criteria due to imbalance.
        if f1 > best_f1:
            best_f1, best_acc, best_cm = f1, acc, cm
            tag = "  *** BEST F1 ***"
            torch.save(model.state_dict(), colab_root / "harness_wavlm_bilstm.pt")
            
        print(f"Ph2 Ep {ep:02d} | Loss {ep_loss/len(tr_loader):.3f} | Acc {acc:.3f} | Macro F1 {f1:.3f}{tag}")
        
    print(f"\n=======================================================")
    print(f"🎉 FINAL FLAGSHIP EVALUATION (Macro F1 Priority)")
    print(f"  Best Val Accuracy  : {best_acc:.3f}")
    print(f"  Best Val Macro F1  : {best_f1:.3f}")
    if best_cm is not None:
        print_confusion_matrix(best_cm, classes_list)
    print(f"=======================================================")

if __name__ == "__main__":
    train()
