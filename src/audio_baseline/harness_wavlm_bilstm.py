"""
HArnESS Final Run — WavLM + BiLSTM + Acoustic Fusion
======================================================
1. Elite Preprocessing: Pre-emphasis filtering + Strict 5s Padding.
2. Temporal Modeling: BiLSTM over WavLM hidden sequences.
3. Attention Pooling: Selects the temporal peak of emotion.
4. Acoustic Fusion: MFCC(13) + Pitch + Energy explicitly fed to head.
5. Imbalance Control: Weighted Sampler + Class Weights + Focal Loss.
6. Evaluation: Tracks Macro F1 and Matrix to debug majority guessing.
7. 🔥 PROGRESSIVE UNFREEZING: Gradually unlocks LoRA layers (Top -> Bottom)
   across epochs to completely eliminate Catastrophic Forgetting.
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

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(logits, labels, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        return (((1 - pt) ** self.gamma) * ce_loss).mean()

class AttentionPool(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.attn = nn.Linear(d, 1)
        nn.init.zeros_(self.attn.weight)
        nn.init.zeros_(self.attn.bias)

    def forward(self, x):
        w = F.softmax(self.attn(x), dim=1) 
        return (x * w).sum(dim=1)          

# ─────────────────────────────────────────────────────────
# ARCHITECTURE
# ─────────────────────────────────────────────────────────
class WavLM_BiLSTM_Acoustic(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        base = WavLMModel.from_pretrained(MODEL_NAME)
        cfg  = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
        self.wavlm = get_peft_model(base, cfg)
        
        self.bilstm = nn.LSTM(input_size=768, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)
        self.attn_pool = AttentionPool(d=512)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(512 + 15, 256), nn.LayerNorm(256),
            nn.GELU(), nn.Dropout(0.3), nn.Linear(256, num_labels)
        )

    def freeze_all(self):
        """Completely lock the deep backbone."""
        for p in self.wavlm.parameters(): p.requires_grad = False

    def unfreeze_target_layers(self, layer_indices):
        """Unfreeze specific transformer layers for progressive training."""
        for n, p in self.wavlm.named_parameters():
            if "lora_" in n and ".layers." in n:
                try:
                    parts = n.split('.')
                    layer_idx = int(parts[parts.index('layers') + 1])
                    if layer_idx in layer_indices:
                        p.requires_grad = True
                except: pass

    def forward(self, wav, acoustic_features):
        outputs = self.wavlm(wav).last_hidden_state
        lstm_out, _ = self.bilstm(outputs)
        deep_features = self.attn_pool(lstm_out)
        fused_features = torch.cat([deep_features, acoustic_features], dim=-1)
        return self.classifier(fused_features)

# ─────────────────────────────────────────────────────────
# FEATURE EXTRACTION 
# ─────────────────────────────────────────────────────────
def compute_acoustic_features(y: np.ndarray) -> np.ndarray:
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13), axis=1)
    rms = np.mean(librosa.feature.rms(y=y))
    f0 = np.mean(np.nan_to_num(librosa.yin(y, fmin=65, fmax=2093))) / 500.0
    features = np.concatenate([mfcc, [rms, f0]])
    return np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

class AdvancedAudioDataset(Dataset):
    def __init__(self, df, path_map, augment=False):
        self.df, self.path_map, self.augment = df.reset_index(drop=True), path_map, augment
    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]; path = self.path_map.get(Path(row["audio_relpath"]).name)
        wav, _ = librosa.load(path, sr=SR)
        
        # Elite Preemphasis to capture formants
        wav = librosa.effects.preemphasis(wav)
        
        if self.augment:
            if random.random() < 0.3: wav = librosa.effects.pitch_shift(wav, sr=SR, n_steps=random.uniform(-1, 1))
            if random.random() < 0.3: wav = librosa.effects.time_stretch(wav, rate=random.uniform(0.9, 1.1))
        
        if len(wav) > MAX_LEN:
            s = random.randint(0, len(wav) - MAX_LEN) if self.augment else 0
            wav = wav[s:s + MAX_LEN]
        else: wav = np.pad(wav, (0, MAX_LEN - len(wav)))
            
        wav = (wav - np.mean(wav)) / (np.std(wav) + 1e-6)
        
        return {
            "wav": torch.tensor(wav, dtype=torch.float32),
            "acoustic": torch.tensor(compute_acoustic_features(wav), dtype=torch.float32),
            "label": torch.tensor(int(row["lid"]), dtype=torch.long)
        }

def get_path_map(colab_root):
    pm = {f.name: str(f) for f in colab_root.rglob("*.wav")}
    if pm: return pm
    zpath = None
    for root, _, files in os.walk("/content/drive/MyDrive"):
        if "Thesis_Audio_Full.zip" in files: zpath = os.path.join(root, "Thesis_Audio_Full.zip"); break
    with zipfile.ZipFile(zpath) as z: z.extractall("/content/dataset")
    return {f.name: str(f) for f in Path("/content/dataset").rglob("*.wav")}

# ─────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────
def print_confusion_matrix(cm, classes):
    print("\n[CONFUSION MATRIX] rows = True, cols = Pred")
    col_w = max(len(c) for c in classes) + 2 
    print("".rjust(col_w) + "".join([c[:3].rjust(5) for c in classes]))
    for i, row in enumerate(cm):
        print(classes[i].rjust(col_w) + "".join([str(val).rjust(5) for val in row]))

def train():
    device = "cuda"
    colab_root = Path("/content/drive/MyDrive/Thesis Project")
    csv_p = colab_root / "data/processed/splits/audio_hc"
    
    path_map = get_path_map(colab_root)
    tr_df, va_df = pd.read_csv(csv_p / "train.csv"), pd.read_csv(csv_p / "val.csv")
    
    classes_list = sorted(tr_df["emotion_final"].unique())
    lid = {l: i for i, l in enumerate(classes_list)}
    tr_df["lid"] = tr_df["emotion_final"].map(lid); va_df["lid"] = va_df["emotion_final"].map(lid)
    
    print("\n🧠 INITIALIZING WIPED ARCHITECTURE (WavLM -> BiLSTM -> Attention)...")
    model = WavLM_BiLSTM_Acoustic(num_labels=len(lid)).to(device)
    
    train_labels = tr_df["lid"].values
    class_weights = torch.tensor(compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels), dtype=torch.float32).to(device)

    tr_ds, va_ds = AdvancedAudioDataset(tr_df, path_map, augment=True), AdvancedAudioDataset(va_df, path_map, augment=False)
    def make_sampler(): return BalancedBatchSampler(train_labels)
    va_loader = DataLoader(va_ds, batch_size=16, shuffle=False, num_workers=0)
    
    focal_criterion = FocalLoss(weight=class_weights, gamma=2.0)
    scaler = GradScaler("cuda")
    
    # Extract Parameter Groups for LR Optimization
    head_params = list(model.bilstm.parameters()) + list(model.attn_pool.parameters()) + list(model.classifier.parameters())
    lora_top = [p for n, p in model.wavlm.named_parameters() if "lora_" in n and ".layers." in n and int(n.split('.')[n.split('.').index('layers')+1]) >= 8]
    lora_mid = [p for n, p in model.wavlm.named_parameters() if "lora_" in n and ".layers." in n and 4 <= int(n.split('.')[n.split('.').index('layers')+1]) < 8]
    lora_bot = [p for n, p in model.wavlm.named_parameters() if "lora_" in n and ".layers." in n and int(n.split('.')[n.split('.').index('layers')+1]) < 4]

    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": 1e-3},
        {"params": lora_top, "lr": 5e-4}, 
        {"params": lora_mid, "lr": 1e-4},
        {"params": lora_bot, "lr": 5e-5},
    ], weight_decay=1e-2)

    sch = get_cosine_schedule_with_warmup(optimizer, len(make_sampler()) * 3, len(make_sampler()) * EPOCHS)

    print("\n🚀 STARTING PROGRESSIVE UNFREEZING TRAINING")
    best_f1, best_acc, best_cm = 0.0, 0.0, None

    for ep in range(1, EPOCHS + 1):
        # 🔥 PROGRESSIVE LAYER UNLOCKING
        if ep == 1:
            print("\n🧊 Phase 1: Warming up BiLSTM Head only (All LoRA FROZEN)")
            model.freeze_all()
            unlocked = []
        elif ep == 5:
            print("\n🔥 Phase 2: Unfreezing Top WavLM Layers (8-11)")
            unlocked.extend([8, 9, 10, 11])
            model.unfreeze_target_layers(unlocked)
        elif ep == 12:
            print("\n🔥 Phase 3: Unfreezing Mid WavLM Layers (4-7)")
            unlocked.extend([4, 5, 6, 7])
            model.unfreeze_target_layers(unlocked)
        elif ep == 20:
            print("\n🔥 Phase 4: Unfreezing Bottom WavLM Layers (0-3)")
            unlocked.extend([0, 1, 2, 3])
            model.unfreeze_target_layers(unlocked)

        tr_loader = DataLoader(tr_ds, batch_sampler=make_sampler(), num_workers=0)
        model.train(); ep_loss = 0.0
        
        for b in tqdm(tr_loader, desc=f"Epoch {ep:02d}", leave=False):
            with autocast("cuda"):
                logits = model(b["wav"].to(device), b["acoustic"].to(device))
                loss = focal_criterion(logits, b["label"].to(device))
            optimizer.zero_grad(); scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); sch.step()
            ep_loss += loss.item()
            
        # Evaluation
        model.eval(); ps, ts = [], []
        with torch.no_grad(), autocast("cuda"):
            for b in va_loader:
                ps.extend(model(b["wav"].to(device), b["acoustic"].to(device)).argmax(1).cpu().numpy())
                ts.extend(b["label"].numpy())
        acc = accuracy_score(ts, ps); f1 = f1_score(ts, ps, average="macro", zero_division=0)
        
        tag = ""
        if f1 > best_f1:
            best_f1, best_acc, best_cm = f1, acc, confusion_matrix(ts, ps)
            tag = "  *** BEST F1 ***"; torch.save(model.state_dict(), colab_root / "harness_wavlm_bilstm.pt")
            
        print(f"Ep {ep:02d} | Loss {ep_loss/len(tr_loader):.3f} | Acc {acc:.3f} | Macro F1 {f1:.3f}{tag}")
        
    print(f"\n=======================================================")
    print(f"🎉 FINAL PROGRESSIVE PIPELINE (Macro F1 Priority)")
    print(f"  Best Val Accuracy  : {best_acc:.3f}")
    print(f"  Best Val Macro F1  : {best_f1:.3f}")
    if best_cm is not None: print_confusion_matrix(best_cm, classes_list)
    print(f"=======================================================")

if __name__ == "__main__": train()
