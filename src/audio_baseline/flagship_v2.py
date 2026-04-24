"""
Flagship v2 — Egyptian Arabic Speech Emotion Recognition
=========================================================
Research Framework for Disciplined SOTA Optimization.

Implemented:
- Phase A: Correctness (Padding Masks, Prosody Discipline, Expanded Prosody)
- Phase B: Modular Pooling (Masked Attention, Mean, Mean+Std, Attentive Stats)
"""

import os, sys, json, time, random, datetime, subprocess
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim.swa_utils import AveragedModel
import librosa
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
from transformers import WavLMModel, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from audiomentations import Compose, AddGaussianNoise, PitchShift

# ─────────────────────────────────────────────────────────
# 1. CONFIGURATION SYSTEM
# ─────────────────────────────────────────────────────────

class FlagshipConfig:
    # Environment
    SR = 16000
    MAX_LEN = 80000  # 5 seconds
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    
    # Data & Splits
    SPLIT_NAME = "audio_hc" # or "text_hc"
    
    # Model Architecture
    BACKBONE = "microsoft/wavlm-base-plus"
    LORA_R = 8
    LORA_ALPHA = 16
    POOLING = "attentive_stats" # choices: attention, mean, mean_std, attentive_stats
    LAYER_FUSION = "last6" # choices: last4, last6, all12, anchored
    
    # Prosody
    PROSODY_MODE = "expanded"  # choices: none, basic4, expanded
    PROSODY_ENCODER = True
    FUSION_TYPE = "concat" # choices: concat, gated
    
    # Optimization
    PHASE1_EPOCHS = 4
    PHASE2_EPOCHS = 26
    SWA_START = 18
    LR_BACKBONE_BASE = 5e-7
    LR_BACKBONE_LORA = 5e-6
    LR_HEAD = 2e-5
    BATCH_SIZE = 14
    K_PER_CLASS = 2
    
    # Loss
    LOSS_TYPE = "focal" # choices: ce, weighted_ce, focal
    FOCAL_GAMMA = 2.0
    USE_SUPCON = False
    SUPCON_WEIGHT = 0.4
    SUPCON_TEMP = 0.07

    def to_dict(self):
        return {k: v for k, v in self.__class__.__dict__.items() if not k.startswith("__") and not callable(v)}

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ─────────────────────────────────────────────────────────
# 2. FEATURE EXTRACTION (PROSODY DISCIPLINE)
# ─────────────────────────────────────────────────────────

def extract_expanded_prosody(y, sr=16000):
    """
    Implements Phase A2 & A3: Multi-dimensional, discipline-safe prosody.
    Only extract from valid audio segment (no padding).
    """
    if len(y) < 100: return np.zeros(35) # Fallback
    
    try:
        # Time Domain
        duration = len(y) / sr
        rms = librosa.feature.rms(y=y)[0]
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        
        # Pitch/F0 (NaN safe)
        f0 = librosa.yin(y, fmin=float(librosa.note_to_hz('C2')), fmax=float(librosa.note_to_hz('C7')))
        voiced_mask = ~np.isnan(f0)
        voiced_ratio = np.mean(voiced_mask)
        
        if voiced_ratio > 0:
            f0_clean = f0[voiced_mask]
            f0_stats = [np.mean(f0_clean), np.std(f0_clean), np.median(f0_clean), np.min(f0_clean), np.max(f0_clean), np.ptp(f0_clean)]
        else:
            f0_stats = [0.0] * 6
            
        # Spectral
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        # MFCC (First 13)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        features = [
            duration, 
            np.mean(rms), np.std(rms), np.min(rms), np.max(rms),
            np.mean(zcr), np.std(zcr),
            voiced_ratio,
            *f0_stats,
            np.mean(spec_cent), np.std(spec_cent),
            np.mean(spec_bw), np.std(spec_bw),
            np.mean(spec_roll), np.std(spec_roll),
            *mfcc_mean
        ]
        return np.nan_to_num(np.array(features, dtype=np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
    except Exception:
        return np.zeros(35)

# ─────────────────────────────────────────────────────────
# 3. DATASET & SAMPLER
# ─────────────────────────────────────────────────────────

class FlagshipDataset(Dataset):
    def __init__(self, df, path_map, config, augment=False, scaler=None):
        self.df = df.reset_index(drop=True)
        self.path_map = path_map
        self.cfg = config
        self.augment = augment
        self.scaler = scaler # To be implemented for Standardization
        
        self.aug_pipe = Compose([
            AddGaussianNoise(p=0.35),
            PitchShift(p=0.35)
        ])
        
        # Intra-class mixup index
        self.class_idx = defaultdict(list)
        for i, row in self.df.iterrows():
            if "lid" in row: self.class_idx[int(row["lid"])].append(i)

    def __len__(self): return len(self.df)

    def _load_audio(self, idx):
        row = self.df.iloc[idx]
        fname = Path(row["audio_relpath"]).name
        if fname not in self.path_map: return None, None, None
        
        try:
            wav, _ = librosa.load(self.path_map[fname], sr=self.cfg.SR)
            # A2: Trim BEFORE padding
            yt, _ = librosa.effects.trim(wav, top_db=30)
            if len(yt) < self.cfg.SR // 4: return None, None, None
            
            # A1: Capture original length
            orig_len = len(yt)
            
            # Handle Length
            if orig_len > self.cfg.MAX_LEN:
                # Random crop during train, center during val
                s = random.randint(0, orig_len - self.cfg.MAX_LEN) if self.augment else (orig_len - self.cfg.MAX_LEN)//2
                yt = yt[s:s + self.cfg.MAX_LEN]
                actual_len = self.cfg.MAX_LEN
            else:
                actual_len = orig_len
                yt = np.pad(yt, (0, self.cfg.MAX_LEN - orig_len))
            
            return yt, int(row["lid"]), actual_len
        except Exception: return None, None, None

    def __getitem__(self, idx):
        yt, lid, actual_len = self._load_audio(idx)
        if yt is None: return self.__getitem__((idx + 1) % len(self.df))
        
        # A2: Extract prosody from VALID portion only
        valid_wav = yt[:actual_len]
        if self.cfg.PROSODY_MODE == "expanded":
            prosody = extract_expanded_prosody(valid_wav, self.cfg.SR)
        else:
            prosody = np.zeros(1) # Stub
            
        if self.augment:
            # Waveform masked augmentation
            yt = self.aug_pipe(samples=yt, sample_rate=self.cfg.SR)
            
        # Create mask
        mask = np.zeros(self.cfg.MAX_LEN, dtype=np.float32)
        mask[:actual_len] = 1.0
        
        return {
            "wav": torch.tensor(yt, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "true_len": torch.tensor(actual_len, dtype=torch.long),
            "prosody": torch.tensor(prosody, dtype=torch.float32),
            "label": torch.tensor(lid, dtype=torch.long)
        }

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, k=2):
        self.k = k
        self.class_indices = defaultdict(list)
        for i, lbl in enumerate(labels): self.class_indices[int(lbl)].append(i)
        self.classes = sorted(self.class_indices.keys())
        self.n_batches = min(len(v) for v in self.class_indices.values()) // k

    def __iter__(self):
        pools = {c: random.sample(idxs, len(idxs)) for c, idxs in self.class_indices.items()}
        ptrs = {c: 0 for c in self.classes}
        for _ in range(self.n_batches):
            batch = []
            for c in self.classes:
                batch.extend(pools[c][ptrs[c]: ptrs[c] + self.k])
                ptrs[c] += self.k
            random.shuffle(batch)
            yield batch
    def __len__(self): return self.n_batches

# ─────────────────────────────────────────────────────────
# 4. MODEL COMPONENTS (POOLING & FUSION)
# ─────────────────────────────────────────────────────────

class MaskedPooling(nn.Module):
    """
    Phase B: Modular Masked Pooling
    """
    def __init__(self, d_in, mode="attention"):
        super().__init__()
        self.mode = mode
        if mode == "attention" or mode == "attentive_stats":
            self.attn = nn.Linear(d_in, 1)
        
    def forward(self, x, mask):
        """
        x: [B, T, D]
        mask: [B, T, 1] (1 for valid, 0 for pad)
        """
        if self.mode == "mean":
            return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-5)
        
        elif self.mode == "mean_std":
            mu = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-5)
            var = (((x - mu.unsqueeze(1))**2) * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-5)
            return torch.cat([mu, torch.sqrt(var + 1e-6)], dim=-1)
            
        elif self.mode == "attention":
            logits = self.attn(x) # [B, T, 1]
            logits = logits.masked_fill(mask == 0, -1e9)
            weights = F.softmax(logits, dim=1)
            return (x * weights).sum(dim=1)
            
        elif self.mode == "attentive_stats":
            logits = self.attn(x)
            logits = logits.masked_fill(mask == 0, -1e9)
            weights = F.softmax(logits, dim=1)
            mu = (x * weights).sum(dim=1)
            var = (((x - mu.unsqueeze(1))**2) * weights).sum(dim=1)
            return torch.cat([mu, torch.sqrt(var + 1e-6)], dim=-1)

class FlagshipModel(nn.Module):
    def __init__(self, cfg, num_labels):
        super().__init__()
        self.cfg = cfg
        
        # Backbone
        base = WavLMModel.from_pretrained(cfg.BACKBONE, output_hidden_states=True)
        lora_cfg = LoraConfig(
            r=cfg.LORA_R, lora_alpha=cfg.LORA_ALPHA,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05, bias="none"
        )
        self.wavlm = get_peft_model(base, lora_cfg)
        
        # D_pooled calculation
        d_base = 768
        d_pooled = d_base if "std" not in cfg.POOLING else d_base * 2
        
        # Layer weights
        n_layers = 12
        if cfg.LAYER_FUSION == "last6": n_fusion = 6
        elif cfg.LAYER_FUSION == "last4": n_fusion = 4
        elif cfg.LAYER_FUSION == "all12": n_fusion = 12
        else: n_fusion = 6 # fallback
        self.layer_weights = nn.Parameter(torch.ones(n_fusion))
        
        self.pooler = MaskedPooling(d_base, mode=cfg.POOLING)
        
        # Prosody Encoder (Phase A4)
        if cfg.PROSODY_MODE == "expanded":
            d_prosody_in = 35
            self.prosody_mlp = nn.Sequential(
                nn.Linear(d_prosody_in, 128),
                nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2)
            )
            d_fusion = d_pooled + 128
        else:
            d_fusion = d_pooled
            
        # Final Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_fusion, 512),
            nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        )

    def forward(self, wav, mask, prosody):
        # Backbone pass
        out = self.wavlm(wav, attention_mask=mask, output_hidden_states=True)
        
        # Layer Fusion
        hidden_states = out.hidden_states[1:] # Layers 1-12
        if self.cfg.LAYER_FUSION == "last6": selected = hidden_states[-6:]
        elif self.cfg.LAYER_FUSION == "last4": selected = hidden_states[-4:]
        elif self.cfg.LAYER_FUSION == "all12": selected = hidden_states
        else: selected = hidden_states[-6:]
        
        w = F.softmax(self.layer_weights, dim=0)
        fused = sum(w[i] * selected[i] for i in range(len(selected))) # [B, T, 768]
        
        # Generate Frame-level mask
        # WavLM downsamples 16000Hz to 50Hz (factor of 320)
        with torch.no_grad():
            B, T_wav = mask.shape
            # Simple 1D pooling to downsample mask
            f_mask = F.avg_pool1d(mask.unsqueeze(1), kernel_size=320, stride=320).transpose(1, 2)
            f_mask = (f_mask > 0.5).float() # Hard mask
            # Ensure T matches hidden_states T
            T_feat = fused.shape[1]
            if f_mask.shape[1] > T_feat: f_mask = f_mask[:, :T_feat, :]
            elif f_mask.shape[1] < T_feat:
                pad = torch.zeros(B, T_feat - f_mask.shape[1], 1, device=f_mask.device)
                f_mask = torch.cat([f_mask, pad], dim=1)
        
        # Pooling
        pooled = self.pooler(fused, f_mask)
        
        # Prosody Fusion
        if self.cfg.PROSODY_MODE == "expanded":
            p_emb = self.prosody_mlp(prosody)
            feat = torch.cat([pooled, p_emb], dim=-1)
        else:
            feat = pooled
            
        logits = self.classifier(feat)
        return logits

# ─────────────────────────────────────────────────────────
# 5. TRAINING & LOGGING
# ─────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, labels):
        ce = F.cross_entropy(logits, labels, reduction="none", weight=self.alpha)
        p_t = torch.exp(-ce)
        return (((1 - p_t) ** self.gamma) * ce).mean()

def run_experiment(config_updates={}):
    cfg = FlagshipConfig()
    for k, v in config_updates.items(): setattr(cfg, k, v)
    
    seed_everything(cfg.SEED)
    
    # Paths
    colab_root = Path("/content/drive/MyDrive/Thesis Project")
    csv_p = colab_root / f"data/processed/splits/{cfg.SPLIT_NAME}"
    results_dir = colab_root / "experiments" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Config
    with open(results_dir / "config.json", "w") as f:
        json.dump(cfg.to_dict(), f, indent=4)
        
    # Data Setup
    tr_df = pd.read_csv(csv_p / "train.csv")
    va_df = pd.read_csv(csv_p / "val.csv")
    emotions = sorted(tr_df["emotion_final"].unique())
    lid = {l: i for i, l in enumerate(emotions)}
    tr_df["lid"] = tr_df["emotion_final"].map(lid)
    va_df["lid"] = va_df["emotion_final"].map(lid)
    
    # Path map
    pm = {f.name: str(f) for f in colab_root.rglob("*.wav")}
    if not pm:
        print("Searching and unzipping...")
        # (Zip logic here if needed, but assuming pm exists or is handled)
    
    tr_ds = FlagshipDataset(tr_df, pm, cfg, augment=True)
    va_ds = FlagshipDataset(va_df, pm, cfg, augment=False)
    
    sampler = BalancedBatchSampler(tr_df["lid"].values, k=cfg.K_PER_CLASS)
    tr_loader = DataLoader(tr_ds, batch_sampler=sampler, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=16, num_workers=0)
    
    model = FlagshipModel(cfg, len(emotions)).to(cfg.DEVICE)
    criterion = FocalLoss(gamma=cfg.FOCAL_GAMMA)
    scaler = GradScaler(cfg.DEVICE)
    
    # Phase 2 Optimizer
    opt = torch.optim.AdamW([
        {"params": [p for n, p in model.wavlm.named_parameters() if "lora_" in n], "lr": cfg.LR_BACKBONE_LORA},
        {"params": [p for n, p in model.named_parameters() if "wavlm" not in n], "lr": cfg.LR_HEAD}
    ], weight_decay=0.01)
    
    total_steps = len(tr_loader) * cfg.PHASE2_EPOCHS
    sch = get_cosine_schedule_with_warmup(opt, total_steps//10, total_steps)
    
    best_f1 = 0.0
    history = []
    
    print(f"\n🚀 Starting Flagship v2 Experiment: {cfg.POOLING} | {cfg.LAYER_FUSION}")
    
    for ep in range(1, cfg.PHASE2_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for b in tqdm(tr_loader, desc=f"Ep {ep}", leave=False):
            w, m, p, l = b["wav"].to(cfg.DEVICE), b["mask"].to(cfg.DEVICE), b["prosody"].to(cfg.DEVICE), b["label"].to(cfg.DEVICE)
            
            with autocast(cfg.DEVICE):
                logits = model(w, m, p)
                loss = criterion(logits, l)
                
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sch.step()
            train_loss += loss.item()
            
        # Eval
        model.eval()
        ps, ts = [], []
        with torch.no_grad():
            for b in va_loader:
                w, m, p, l = b["wav"].to(cfg.DEVICE), b["mask"].to(cfg.DEVICE), b["prosody"].to(cfg.DEVICE), b["label"].to(cfg.DEVICE)
                logits = model(w, m, p)
                ps.extend(logits.argmax(1).cpu().numpy())
                ts.extend(l.cpu().numpy())
        
        acc = accuracy_score(ts, ps)
        f1 = f1_score(ts, ps, average="macro")
        uar = recall_score(ts, ps, average="macro") # Balanced Accuracy / UAR
        
        print(f"Ep {ep:02d} | Loss: {train_loss/len(tr_loader):.3f} | Acc: {acc:.3f} | F1: {f1:.3f} | UAR: {uar:.3f}")
        
        history.append({"epoch": ep, "acc": acc, "f1": f1, "uar": uar})
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), results_dir / "best_model.pt")
            # Save stats
            cm = confusion_matrix(ts, ps)
            np.save(results_dir / "best_cm.npy", cm)
            
    # Final Report
    print(f"\n✅ Experiment Complete. Best Macro F1: {best_f1:.4f}")
    pd.DataFrame(history).to_csv(results_dir / "history.csv", index=False)

if __name__ == "__main__":
    # Baseline with Mask Fix (A1)
    run_experiment({
        "POOLING": "attention",
        "LAYER_FUSION": "last6",
        "PROSODY_MODE": "expanded"
    })
