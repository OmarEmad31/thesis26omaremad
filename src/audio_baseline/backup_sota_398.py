"""
Egyptian Arabic SER — Research Platform v2 (Hardened)
=====================================================
Strict Research Architecture for Diagnostic Control.

Fixes:
1. Path Resolution: Full-path matching to solve segment_XXX.wav collisions.
2. Preprocessing Discipline: Load -> Trim -> Crop -> Prosody -> Aug -> Pad -> Mask.
3. Preprocessing Audit: Statistical reporting of audio health before training.
4. Macro F1 Centered: All selection and metrics optimized for class balancing.
"""

import os, sys, json, time, random, datetime, subprocess, argparse, warnings
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, Sampler
import librosa
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, 
    recall_score, precision_score, classification_report,
    balanced_accuracy_score
)
from transformers import WavLMModel, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from audiomentations import Compose, AddGaussianNoise, Gain

# ─────────────────────────────────────────────────────────
# 1. CONFIGURATION SYSTEM
# ─────────────────────────────────────────────────────────

class ResearchConfig:
    # Environment
    SR = 16000
    MAX_LEN = 80000 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    
    # Selection
    SELECTION_METRIC = "macro_f1" # macro_f1, uar, accuracy
    PHASE1_EPOCHS = 4
    PHASE2_EPOCHS = 26
    LR_BACKBONE_LORA = 5e-6
    LR_HEAD = 2e-5
    BATCH_SIZE = 14
    
    # Preprocessing (Stage 2 Requirements)
    TRIM_AUDIO = True
    TRIM_TOP_DB = 40
    TRAIN_CROP_MODE = "center" # options: center, random, energy
    VAL_CROP_MODE = "center"
    
    # Augmentation Profiles
    AUGMENTATION_PROFILE = "none" # options: none, safe, current
    
    # Architecture
    MODEL_NAME = "microsoft/wavlm-base-plus"
    POOLING = "mean_std" 
    
    # Objective (Strict Baseline)
    USE_WEIGHTED_CE = True
    CLASS_WEIGHT_MODE = "effective_num"
    USE_SWA = False
    USE_SUPCON = False
    USE_BALANCED_BATCH_SAMPLER = False
    USE_FOCAL_LOSS = False

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
# 2. PATH RESOLUTION SYSTEM (Stage 1)
# ─────────────────────────────────────────────────────────

def resolve_exact_paths(df, colab_root, extracted_root=Path("/content/dataset")):
    print(f"\n🔍 [STAGE 1] Path Resolution (Total Rows: {len(df)})")
    
    # 1. Inspect data
    print("CSV Columns:", df.columns.tolist())
    print("First 10 audio_relpath examples:")
    for p in df["audio_relpath"].head(10): print(f"  {p}")
    
    # 2. Build local file map using full relative paths
    print("Building index of all available WAV files on Drive and Local...")
    abs_map = {} # Maps full relative path suffix to absolute path
    name_map = defaultdict(list) # Basename fallback (only if unique)
    
    # Potential roots to check
    search_paths = [colab_root, extracted_root]
    
    # If in Colab, search the whole drive first
    if os.path.exists("/content/drive/MyDrive"):
        search_paths.append(Path("/content/drive/MyDrive"))

    for root_p in search_paths:
        if not root_p.exists(): continue
        print(f"Scanning root: {root_p}")
        for p in root_p.rglob("*.wav"):
            # Create a key representing the relative folder structure
            # e.g. 'videoplayback (1)_segments/SPEAKER_00_segment_0007.wav'
            try:
                rel_key = str(p.relative_to(root_p)).replace("\\", "/")
                abs_map[rel_key] = str(p)
                # Also index basenames for the fallback
                name_map[p.name].append(str(p))
            except: continue

    # 3. Resolve
    resolved, unresolved, ambiguous = [], [], []
    
    for i, row in df.iterrows():
        rel = str(row["audio_relpath"]).replace("\\", "/")
        
        # Priority 1: Exact Suffix Match (Handles folders correctly)
        found = False
        if rel in abs_map:
            resolved.append(abs_map[rel])
            found = True
        else:
            # Check if any indexed path ends with this relative path
            for r_key, a_path in abs_map.items():
                if r_key.endswith(rel):
                    resolved.append(a_path)
                    found = True
                    break
        
        if found: continue
            
        # Priority 2: Basename check as last resort
        fname = Path(rel).name
        matches = name_map.get(fname, [])
        if len(matches) == 1:
            resolved.append(matches[0])
        elif len(matches) > 1:
            ambiguous.append((rel, matches))
            resolved.append(None)
        else:
            unresolved.append(rel)
            resolved.append(None)

    df["resolved_path"] = resolved
    
    print(f"✅ Resolved Unique Paths: {len(df) - len(unresolved) - len(ambiguous)}")
    print(f"❌ Unresolved: {len(unresolved)}")
    print(f"⚠️ Ambiguous (Multiple matches): {len(ambiguous)}")
    
    if unresolved:
        print("\nFirst 10 Unresolved Examples:")
        for u in unresolved[:10]: print(f"  {u}")
    if ambiguous:
        print("\nFirst 10 Ambiguous Examples (Collisions):")
        for u_rel, u_matches in ambiguous[:10]:
            print(f"  Rel Path from CSV: {u_rel}")
            print(f"  Potential Collisions: {u_matches}")

    if None in resolved:
        print("\n[ERROR] Path resolution failed. You have audio_relpaths in your CSV that do not exist or are ambiguous.")
        raise FileNotFoundError("CRITICAL: Not all audio paths could be resolved uniquely. Clean your dataset or fix relative paths.")
    
    # 6. Random Samples for user
    print("\n💎 10 Random Resolved Samples (Spot Check):")
    samples = df.sample(min(10, len(df)))
    for _, r in samples.iterrows():
        print(f"  Label: {r['emotion_final']} | Rel: {r['audio_relpath']} -> Abs: {r['resolved_path']}")
        
    return df

# ─────────────────────────────────────────────────────────
# 3. HARDENED PREPROCESSING (Stage 2)
# ─────────────────────────────────────────────────────────

def energy_crop(y, target_len):
    if len(y) <= target_len: return y, 0
    win = target_len; hop = target_len // 4
    energies = [np.sum(y[i:i+win]**2) for i in range(0, len(y)-win+1, hop)]
    best_idx = np.argmax(energies) * hop
    return y[best_idx : best_idx + win], best_idx

def extract_prosody(y):
    """Extraction from real segment ONLY."""
    if len(y) < 100: return np.zeros(4, dtype=np.float32)
    try:
        rms = float(np.mean(librosa.feature.rms(y=y)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
        f0 = librosa.yin(y, fmin=float(librosa.note_to_hz('C2')), fmax=float(librosa.note_to_hz('C7')))
        voiced = f0[~np.isnan(f0)]
        f0m = float(np.mean(voiced)) / 500.0 if len(voiced) > 0 else 0.0
        f0s = float(np.std(voiced)) / 100.0 if len(voiced) > 0 else 0.0
        return np.nan_to_num(np.array([rms, zcr, f0m, f0s], dtype=np.float32))
    except: return np.zeros(4, dtype=np.float32)

class HardenedDataset(Dataset):
    def __init__(self, df, config, augment=False):
        self.df = df.reset_index(drop=True)
        self.cfg = config
        self.augment = augment
        
        # Profile
        if config.AUGMENTATION_PROFILE == "safe":
            self.aug = Compose([Gain(p=0.5, min_gain_db=-6, max_gain_db=6), AddGaussianNoise(p=0.2, min_amplitude=0.001, max_amplitude=0.01)])
        elif config.AUGMENTATION_PROFILE == "current":
            from audiomentations import PitchShift
            self.aug = Compose([AddGaussianNoise(p=0.35), PitchShift(p=0.35)])
        else:
            self.aug = None

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["resolved_path"]
        
        try:
            # 1. Load exact path
            wav, _ = librosa.load(path, sr=self.cfg.SR)
            orig_duration = len(wav) / self.cfg.SR
            
            # 2. Trim
            if self.cfg.TRIM_AUDIO:
                wav, _ = librosa.effects.trim(wav, top_db=self.cfg.TRIM_TOP_DB)
            
            # 3. Crop Segment (Select before padding)
            mode = self.cfg.TRAIN_CROP_MODE if self.augment else self.cfg.VAL_CROP_MODE
            if len(wav) > self.cfg.MAX_LEN:
                if mode == "random":
                    s = random.randint(0, len(wav) - self.cfg.MAX_LEN)
                    yt = wav[s : s + self.cfg.MAX_LEN]
                elif mode == "energy":
                    yt, s = energy_crop(wav, self.cfg.MAX_LEN)
                else: # center
                    s = (len(wav) - self.cfg.MAX_LEN) // 2
                    yt = wav[s : s + self.cfg.MAX_LEN]
                selected_len = self.cfg.MAX_LEN
            else:
                yt = wav
                selected_len = len(wav)
            
            # 4. Prosody from real segment (Stage 2.6)
            prosody = extract_prosody(yt)
            
            # 5. Augment real segment (Stage 2.7)
            if self.augment and self.aug:
                yt = self.aug(samples=yt, sample_rate=self.cfg.SR)
            
            # 6. Pad & Mask (Stage 2.8 & 2.9)
            final_wav = np.zeros(self.cfg.MAX_LEN, dtype=np.float32)
            final_wav[:selected_len] = yt[:selected_len]
            
            mask = np.zeros(self.cfg.MAX_LEN, dtype=np.int64)
            mask[:selected_len] = 1
            
            return {
                "wav": torch.tensor(final_wav, dtype=torch.float32),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
                "prosody": torch.tensor(prosody, dtype=torch.float32),
                "label": torch.tensor(int(row["lid"]), dtype=torch.long),
                "meta": {
                    "path": str(row["audio_relpath"]),
                    "resolved": path,
                    "orig_dur": orig_duration,
                    "selected_len": selected_len,
                    "crop_mode": mode,
                    "aug": self.cfg.AUGMENTATION_PROFILE
                }
            }
        except Exception as e:
            print(f"\n[FATAL ERROR] Failed to load sample: {path}")
            print(f"Context: {row}")
            raise e

# ─────────────────────────────────────────────────────────
# 4. PREPROCESSING AUDIT
# ─────────────────────────────────────────────────────────

def run_preprocessing_audit(ds, emotions):
    print("\n🛠️ [STAGE 2] Preprocessing Audit (Small Batch)")
    stats = []
    for i in tqdm(range(min(50, len(ds))), desc="Auditing"):
        b = ds[i]
        stats.append({
            "label": emotions[b["label"].item()],
            "orig_dur": b["meta"]["orig_dur"],
            "selected_len": b["meta"]["selected_len"] / 16000,
            "prosody_nan": np.isnan(b["prosody"].numpy()).any()
        })
    df = pd.DataFrame(stats)
    print("\nAudit Results:")
    print(f"  Avg Original Duration: {df['orig_dur'].mean():.2f}s")
    print(f"  Avg Processed Duration: {df['selected_len'].mean():.2f}s")
    print(f"  Short clips (<0.5s): {len(df[df['selected_len'] < 0.5])}")
    print(f"  Prosody NaNs found: {df['prosody_nan'].sum()}")
    if len(df[df['selected_len'] < 0.5]) > 5:
        print("[WARNING] Many clips are very short after trimming. Consider lowering TRIM_TOP_DB.")

# ─────────────────────────────────────────────────────────
# 5. MODEL COMPONENTS
# ─────────────────────────────────────────────────────────

class MaskedPooling(nn.Module):
    def __init__(self, d_in, mode="mean_std"):
        super().__init__()
        self.mode = mode
        
    def forward(self, x, mask):
        # x: [B, T, D], mask: [B, T] (bool)
        mask_expanded = mask.unsqueeze(-1)
        mu = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-5)
        if self.mode == "mean_std":
            var = (((x - mu.unsqueeze(1))**2) * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-5)
            return torch.cat([mu, torch.sqrt(var + 1e-6)], dim=-1)
        return mu

class ResearchModel(nn.Module):
    def __init__(self, cfg, num_labels):
        super().__init__()
        self.cfg = cfg
        base = WavLMModel.from_pretrained(cfg.MODEL_NAME, output_hidden_states=True)
        lora_cfg = LoraConfig(
            r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05, bias="none"
        )
        self.wavlm = get_peft_model(base, lora_cfg)
        
        d_p = 768 * (2 if cfg.POOLING == "mean_std" else 1)
        self.layer_weights = nn.Parameter(torch.ones(6))
        self.pooler = MaskedPooling(768, mode=cfg.POOLING)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_p + 4, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        )

    def _get_frame_mask(self, hidden_len, sample_mask):
        B, S = sample_mask.shape
        ratio = hidden_len / S
        valid_samples = sample_mask.sum(dim=1)
        valid_frames = torch.ceil(valid_samples.float() * ratio).long().clamp(min=1, max=hidden_len)
        ar = torch.arange(hidden_len, device=sample_mask.device).unsqueeze(0)
        return ar < valid_frames.unsqueeze(1) # [B, T] Bool

    def forward(self, wav, mask, prosody):
        out = self.wavlm(wav, attention_mask=mask, output_hidden_states=True)
        h = out.hidden_states[-6:]
        w = F.softmax(self.layer_weights, dim=0)
        weighted = sum(w[i] * h[i] for i in range(len(h)))
        f_mask = self._get_frame_mask(weighted.shape[1], mask)
        pooled = self.pooler(weighted, f_mask)
        logits = self.classifier(torch.cat([pooled, prosody], dim=-1))
        return logits

# ─────────────────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────────────────

def run_diagnostics(truths, preds, probs, metas, emotions, prefix="Val"):
    acc = accuracy_score(truths, preds)
    mf1 = f1_score(truths, preds, average="macro")
    wf1 = f1_score(truths, preds, average="weighted")
    uar = balanced_accuracy_score(truths, preds)
    
    print(f"\n📊 Diagnostics [{prefix}]")
    print(f"Acc: {acc:.4f} | MacroF1: {mf1:.4f} | WeightedF1: {wf1:.4f} | UAR: {uar:.4f}")
    
    report = classification_report(truths, preds, labels=list(range(len(emotions))), target_names=emotions, zero_division=0)
    print("\n" + report)
    
    cm = confusion_matrix(truths, preds, labels=list(range(len(emotions))))
    print("Confusion Matrix:")
    print(cm)
    
    df_rows = []
    for i in range(len(truths)):
        df_rows.append({
            "path": metas[i]["path"],
            "true": emotions[truths[i]],
            "pred": emotions[preds[i]],
            "conf": np.max(probs[i]),
            "duration": metas[i]["orig_dur"]
        })
    return pd.DataFrame(df_rows), mf1

# ─────────────────────────────────────────────────────────
# 7. TRAINING ENGINE
# ─────────────────────────────────────────────────────────

def train():
    cfg = ResearchConfig()
    seed_everything(cfg.SEED)
    
    colab_root = Path("/content/drive/MyDrive/Thesis Project")
    csv_p = colab_root / "data/processed/splits/text_hc"
    results_dir = colab_root / "experiments" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    tr_df = pd.read_csv(csv_p / "train.csv")
    va_df = pd.read_csv(csv_p / "val.csv")
    emotions = sorted(tr_df["emotion_final"].unique())
    lid = {e: i for i, e in enumerate(emotions)}
    tr_df["lid"] = tr_df["emotion_final"].map(lid)
    va_df["lid"] = va_df["emotion_final"].map(lid)
    
    # STAGE 1: Path Resolution
    tr_df = resolve_exact_paths(tr_df, colab_root)
    va_df = resolve_exact_paths(va_df, colab_root)
    
    tr_ds = HardenedDataset(tr_df, cfg, augment=True)
    va_ds = HardenedDataset(va_df, cfg, augment=False)
    
    # STAGE 2: Preprocessing Audit
    run_preprocessing_audit(tr_ds, emotions)
    
    tr_loader = DataLoader(tr_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=16)
    
    model = ResearchModel(cfg, len(emotions)).to(cfg.DEVICE)
    scaler = GradScaler(cfg.DEVICE)
    weights = (1.0 - 0.999) / (1.0 - np.power(0.999, tr_df["lid"].value_counts().sort_index().values))
    weights = torch.tensor(weights / np.mean(weights), dtype=torch.float32).to(cfg.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    # PHASE 1
    print(f"\n🔥 PHASE 1: Warmup")
    for n, p in model.wavlm.named_parameters(): p.requires_grad = False
    opt1 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    
    for ep in range(1, cfg.PHASE1_EPOCHS + 1):
        model.train(); ep_loss = 0.0
        for b in tqdm(tr_loader, desc=f"Ph1 Ep{ep}", leave=False):
            w, m, p, l = b["wav"].to(cfg.DEVICE), b["attention_mask"].to(cfg.DEVICE), b["prosody"].to(cfg.DEVICE), b["label"].to(cfg.DEVICE)
            with autocast(cfg.DEVICE):
                logits = model(w, m, p)
                loss = criterion(logits, l)
            opt1.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt1); scaler.update()
            ep_loss += loss.item()
        
        # Validation
        model.eval(); ts, ps, prbs, metas = [], [], [], []
        with torch.no_grad():
            for b in va_loader:
                w, m, p, l = b["wav"].to(cfg.DEVICE), b["attention_mask"].to(cfg.DEVICE), b["prosody"].to(cfg.DEVICE), b["label"].to(cfg.DEVICE)
                logits = model(w, m, p)
                prbs.extend(F.softmax(logits, dim=-1).cpu().numpy())
                ps.extend(logits.argmax(1).cpu().numpy())
                ts.extend(l.cpu().numpy())
                metas.extend([dict(zip(b['meta'].keys(), values)) for values in zip(*b['meta'].values())])
        run_diagnostics(ts, ps, prbs, metas, emotions, prefix=f"Ph1_Ep{ep}")

    # PHASE 2
    print(f"\n🚀 PHASE 2: Core Research")
    for n, p in model.wavlm.named_parameters(): p.requires_grad = "lora_" in n
    opt2 = torch.optim.AdamW([
        {"params": [p for n, p in model.wavlm.named_parameters() if "lora_" in n], "lr": cfg.LR_BACKBONE_LORA},
        {"params": [p for n, p in model.named_parameters() if "wavlm" not in n], "lr": cfg.LR_HEAD}
    ], weight_decay=0.01)
    
    best_mf1 = -1.0
    for ep in range(1, cfg.PHASE2_EPOCHS + 1):
        model.train(); ep_loss = 0.0
        for b in tqdm(tr_loader, desc=f"Ph2 Ep{ep}", leave=False):
            w, m, p, l = b["wav"].to(cfg.DEVICE), b["attention_mask"].to(cfg.DEVICE), b["prosody"].to(cfg.DEVICE), b["label"].to(cfg.DEVICE)
            with autocast(cfg.DEVICE):
                logits = model(w, m, p)
                loss = criterion(logits, l)
            opt2.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt2); scaler.update()
            ep_loss += loss.item()
            
        model.eval(); ts, ps, prbs, metas = [], [], [], []
        with torch.no_grad():
            for b in va_loader:
                w, m, p, l = b["wav"].to(cfg.DEVICE), b["attention_mask"].to(cfg.DEVICE), b["prosody"].to(cfg.DEVICE), b["label"].to(cfg.DEVICE)
                logits = model(w, m, p)
                prbs.extend(F.softmax(logits, dim=-1).cpu().numpy())
                ps.extend(logits.argmax(1).cpu().numpy())
                ts.extend(l.cpu().numpy())
                metas.extend([dict(zip(b['meta'].keys(), values)) for values in zip(*b['meta'].values())])
        
        diag_df, mf1 = run_diagnostics(ts, ps, prbs, metas, emotions, prefix=f"Ph2_Ep{ep}")
        if mf1 > best_mf1:
            best_mf1 = mf1
            torch.save(model.state_dict(), results_dir / "best_model.pt")
            diag_df.to_csv(results_dir / "best_predictions.csv", index=False)
            print(f"🌟 New Best Macro F1: {mf1:.4f}")

if __name__ == "__main__":
    train()
