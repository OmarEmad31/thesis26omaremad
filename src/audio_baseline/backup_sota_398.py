"""
Egyptian Arabic SER — Research Platform v1
==========================================
Disciplined architecture for Macro F1 optimization.

Key Features:
- Multi-mode evaluation (Center, Energy, Sliding)
- Diagnostic reporting (Per-class F1, Precision/Recall, Prediction CSV)
- Configurable loss (Weighted CE with Effective-Number weighting)
- Mask-aware Mean+Std pooling options
- Scientific metadata tracking (Valid length, duration, confidence)
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
from torch.optim.swa_utils import AveragedModel
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
    # Basic
    SR = 16000
    MAX_LEN = 80000 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    
    # Selection & Optimization
    SELECTION_METRIC = "macro_f1" # options: macro_f1, uar, accuracy
    PHASE1_EPOCHS = 4
    PHASE2_EPOCHS = 26
    SWA_START = 18
    LR_BACKBONE_LORA = 5e-6
    LR_HEAD = 2e-5
    BATCH_SIZE = 14
    
    # Architecture
    MODEL_NAME = "microsoft/wavlm-base-plus"
    POOLING = "mean_std" # options: attention, mean_std
    LAYER_FUSION = "last6"
    
    # Objective (Next Experiment Defaults)
    USE_SUPCON = False
    SUPCON_WEIGHT = 0.4
    USE_BALANCED_BATCH_SAMPLER = False
    USE_FOCAL_LOSS = False
    USE_WEIGHTED_CE = True
    CLASS_WEIGHT_MODE = "effective_num" # none, freq, effective_num
    USE_SWA = False
    
    # Preprocessing
    TRAIN_CROP_MODE = "center" # random, center, energy
    VAL_CROP_MODE = "center"
    TRIM_AUDIO = True
    TRIM_TOP_DB = 30
    
    # Augmentation
    AUGMENTATION_PROFILE = "none" # none, safe, current
    
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
# 2. FEATURE EXTRACTION & UTILS
# ─────────────────────────────────────────────────────────

def extract_prosody(y, sr=16000):
    """4-dim prosody from real segment. NaN safe."""
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

def get_class_weights(labels, mode="effective_num", beta=0.999):
    counts = Counter(labels)
    keys = sorted(counts.keys())
    counts_arr = np.array([counts[k] for k in keys])
    
    if mode == "freq":
        weights = 1.0 / counts_arr
    elif mode == "effective_num":
        effective_num = 1.0 - np.power(beta, counts_arr)
        weights = (1.0 - beta) / np.array(effective_num)
    else:
        weights = np.ones(len(keys))
    
    weights = weights / np.mean(weights) # Normalized to mean 1
    return torch.tensor(weights, dtype=torch.float32)

def energy_crop(y, target_len):
    if len(y) <= target_len: return y, 0
    # Find segment with highest energy
    win = target_len
    hop = target_len // 4
    energies = [np.sum(y[i:i+win]**2) for i in range(0, len(y)-win+1, hop)]
    best_idx = np.argmax(energies) * hop
    return y[best_idx : best_idx + win], best_idx

# ─────────────────────────────────────────────────────────
# 3. DATASET
# ─────────────────────────────────────────────────────────

class ResearchDataset(Dataset):
    def __init__(self, df, path_map, config, augment=False):
        self.df = df.reset_index(drop=True)
        self.path_map = path_map
        self.cfg = config
        self.augment = augment
        
        # Profile-based augmentation
        if config.AUGMENTATION_PROFILE == "safe":
            self.aug = Compose([AddGaussianNoise(p=0.2), Gain(p=0.2)])
        elif config.AUGMENTATION_PROFILE == "current":
            from audiomentations import PitchShift
            self.aug = Compose([AddGaussianNoise(p=0.35), PitchShift(p=0.35)])
        else:
            self.aug = None

    def __len__(self): return len(self.df)

    def _load_process(self, idx):
        row = self.df.iloc[idx]
        fname = Path(row["audio_relpath"]).name
        if fname not in self.path_map: return None
        
        try:
            wav, _ = librosa.load(self.path_map[fname][0], sr=self.cfg.SR)
            if self.cfg.TRIM_AUDIO:
                wav, _ = librosa.effects.trim(wav, top_db=self.cfg.TRIM_TOP_DB)
            
            orig_len = len(wav)
            duration = orig_len / self.cfg.SR
            
            # Cropping Logic
            mode = self.cfg.TRAIN_CROP_MODE if self.augment else self.cfg.VAL_CROP_MODE
            if orig_len > self.cfg.MAX_LEN:
                if mode == "random":
                    s = random.randint(0, orig_len - self.cfg.MAX_LEN)
                    yt = wav[s : s + self.cfg.MAX_LEN]
                elif mode == "energy":
                    yt, s = energy_crop(wav, self.cfg.MAX_LEN)
                else: # center
                    s = (orig_len - self.cfg.MAX_LEN) // 2
                    yt = wav[s : s + self.cfg.MAX_LEN]
                valid_len = self.cfg.MAX_LEN
            else:
                yt = wav
                valid_len = orig_len
                
            # Prosody & Augment BEFORE padding
            prosody = extract_prosody(yt, self.cfg.SR)
            if self.augment and self.aug:
                yt = self.aug(samples=yt, sample_rate=self.cfg.SR)
            
            # Padding
            final_wav = np.zeros(self.cfg.MAX_LEN, dtype=np.float32)
            final_wav[:valid_len] = yt[:valid_len]
            
            mask = np.zeros(self.cfg.MAX_LEN, dtype=np.int64)
            mask[:valid_len] = 1
            
            return {
                "wav": torch.tensor(final_wav, dtype=torch.float32),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
                "prosody": torch.tensor(prosody, dtype=torch.float32),
                "label": torch.tensor(int(row["lid"]), dtype=torch.long),
                "meta": {
                    "path": str(row["audio_relpath"]),
                    "valid_len": valid_len,
                    "duration": duration
                }
            }
        except: return None

    def __getitem__(self, idx):
        for attempt in range(10):
            res = self._load_process(idx)
            if res: return res
            print(f"[WARNING] Load failed for index {idx} (attempt {attempt+1}). Skipping.")
            idx = (idx + 1) % len(self.df)
        raise FileNotFoundError(f"CRITICAL: Exhausted 10 retries. Last attempted index: {idx}")

# ─────────────────────────────────────────────────────────
# 4. MODEL COMPONENTS
# ─────────────────────────────────────────────────────────

class MaskedPooling(nn.Module):
    def __init__(self, d_in, mode="attention"):
        super().__init__()
        self.mode = mode
        if mode == "attention":
            self.attn = nn.Linear(d_in, 1)
        
    def forward(self, x, mask):
        # x: [B, T, D], mask: [B, T] (bool)
        mask_expanded = mask.unsqueeze(-1)
        
        if self.mode == "mean_std":
            mu = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-5)
            var = (((x - mu.unsqueeze(1))**2) * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-5)
            return torch.cat([mu, torch.sqrt(var + 1e-6)], dim=-1)
            
        else: # attention
            logits = self.attn(x).squeeze(-1) # [B, T]
            logits = logits.masked_fill(~mask, -30000)
            weights = F.softmax(logits, dim=1).unsqueeze(-1)
            return (x * weights).sum(dim=1)

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
        
        self.proj_head = nn.Sequential(nn.Linear(d_p, 256), nn.ReLU(), nn.Linear(256, 128))
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
        # mask is [B, S] (int 1/0)
        out = self.wavlm(wav, attention_mask=mask, output_hidden_states=True)
        
        # Layer fusion
        h = out.hidden_states[-6:]
        w = F.softmax(self.layer_weights, dim=0)
        weighted = sum(w[i] * h[i] for i in range(len(h)))
        
        # Pooling
        f_mask = self._get_frame_mask(weighted.shape[1], mask)
        pooled = self.pooler(weighted, f_mask)
        
        proj = F.normalize(self.proj_head(pooled), dim=-1)
        logits = self.classifier(torch.cat([pooled, prosody], dim=-1))
        return logits, proj

# ─────────────────────────────────────────────────────────
# 5. DIAGNOSTICS & EVALUATION
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
    
    # Prediction counts
    t_counts = Counter(truths)
    p_counts = Counter(preds)
    print("\nClass Balancing (True vs Pred):")
    for i, emo in enumerate(emotions):
        print(f"  {emo:12s}: {t_counts[i]:3d} vs {p_counts[i]:3d}")
        
    # Build CSV DataFrame
    df_rows = []
    for i in range(len(truths)):
        df_rows.append({
            "path": metas[i]["path"],
            "true": emotions[truths[i]],
            "pred": emotions[preds[i]],
            "conf": np.max(probs[i]),
            "valid_len": metas[i]["valid_len"],
            "duration": metas[i]["duration"],
            **{f"prob_{emotions[j]}": probs[i][j] for j in range(len(emotions))}
        })
    return pd.DataFrame(df_rows), mf1

def evaluate_multi_mode(model, df, path_map, cfg, emotions):
    model.eval()
    modes = ["single_center", "single_energy", "sliding_average", "sliding_topk_average", "sliding_energy_weighted"]
    results = {}
    
    for mode in modes:
        print(f"🧪 Evaluating Mode: {mode}...")
        ps, ts, prbs = [], [], []
        
        for _, row in tqdm(df.iterrows(), total=len(df), leave=False):
            fname = Path(row["audio_relpath"]).name
            if fname not in path_map: continue
            
            try:
                wav, _ = librosa.load(path_map[fname][0], sr=cfg.SR)
                if cfg.TRIM_AUDIO: wav, _ = librosa.effects.trim(wav, top_db=cfg.TRIM_TOP_DB)
                
                if mode == "single_center":
                    crops = [wav[(len(wav)-cfg.MAX_LEN)//2 : (len(wav)-cfg.MAX_LEN)//2 + cfg.MAX_LEN] if len(wav)>cfg.MAX_LEN else wav]
                    weights = [1.0]
                elif mode == "single_energy":
                    c, _ = energy_crop(wav, cfg.MAX_LEN)
                    crops = [c]
                    weights = [1.0]
                else: # sliding variants
                    if len(wav) <= cfg.MAX_LEN: 
                        crops = [wav]
                        weights = [1.0]
                    else:
                        stride = cfg.SR // 1 # 1s stride
                        starts = range(0, len(wav)-cfg.MAX_LEN+1, stride)
                        crops = [wav[s:s+cfg.MAX_LEN] for s in starts]
                        # For energy weighting
                        weights = [np.sum(c**2) for c in crops]
                
                probs_list = []
                for c in crops:
                    v_len = min(len(c), cfg.MAX_LEN)
                    seg = np.zeros(cfg.MAX_LEN, dtype=np.float32)
                    seg[:v_len] = c[:v_len]
                    m = torch.zeros((1, cfg.MAX_LEN), dtype=torch.long)
                    m[0, :v_len] = 1
                    p = extract_prosody(c, cfg.SR)
                    with torch.no_grad(), autocast(cfg.DEVICE):
                        l, _ = model(torch.tensor(seg).unsqueeze(0).to(cfg.DEVICE), m.to(cfg.DEVICE), torch.tensor(p).unsqueeze(0).to(cfg.DEVICE))
                    probs_list.append(F.softmax(l, dim=-1).cpu().numpy()[0])
                
                probs_arr = np.array(probs_list)
                if mode == "sliding_topk_average":
                    # Take top 3 most confident windows
                    confs = np.max(probs_arr, axis=1)
                    top_k = min(3, len(probs_arr))
                    idx = np.argsort(confs)[-top_k:]
                    final_prob = np.mean(probs_arr[idx], axis=0)
                elif mode == "sliding_energy_weighted":
                    w = np.array(weights) / (np.sum(weights) + 1e-6)
                    final_prob = np.sum(probs_arr * w.reshape(-1, 1), axis=0)
                else:
                    final_prob = np.mean(probs_arr, axis=0)
                    
                prbs.append(final_prob)
                ps.append(np.argmax(final_prob))
                ts.append(int(row["lid"]))
            except: pass
            
        results[mode] = {
            "acc": accuracy_score(ts, ps),
            "mf1": f1_score(ts, ps, average="macro"),
            "wf1": f1_score(ts, ps, average="weighted"),
            "uar": balanced_accuracy_score(ts, ps)
        }
    
    print("\n🏆 Final Multi-Mode Comparison:")
    rep_df = pd.DataFrame(results).T
    print(rep_df[["acc", "mf1", "wf1", "uar"]])

# ─────────────────────────────────────────────────────────
# 6. TRAINING ENGINE
# ─────────────────────────────────────────────────────────

def train():
    cfg = ResearchConfig()
    seed_everything(cfg.SEED)
    
    colab_root = Path("/content/drive/MyDrive/Thesis Project")
    csv_p = colab_root / "data/processed/splits/text_hc"
    results_dir = colab_root / "experiments" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Path Mapping
    pm = defaultdict(list)
    for p in colab_root.rglob("*.wav"): pm[p.name].append(str(p))
    if not pm: 
        print("Searching for Thesis_Audio_Full.zip recursively on Drive...")
        zname = "Thesis_Audio_Full.zip"
        zpath = None
        for root, _, files in os.walk("/content/drive/MyDrive"):
            if zname in files:
                zpath = os.path.join(root, zname)
                break
        if not zpath: raise FileNotFoundError(f"CRITICAL: {zname} not found anywhere on Google Drive.")
        
        print(f"📦 Extracting {zpath} to local runtime...")
        import zipfile
        with zipfile.ZipFile(zpath, 'r') as z: z.extractall("/content/dataset")
        for p in Path("/content/dataset").rglob("*.wav"): pm[p.name].append(str(p))

    # Duplicate check
    dups = {name: paths for name, paths in pm.items() if len(paths) > 1}
    print(f"[PATH CHECK] Total basenames: {len(pm)}")
    print(f"[PATH CHECK] Duplicate basenames: {len(dups)}")
    if dups:
        print("[WARNING] Basename mapping may be unsafe. First 10 duplicates:")
        for name in list(dups.keys())[:10]:
            print(f"  {name}: {dups[name]}")

    tr_df = pd.read_csv(csv_p / "train.csv")
    va_df = pd.read_csv(csv_p / "val.csv")
    emotions = sorted(tr_df["emotion_final"].unique())
    lid = {e: i for i, e in enumerate(emotions)}
    tr_df["lid"] = tr_df["emotion_final"].map(lid)
    va_df["lid"] = va_df["emotion_final"].map(lid)
    
    # Loss Weights
    weights = get_class_weights(tr_df["lid"].values, mode=cfg.CLASS_WEIGHT_MODE).to(cfg.DEVICE)
    
    tr_ds = ResearchDataset(tr_df, pm, cfg, augment=True)
    va_ds = ResearchDataset(va_df, pm, cfg, augment=False)
    
    tr_loader = DataLoader(tr_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=16)
    
    model = ResearchModel(cfg, len(emotions)).to(cfg.DEVICE)
    scaler = GradScaler(cfg.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights) if cfg.USE_WEIGHTED_CE else nn.CrossEntropyLoss()
    
    # PHASE 1: Warmup
    print(f"\n🔥 PHASE 1: Warmup ({cfg.PHASE1_EPOCHS} epochs)")
    for n, p in model.wavlm.named_parameters(): p.requires_grad = False
    opt1 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    
    for ep in range(1, cfg.PHASE1_EPOCHS + 1):
        model.train(); ep_loss = 0.0
        for b in tqdm(tr_loader, desc=f"Ph1 Ep{ep}", leave=False):
            w, m, p, l = b["wav"].to(cfg.DEVICE), b["attention_mask"].to(cfg.DEVICE), b["prosody"].to(cfg.DEVICE), b["label"].to(cfg.DEVICE)
            with autocast(cfg.DEVICE):
                logits, _ = model(w, m, p)
                loss = criterion(logits, l)
            opt1.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt1); scaler.update()
            ep_loss += loss.item()
        
        # Phase 1 Diagnostic
        model.eval(); ts_v, ps_v, prbs_v, metas_v = [], [], [], []
        with torch.no_grad():
            for b in va_loader:
                w, m, p, l = b["wav"].to(cfg.DEVICE), b["attention_mask"].to(cfg.DEVICE), b["prosody"].to(cfg.DEVICE), b["label"].to(cfg.DEVICE)
                logits, _ = model(w, m, p)
                prbs_v.extend(F.softmax(logits, dim=-1).cpu().numpy())
                ps_v.extend(logits.argmax(1).cpu().numpy())
                ts_v.extend(l.cpu().numpy())
                metas_v.extend([dict(zip(b['meta'].keys(), values)) for values in zip(*b['meta'].values())])
        
        print(f"\nPh1 Ep {ep} | Loss: {ep_loss/len(tr_loader):.3f}")
        run_diagnostics(ts_v, ps_v, prbs_v, metas_v, emotions, prefix=f"Ph1_Ep{ep}")

    # PHASE 2: Core Research
    print(f"\n🚀 PHASE 2: Research Run ({cfg.PHASE2_EPOCHS} epochs)")
    for n, p in model.wavlm.named_parameters(): p.requires_grad = "lora_" in n
    opt2 = torch.optim.AdamW([
        {"params": [p for n, p in model.wavlm.named_parameters() if "lora_" in n], "lr": cfg.LR_BACKBONE_LORA},
        {"params": [p for n, p in model.named_parameters() if "wavlm" not in n], "lr": cfg.LR_HEAD}
    ], weight_decay=0.01)
    
    best_metric_val = -1.0
    for ep in range(1, cfg.PHASE2_EPOCHS + 1):
        model.train(); ep_loss = 0.0
        for b in tqdm(tr_loader, desc=f"Ph2 Ep{ep}", leave=False):
            w, m, p, l = b["wav"].to(cfg.DEVICE), b["attention_mask"].to(cfg.DEVICE), b["prosody"].to(cfg.DEVICE), b["label"].to(cfg.DEVICE)
            with autocast(cfg.DEVICE):
                logits, _ = model(w, m, p)
                loss = criterion(logits, l)
            opt2.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt2); scaler.update()
            ep_loss += loss.item()
            
        # Eval
        model.eval(); ts, ps, prbs, metas = [], [], [], []
        with torch.no_grad():
            for b in va_loader:
                w, m, p, l = b["wav"].to(cfg.DEVICE), b["attention_mask"].to(cfg.DEVICE), b["prosody"].to(cfg.DEVICE), b["label"].to(cfg.DEVICE)
                logits, _ = model(w, m, p)
                prbs.extend(F.softmax(logits, dim=-1).cpu().numpy())
                ps.extend(logits.argmax(1).cpu().numpy())
                ts.extend(l.cpu().numpy())
                metas.extend([dict(zip(b['meta'].keys(), values)) for values in zip(*b['meta'].values())])
        
        diag_df, mf1 = run_diagnostics(ts, ps, prbs, metas, emotions)
        
        # Checkpoint Selection
        metric = mf1 if cfg.SELECTION_METRIC == "macro_f1" else accuracy_score(ts, ps)
        if metric > best_metric_val:
            best_metric_val = metric
            torch.save(model.state_dict(), results_dir / "best_model.pt")
            diag_df.to_csv(results_dir / "best_predictions.csv", index=False)
            print(f"🌟 New Best {cfg.SELECTION_METRIC}: {metric:.4f}")

    # Final Mode Report
    model.load_state_dict(torch.load(results_dir / "best_model.pt"))
    evaluate_multi_mode(model, va_df, pm, cfg, emotions)

if __name__ == "__main__":
    train()
