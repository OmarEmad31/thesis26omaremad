"""
Audio v8 — Egyptian Arabic SER
================================
Root cause of all Phase 2 failures:
  LoRA adaptation + SupCon on balanced batches eval'd on imbalanced val = spiral down.

Solution: REMOVE LoRA fine-tuning entirely. REMOVE SupCon.
Instead: frozen WavLM-Base-Plus + 2-layer Transformer head trained from scratch.

This gives:
  - NO Phase 2 instability (no LoRA, no backbone disruption)
  - 10M-param Transformer head for powerful temporal emotion modeling
  - WavLM features stay pristine (94M params, expertly pre-trained)
  - Full 25-epoch single-phase training (no disruption transitions)
  - 74 batches/epoch with improved BalancedBatchSampler (max-class based)
  - Speed perturbation (0.9x/1.1x resampling) as key new augmentation

Heavy augmentations:
  1. Speed perturbation via resampling (0.9x, 1.1x) — NOT TimeStretch (NaN-safe)
  2. Gaussian noise
  3. PitchShift
  4. SpecAugment time masking
  5. Intra-class mixup (rare classes only)

Test-time augmentation:
  Average predictions over original + 0.9x speed + 1.1x speed crops
"""
import os, sys, subprocess, zipfile, math, random
from collections import defaultdict

def install_deps():
    try:
        import audiomentations, transformers
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "audiomentations", "transformers", "-q"])

if "google.colab" in sys.modules or os.path.exists("/content"):
    install_deps()

import torch, torch.nn as nn, torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim.swa_utils import AveragedModel
import pandas as pd, numpy as np, librosa
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from transformers import WavLMModel, get_cosine_schedule_with_warmup
from audiomentations import Compose, AddGaussianNoise, PitchShift

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
SR             = 16000
MAX_LEN        = 80000        # 5-second window
K_PER_CLASS    = 2            # samples per class per batch
NUM_CLASSES    = 7
BATCH_SIZE     = K_PER_CLASS * NUM_CLASSES   # = 14
NUM_EPOCHS     = 25
SWA_START      = 18
FOCAL_GAMMA    = 2.0
MODEL_NAME     = "microsoft/wavlm-base-plus"
RARE_THRESHOLD = 50           # classes with < 50 train samples get mixup


# ─────────────────────────────────────────────────────────
# BALANCED BATCH SAMPLER (max-class based)
# n_batches = max_count // k  → uses ALL samples every epoch
# Smaller classes cycle with repetition → Fear seen ~5x per epoch
# ─────────────────────────────────────────────────────────
class BalancedBatchSampler(Sampler):
    def __init__(self, labels, k=K_PER_CLASS):
        self.k = k
        self.class_indices = defaultdict(list)
        for i, lbl in enumerate(labels):
            self.class_indices[int(lbl)].append(i)
        self.classes   = sorted(self.class_indices.keys())
        max_count      = max(len(v) for v in self.class_indices.values())
        self.n_batches = max_count // k   # 148 // 2 = 74 batches

    def __iter__(self):
        # Build cycling pools so small classes repeat to fill 74 batches
        pools = {}
        for c, idxs in self.class_indices.items():
            shuffled = random.sample(idxs, len(idxs))
            needed   = self.n_batches * self.k
            pools[c] = (shuffled * (needed // len(idxs) + 1))[:needed]

        for b in range(self.n_batches):
            batch = []
            for c in self.classes:
                batch.extend(pools[c][b * self.k: b * self.k + self.k])
            random.shuffle(batch)
            yield batch

    def __len__(self): return self.n_batches


# ─────────────────────────────────────────────────────────
# FOCAL LOSS
# ─────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, labels):
        ce  = F.cross_entropy(logits, labels, reduction="none")
        p_t = torch.exp(-ce)
        return (((1 - p_t) ** self.gamma) * ce).mean()


# ─────────────────────────────────────────────────────────
# ATTENTION POOLING (zero-init = mean pool at start)
# ─────────────────────────────────────────────────────────
class AttentionPool(nn.Module):
    def __init__(self, d=768):
        super().__init__()
        self.attn = nn.Linear(d, 1)
        nn.init.zeros_(self.attn.weight)
        nn.init.zeros_(self.attn.bias)

    def forward(self, x):                   # [B, T, D]
        w = F.softmax(self.attn(x), dim=1)  # [B, T, 1]
        return (x * w).sum(dim=1)           # [B, D]


# ─────────────────────────────────────────────────────────
# MODEL: Frozen WavLM + Transformer Head
# No LoRA, no backbone disruption, pure head training.
# ─────────────────────────────────────────────────────────
class WavLMSER_v8(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        # Fully frozen backbone — no LoRA, no gradient through WavLM
        self.wavlm = WavLMModel.from_pretrained(MODEL_NAME, output_hidden_states=True)
        for p in self.wavlm.parameters():
            p.requires_grad = False

        # Learnable weights for last 6 WavLM layers
        self.layer_weights = nn.Parameter(torch.ones(6))

        # 2-layer Transformer head: temporal emotion modeling over WavLM frames
        enc_layer = nn.TransformerEncoderLayer(
            d_model=768, nhead=8, dim_feedforward=2048,
            dropout=0.1, batch_first=True
        )
        self.transformer_head = nn.TransformerEncoder(enc_layer, num_layers=2)
        # Identity initialization: zero out output projections so each layer
        # starts as an identity function (x + 0 = x via residuals).
        # This ensures Ep1 starts at ~28% not 22% (random scramble).
        for layer in self.transformer_head.layers:
            nn.init.zeros_(layer.self_attn.out_proj.weight)
            nn.init.zeros_(layer.self_attn.out_proj.bias)
            nn.init.zeros_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)


        # Attention pooling (zero-init = mean pool at start)
        self.attn_pool = AttentionPool(768)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(768 + 4, 512),
            nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(512, num_labels)
        )

    def extract_features(self, wav):
        """Frozen WavLM forward — builds NO computation graph."""
        wav  = (wav - wav.mean(-1, keepdim=True)) / (wav.std(-1, keepdim=True) + 1e-6)
        mask = torch.ones(wav.shape[:2], device=wav.device)
        with torch.no_grad():
            out = self.wavlm(wav, attention_mask=mask, output_hidden_states=True)
        # Weighted sum of last 6 layers — graph starts HERE (layer_weights has grad)
        hidden = out.hidden_states[-6:]           # 6 × [B, T, 768]
        w = F.softmax(self.layer_weights, dim=0)
        return sum(w[i] * h.detach() for i, h in enumerate(hidden))  # [B, T, 768]

    def forward(self, wav, prosody):
        x      = self.extract_features(wav)            # [B, T, 768]
        x      = self.transformer_head(x)              # [B, T, 768]
        pooled = self.attn_pool(x)                     # [B, 768]
        return self.classifier(torch.cat([pooled, prosody], dim=-1))


# ─────────────────────────────────────────────────────────
# PROSODY: 4-dim, NaN-safe
# ─────────────────────────────────────────────────────────
def extract_prosody(y: np.ndarray) -> np.ndarray:
    rms = float(np.mean(librosa.feature.rms(y=y)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    f0  = librosa.yin(y, fmin=65, fmax=2093)
    f0m = float(np.nanmean(f0)) / 500.0
    f0s = float(np.nanstd(f0))  / 100.0
    vec = np.array([rms, zcr, f0m, f0s], dtype=np.float32)
    return np.clip(np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0), -10, 10)


# ─────────────────────────────────────────────────────────
# SAFE SPEED PERTURBATION (resampling, no NaN like TimeStretch)
# Changes both speed and pitch — valid SER augmentation.
# ─────────────────────────────────────────────────────────
def safe_speed_perturb(y: np.ndarray, factor: float) -> np.ndarray:
    try:
        yr = librosa.resample(y, orig_sr=SR, target_sr=int(SR * factor))
        if len(yr) >= MAX_LEN: return yr[:MAX_LEN]
        return np.pad(yr, (0, MAX_LEN - len(yr)))
    except Exception:
        return y


# ─────────────────────────────────────────────────────────
# DATASET with aggressive safe augmentation
# ─────────────────────────────────────────────────────────
class AudioDataset(Dataset):
    def __init__(self, df, path_map, augment=False, rare_lids=None):
        self.df       = df.reset_index(drop=True)
        self.path_map = path_map
        self.augment  = augment
        self.rare_lids = set(rare_lids or [])
        self.aug_pipe = Compose([AddGaussianNoise(p=0.35), PitchShift(p=0.35)])
        self.class_idx = defaultdict(list)
        for i, row in self.df.iterrows():
            self.class_idx[int(row["lid"])].append(i)

    def __len__(self): return len(self.df)

    def _load_audio(self, idx):
        row   = self.df.iloc[idx]
        fname = Path(row["audio_relpath"]).name
        if fname not in self.path_map: return None, None
        try:
            wav, _ = librosa.load(self.path_map[fname], sr=SR)
            yt, _  = librosa.effects.trim(wav, top_db=30)
            if len(yt) < SR // 2: return None, None
            if len(yt) > MAX_LEN:
                s  = random.randint(0, len(yt) - MAX_LEN) if self.augment else 0
                yt = yt[s:s + MAX_LEN]
            else:
                yt = np.pad(yt, (0, MAX_LEN - len(yt)))
            return yt, int(row["lid"])
        except Exception:
            return None, None

    def _augment(self, yt, lid):
        # Speed perturbation (safe resampling, no TimeStretch)
        if random.random() < 0.4:
            factor = random.choice([0.9, 1.1])
            yt = safe_speed_perturb(yt, factor)

        # Intra-class mixup for rare classes
        if lid in self.rare_lids and random.random() < 0.5:
            pidx = random.choice(self.class_idx[lid])
            yt2, _ = self._load_audio(pidx)
            if yt2 is not None:
                a = np.random.beta(0.4, 0.4)
                yt = a * yt + (1 - a) * yt2

        # SpecAugment time mask (up to 15%)
        T = len(yt); w = int(T * 0.15 * random.random())
        if w > 0:
            s = random.randint(0, T - w)
            yt = yt.copy(); yt[s:s + w] = 0.0

        # Gaussian noise + PitchShift (via audiomentations)
        return self.aug_pipe(samples=yt.astype(np.float32), sample_rate=SR)

    def _load(self, idx):
        yt, lid = self._load_audio(idx)
        if yt is None: return None
        try:
            if self.augment:
                yt = self._augment(yt, lid)
            return {
                "wav":     torch.tensor(yt,                  dtype=torch.float32),
                "prosody": torch.tensor(extract_prosody(yt), dtype=torch.float32),
                "label":   torch.tensor(lid,                 dtype=torch.long),
            }
        except Exception:
            return None

    def __getitem__(self, idx):
        for _ in range(len(self.df)):
            s = self._load(idx)
            if s is not None: return s
            idx = (idx + 1) % len(self.df)
        raise FileNotFoundError("Could not load audio.")


# ─────────────────────────────────────────────────────────
# PATH MAP
# ─────────────────────────────────────────────────────────
def get_path_map(colab_root):
    pm = {f.name: str(f) for f in colab_root.rglob("*.wav")}
    if pm:
        print(f"[SUCCESS] Found {len(pm)} wav files."); return pm
    zname = "Thesis_Audio_Full.zip"; zpath = None
    for root, _, files in os.walk("/content/drive/MyDrive"):
        if zname in files: zpath = os.path.join(root, zname); break
    if not zpath: raise FileNotFoundError(f"{zname} not found.")
    with zipfile.ZipFile(zpath) as z: z.extractall("/content/dataset")
    pm = {f.name: str(f) for f in Path("/content/dataset").rglob("*.wav")}
    print(f"[SUCCESS] Extracted {len(pm)} files."); return pm


# ─────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────
def fast_eval(model, loader, device):
    model.eval(); ps, ts = [], []
    with torch.no_grad():
        for b in loader:
            with autocast("cuda"):
                logits = model(b["wav"].to(device), b["prosody"].to(device))
            ps.extend(logits.argmax(1).cpu().numpy())
            ts.extend(b["label"].numpy())
    return accuracy_score(ts, ps), f1_score(ts, ps, average="macro", zero_division=0)


def tta_eval(model, df, path_map, device):
    """Test-time augmentation: average over original + 0.9x + 1.1x speed."""
    model.eval(); preds, truths = [], []
    speeds = [1.0, 0.9, 1.1]
    for _, row in tqdm(df.iterrows(), total=len(df), desc="TTA Eval", leave=False):
        fname = Path(row["audio_relpath"]).name
        if fname not in path_map: continue
        try:
            wav, _ = librosa.load(path_map[fname], sr=SR)
            yt, _  = librosa.effects.trim(wav, top_db=30)
            if len(yt) < SR // 2: continue
            logits_list = []
            for spd in speeds:
                seg = safe_speed_perturb(yt, spd) if spd != 1.0 else (
                    yt[:MAX_LEN] if len(yt) >= MAX_LEN else
                    np.pad(yt, (0, MAX_LEN - len(yt)))
                )
                p = extract_prosody(seg)
                with torch.no_grad(), autocast("cuda"):
                    l = model(torch.tensor(seg).unsqueeze(0).to(device),
                               torch.tensor(p).unsqueeze(0).to(device))
                logits_list.append(F.softmax(l, dim=-1))
            preds.append(torch.stack(logits_list).mean(0).argmax(1).item())
            truths.append(int(row["lid"]))
        except Exception: pass
    return (accuracy_score(truths, preds),
            f1_score(truths, preds, average="macro", zero_division=0))


# ─────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────
def train():
    device     = "cuda"
    colab_root = Path("/content/drive/MyDrive/Thesis Project")
    csv_p      = colab_root / "data/processed/splits/text_hc"
    save_path  = colab_root / "wavlm_v8_best.pt"

    path_map = get_path_map(colab_root)

    tr_df = pd.read_csv(csv_p / "train.csv")
    va_df = pd.read_csv(csv_p / "val.csv")
    lid   = {l: i for i, l in enumerate(sorted(tr_df["emotion_final"].unique()))}
    tr_df["lid"] = tr_df["emotion_final"].map(lid)
    va_df["lid"] = va_df["emotion_final"].map(lid)

    class_counts   = tr_df["emotion_final"].value_counts()
    rare_lids      = {lid[l] for l in class_counts[class_counts < RARE_THRESHOLD].index
                      if l in lid}

    print(f"\n[DATA] Train: {len(tr_df)} | Val: {len(va_df)} | Classes: {len(lid)}")
    print("[DATA] Train distribution:")
    for emo, cnt in class_counts.items():
        tag = " ← mixup" if lid.get(emo, -1) in rare_lids else ""
        print(f"  {emo:12s}: {cnt:3d} ({100*cnt/len(tr_df):.1f}%){tag}")

    tr_ds = AudioDataset(tr_df, path_map, augment=True,  rare_lids=rare_lids)
    va_ds = AudioDataset(va_df, path_map, augment=False, rare_lids=None)

    def make_sampler():
        return BalancedBatchSampler(tr_df["lid"].values, k=K_PER_CLASS)

    smp = make_sampler()
    print(f"\n[DATA] BalancedBatchSampler: {K_PER_CLASS}/class × {NUM_CLASSES} = "
          f"batch {BATCH_SIZE} | {len(smp)} batches/epoch (max-class based)")

    va_loader = DataLoader(va_ds, batch_size=16, num_workers=0)

    model     = WavLMSER_v8(num_labels=len(lid)).to(device)
    swa_model = AveragedModel(model)
    crit      = FocalLoss(gamma=FOCAL_GAMMA)
    scaler    = GradScaler("cuda")
    best_acc  = 0.0; swa_active = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[MODEL] Trainable params: {sum(p.numel() for p in trainable_params):,} "
          f"(WavLM frozen, head + layer_weights + attn_pool trained)")

    opt = torch.optim.AdamW([
        {"params": model.layer_weights,                     "lr": 1e-2},
        {"params": model.transformer_head.parameters(),     "lr": 1e-3},
        {"params": model.attn_pool.parameters(),            "lr": 1e-3},
        {"params": model.classifier.parameters(),           "lr": 1e-3},
    ], weight_decay=0.01)
    sch = get_cosine_schedule_with_warmup(opt, len(smp), len(smp) * NUM_EPOCHS)

    print(f"\n{'='*60}")
    print(f"  v8 Training — {NUM_EPOCHS} epochs | Focal Loss | SWA from Ep{SWA_START}")
    print(f"  Augmentation: Speed(0.9x/1.1x) + Noise + Pitch + Mask + Mixup")
    print(f"  Eval at end: TTA (original + 0.9x + 1.1x speed average)")
    print(f"{'='*60}")

    for ep in range(1, NUM_EPOCHS + 1):
        smp       = make_sampler()
        tr_loader = DataLoader(tr_ds, batch_sampler=smp, num_workers=0)
        model.train(); ep_loss = 0.0

        for b in tqdm(tr_loader, desc=f"Ep {ep:02d}", leave=False):
            with autocast("cuda"):
                loss = crit(model(b["wav"].to(device), b["prosody"].to(device)),
                            b["label"].to(device))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            ep_loss += loss.item()

        if ep >= SWA_START:
            swa_model.update_parameters(model); swa_active = True

        acc, f1 = fast_eval(model, va_loader, device)
        tag = ""
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path); tag = "  *** BEST ***"
        print(f"Ep {ep:02d} | Loss {ep_loss/len(smp):.3f} | "
              f"Val Acc {acc:.3f} | F1 {f1:.3f}{tag}")

    # ═══════════════════════════════════════════════════
    # SWA
    # ═══════════════════════════════════════════════════
    if swa_active:
        print("\n[SWA] Finalizing averaged model...")
        swa_model.train()
        smp = make_sampler()
        with torch.no_grad():
            tr_loader = DataLoader(tr_ds, batch_sampler=smp, num_workers=0)
            for b in tqdm(tr_loader, desc="SWA BN", leave=False):
                with autocast("cuda"):
                    swa_model(b["wav"].to(device), b["prosody"].to(device))
        swa_acc, swa_f1 = fast_eval(swa_model, va_loader, device)
        print(f"[SWA] Val Acc {swa_acc:.3f} | F1 {swa_f1:.3f}")
        if swa_acc > best_acc:
            best_acc = swa_acc
            torch.save(swa_model.state_dict(), save_path)
            print("[SWA] SWA is new best — saved.")

    # ═══════════════════════════════════════════════════
    # TTA EVALUATION (3-way speed average)
    # ═══════════════════════════════════════════════════
    print("\n[TTA] Test-time augmentation evaluation...")
    model.load_state_dict(torch.load(save_path, map_location=device))
    tta_acc, tta_f1 = tta_eval(model, va_df, path_map, device)

    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS")
    print(f"  Best Single-Crop Val Acc : {best_acc:.4f}")
    print(f"  TTA Val Acc (3-way)      : {tta_acc:.4f}")
    print(f"  TTA Val F1               : {tta_f1:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
