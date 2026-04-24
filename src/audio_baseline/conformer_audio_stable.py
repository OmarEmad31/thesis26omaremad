"""
Audio v9 — Egyptian Arabic SER
================================
Based on backup_sota_398.py (confirmed Phase1 Ep3=38.1%, Ph2 Ep2=39.8%).

ONLY 4 changes from the backup (everything else identical):
  1. REMOVE SupCon — root cause of every Phase 2 decline. Focal Loss only.
  2. BalancedBatchSampler: max-class based (74 batches/epoch vs 15 before)
     Fear cycles 5x per epoch → more gradient signal for rare classes
  3. Speed perturbation augmentation: 0.9x / 1.1x via resampling (NaN-safe)
  4. TTA eval: average over original + 0.9x + 1.1x speed crops

Phase 1: 5 epochs (was 4) — one more warmup epoch since Phase 1 is safe
Phase 2: 15 epochs (was 26) — shorter, SupCon removal means no recovery wait needed
"""
import os, sys, subprocess, zipfile, random
from collections import defaultdict

def install_deps():
    pkgs = []
    try: import audiomentations
    except ImportError: pkgs.append("audiomentations")
    try: import peft
    except ImportError: pkgs.append("peft")
    try: import transformers
    except ImportError: pkgs.append("transformers")
    try: import noisereduce
    except ImportError: pkgs.append("noisereduce")
    try: import pyloudnorm
    except ImportError: pkgs.append("pyloudnorm")
    if pkgs:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *pkgs, "-q"])

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
from peft import LoraConfig, get_peft_model
from audiomentations import Compose, AddGaussianNoise, PitchShift
import noisereduce as nr
import pyloudnorm as pyln

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
SR             = 16000
MAX_LEN        = 80000
K_PER_CLASS    = 2
NUM_CLASSES    = 7
BATCH_SIZE     = K_PER_CLASS * NUM_CLASSES   # = 14
BATCH_FROZEN   = 32
PHASE1_EPOCHS  = 5      # +1 extra warmup vs backup
PHASE2_EPOCHS  = 15     # shorter — no SupCon recovery wait needed
SWA_START      = 10     # earlier SWA since Phase 2 is shorter
FOCAL_GAMMA    = 2.0
RARE_THRESHOLD = 50
MODEL_NAME     = "microsoft/wavlm-base-plus"


# ─────────────────────────────────────────────────────────
# BALANCED BATCH SAMPLER — max-class based
# n_batches = max_count // k = 148//2 = 74 batches/epoch
# Small classes cycle with repetition (Fear seen ~5x per epoch)
# ─────────────────────────────────────────────────────────
class BalancedBatchSampler(Sampler):
    def __init__(self, labels, k=K_PER_CLASS):
        self.k = k
        self.class_indices = defaultdict(list)
        for i, lbl in enumerate(labels):
            self.class_indices[int(lbl)].append(i)
        self.classes   = sorted(self.class_indices.keys())
        max_count      = max(len(v) for v in self.class_indices.values())
        self.n_batches = max_count // k   # 148//2 = 74

    def __iter__(self):
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
# FOCAL LOSS (only loss — SupCon removed)
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
# ATTENTION POOLING — zero-init = mean pool at start
# ─────────────────────────────────────────────────────────
class AttentionPool(nn.Module):
    def __init__(self, d=768):
        super().__init__()
        self.attn = nn.Linear(d, 1)

    def forward(self, x):
        w = F.softmax(self.attn(x), dim=1)
        return (x * w).sum(dim=1)


# ─────────────────────────────────────────────────────────
# MODEL — identical to backup_sota_398, proj_head removed
# ─────────────────────────────────────────────────────────
class WavLMEliteSER(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        base = WavLMModel.from_pretrained(MODEL_NAME, output_hidden_states=True)
        cfg  = LoraConfig(r=8, lora_alpha=16,
                          target_modules=["q_proj", "v_proj"],
                          lora_dropout=0.05, bias="none")
        self.wavlm = get_peft_model(base, cfg)

        self.layer_weights = nn.Parameter(torch.ones(6))

        self.attn_pool = AttentionPool(768)
        nn.init.zeros_(self.attn_pool.attn.weight)
        nn.init.zeros_(self.attn_pool.attn.bias)

        self.classifier = nn.Sequential(
            nn.Linear(768 + 4, 512),
            nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        )

    def freeze_backbone(self):
        for n, p in self.wavlm.named_parameters():
            p.requires_grad = "lora_" in n

    def unfreeze_lora(self):
        for n, p in self.wavlm.named_parameters():
            p.requires_grad = "lora_" in n

    def forward(self, wav, prosody):
        wav  = (wav - wav.mean(-1, keepdim=True)) / (wav.std(-1, keepdim=True) + 1e-6)
        mask = torch.ones(wav.shape[:2], device=wav.device)
        out  = self.wavlm(wav, attention_mask=mask, output_hidden_states=True)

        hidden = out.hidden_states[-6:]
        w = F.softmax(self.layer_weights, dim=0)
        weighted = sum(w[i] * hidden[i] for i in range(6))

        pooled = self.attn_pool(weighted)
        return self.classifier(torch.cat([pooled, prosody], dim=-1))


# ─────────────────────────────────────────────────────────
# PROSODY
# ─────────────────────────────────────────────────────────
def extract_prosody(y: np.ndarray) -> np.ndarray:
    rms = float(np.mean(librosa.feature.rms(y=y)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    f0  = librosa.yin(y, fmin=65, fmax=2093)
    f0m = float(np.nanmean(f0)) / 500.0
    f0s = float(np.nanstd(f0)) / 100.0
    vec = np.array([rms, zcr, f0m, f0s], dtype=np.float32)
    return np.clip(np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0), -10, 10)


# ─────────────────────────────────────────────────────────
# SAFE SPEED PERTURBATION — resampling, no TimeStretch NaN
# ─────────────────────────────────────────────────────────
def safe_speed_perturb(y: np.ndarray, factor: float) -> np.ndarray:
    try:
        yr = librosa.resample(y, orig_sr=SR, target_sr=int(SR * factor))
        return yr[:MAX_LEN] if len(yr) >= MAX_LEN else np.pad(yr, (0, MAX_LEN - len(yr)))
    except Exception:
        return y


# ─────────────────────────────────────────────────────────
# ELITE PREPROCESSING PIPELINE
# Applied once per file, cached in memory. Cleans audio before
# WavLM sees it — WavLM was trained on LibriSpeech clean.
# ─────────────────────────────────────────────────────────
def elite_preprocess(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    1. Loudness normalize (-23 LUFS): standardizes amplitude across all clips
    2. Spectral noise reduction: removes stationary background noise
    3. Pre-emphasis (γ=0.97): enhances emotion-carrying formants 300-8000 Hz
    4. Trim silence (top_db=25): tighter than default 30, removes more silence
    5. Smart crop: picks the MAX_LEN window with highest RMS energy
    """
    # 1. Loudness normalization
    try:
        meter    = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y.astype(np.float64))
        if np.isfinite(loudness) and -80 < loudness < 0:
            y = pyln.normalize.loudness(
                y.astype(np.float64), loudness, -23.0).astype(np.float32)
    except Exception:
        pass

    # 2. Spectral noise reduction (prop_decrease=0.75: aggressive but not over-filtered)
    try:
        y = nr.reduce_noise(y=y, sr=sr, stationary=True,
                            prop_decrease=0.75).astype(np.float32)
    except Exception:
        pass

    # 3. Pre-emphasis filter
    y = np.append(y[0], y[1:] - 0.97 * y[:-1]).astype(np.float32)

    # 4. Trim silence (tighter threshold)
    yt, _ = librosa.effects.trim(y, top_db=25)
    if len(yt) < sr // 4:
        yt = y  # fallback: don't over-trim

    # 5. Smart crop: find MAX_LEN window with highest mean RMS energy
    if len(yt) > MAX_LEN:
        step       = sr // 4  # 250ms steps
        best_start = 0
        best_rms   = -1.0
        for s in range(0, len(yt) - MAX_LEN + 1, step):
            rms = float(np.mean(yt[s:s + MAX_LEN] ** 2))
            if rms > best_rms:
                best_rms   = rms
                best_start = s
        yt = yt[best_start:best_start + MAX_LEN]
    else:
        yt = np.pad(yt, (0, MAX_LEN - len(yt)))

    return np.clip(yt, -1.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────
# DATASET — elite preprocessing + in-memory cache
# ─────────────────────────────────────────────────────────
class AudioDataset(Dataset):
    def __init__(self, df, path_map, augment=False, rare_classes=None):
        self.df           = df.reset_index(drop=True)
        self.path_map     = path_map
        self.augment      = augment
        self.rare_classes = set(rare_classes or [])
        self.aug_pipe     = Compose([AddGaussianNoise(p=0.35), PitchShift(p=0.35)])
        self.class_idx    = defaultdict(list)
        for i, row in self.df.iterrows():
            self.class_idx[int(row["lid"])].append(i)
        # In-memory preprocessing cache: runs once per file, reused every epoch
        self._cache: dict = {}  # fname → preprocessed np.float32 [MAX_LEN]

    def __len__(self): return len(self.df)

    def _load_audio(self, idx):
        row   = self.df.iloc[idx]
        fname = Path(row["audio_relpath"]).name
        if fname not in self.path_map: return None, None
        # Return cached preprocessed audio if available
        if fname in self._cache:
            return self._cache[fname].copy(), int(row["lid"])
        try:
            raw, _ = librosa.load(self.path_map[fname], sr=SR)
            yt     = elite_preprocess(raw, SR)   # full elite pipeline
            self._cache[fname] = yt              # cache for all future epochs
            return yt.copy(), int(row["lid"])
        except Exception:
            return None, None

    def _load(self, idx):
        yt, lid = self._load_audio(idx)
        if yt is None: return None
        try:
            if self.augment:
                # Speed perturbation (NaN-safe resampling)
                if random.random() < 0.4:
                    yt = safe_speed_perturb(yt, random.choice([0.9, 1.1]))

                # Intra-class mixup for rare classes
                if lid in self.rare_classes and random.random() < 0.5:
                    peer = random.choice(self.class_idx[lid])
                    yt2, _ = self._load_audio(peer)
                    if yt2 is not None:
                        a = np.random.beta(0.4, 0.4)
                        yt = a * yt + (1 - a) * yt2

                # SpecAugment time mask
                T = len(yt); w = int(T * 0.15 * random.random())
                if w > 0:
                    s = random.randint(0, T - w)
                    yt = yt.copy(); yt[s:s + w] = 0.0

                yt = self.aug_pipe(samples=yt.astype(np.float32), sample_rate=SR)

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
    """TTA: average predictions over original + 0.9x + 1.1x speed."""
    model.eval(); preds, truths = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="TTA", leave=False):
        fname = Path(row["audio_relpath"]).name
        if fname not in path_map: continue
        try:
            wav, _ = librosa.load(path_map[fname], sr=SR)
            yt, _  = librosa.effects.trim(wav, top_db=30)
            if len(yt) < SR // 2: continue
            logits_all = []
            for factor in [1.0, 0.9, 1.1]:
                seg = safe_speed_perturb(yt, factor) if factor != 1.0 else (
                    yt[:MAX_LEN] if len(yt) >= MAX_LEN else np.pad(yt, (0, MAX_LEN - len(yt))))
                p = extract_prosody(seg)
                with torch.no_grad(), autocast("cuda"):
                    l = model(torch.tensor(seg).unsqueeze(0).to(device),
                               torch.tensor(p).unsqueeze(0).to(device))
                logits_all.append(F.softmax(l, dim=-1))
            preds.append(torch.stack(logits_all).mean(0).argmax(1).item())
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
    save_path  = colab_root / "wavlm_v9_best.pt"

    path_map = get_path_map(colab_root)

    tr_df = pd.read_csv(csv_p / "train.csv")
    va_df = pd.read_csv(csv_p / "val.csv")
    lid   = {l: i for i, l in enumerate(sorted(tr_df["emotion_final"].unique()))}
    tr_df["lid"] = tr_df["emotion_final"].map(lid)
    va_df["lid"] = va_df["emotion_final"].map(lid)

    class_counts = tr_df["emotion_final"].value_counts()
    rare_names   = set(class_counts[class_counts < RARE_THRESHOLD].index)
    rare_lids    = {lid[l] for l in rare_names if l in lid}

    print(f"\n[DATA] Train: {len(tr_df)} | Val: {len(va_df)} | Classes: {len(lid)}")
    for emo, cnt in class_counts.items():
        tag = " ← mixup+speed" if lid.get(emo, -1) in rare_lids else ""
        print(f"  {emo:12s}: {cnt:3d} ({100*cnt/len(tr_df):.1f}%){tag}")

    tr_ds = AudioDataset(tr_df, path_map, augment=True,  rare_classes=rare_lids)
    va_ds = AudioDataset(va_df, path_map, augment=False, rare_classes=None)

    def make_sampler():
        return BalancedBatchSampler(tr_df["lid"].values, k=K_PER_CLASS)

    smp = make_sampler()
    print(f"\n[SAMPLER] {K_PER_CLASS}/class × {NUM_CLASSES} = batch {BATCH_SIZE} | "
          f"{len(smp)} batches/epoch (max-class, Fear cycles ~5x)")

    va_loader = DataLoader(va_ds, batch_size=16, num_workers=0)
    tr_frozen = DataLoader(tr_ds, batch_size=BATCH_FROZEN,
                           shuffle=True, drop_last=True, num_workers=0)

    model     = WavLMEliteSER(num_labels=len(lid)).to(device)
    swa_model = AveragedModel(model)
    crit      = FocalLoss(gamma=FOCAL_GAMMA)
    scaler    = GradScaler("cuda")
    best_acc  = 0.0; swa_active = False

    # ═══════════════════════════════════════════════════
    # PHASE 1 — Frozen WavLM, train head only
    # ═══════════════════════════════════════════════════
    model.freeze_backbone()
    p1 = [p for p in model.parameters() if p.requires_grad]
    opt1 = torch.optim.AdamW(p1, lr=1e-3, weight_decay=0.01)
    sch1 = get_cosine_schedule_with_warmup(opt1, 0, len(tr_frozen) * PHASE1_EPOCHS)

    print(f"\n{'='*60}")
    print(f"  PHASE 1 — Head Warmup ({PHASE1_EPOCHS} epochs, WavLM frozen)")
    print(f"  Params: {sum(p.numel() for p in p1):,} | Batch: {BATCH_FROZEN} | Loss: Focal")
    print(f"{'='*60}")

    for ep in range(1, PHASE1_EPOCHS + 1):
        model.train(); ep_loss = 0.0
        for b in tqdm(tr_frozen, desc=f"Ph1 Ep{ep:02d}", leave=False):
            with autocast("cuda"):
                loss = crit(model(b["wav"].to(device), b["prosody"].to(device)),
                            b["label"].to(device))
            opt1.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt1); nn.utils.clip_grad_norm_(p1, 1.0)
            scaler.step(opt1); scaler.update(); sch1.step()
            ep_loss += loss.item()
        acc, f1 = fast_eval(model, va_loader, device)
        tag = ""
        if acc > best_acc:
            best_acc = acc; torch.save(model.state_dict(), save_path); tag = "  *** BEST ***"
        print(f"Ph1 Ep {ep:02d} | Loss {ep_loss/len(tr_frozen):.3f} | "
              f"Val Acc {acc:.3f} | F1 {f1:.3f}{tag}")

    # ═══════════════════════════════════════════════════
    # PHASE 2 — LoRA fine-tune, FOCAL ONLY (no SupCon)
    # ═══════════════════════════════════════════════════
    model.unfreeze_lora()
    # Phase 2: natural distribution (no balanced sampler)
    # Focal Loss handles imbalance. Balanced sampler here causes distribution
    # shift that collapses accuracy (model calibrates for uniform, val is not).
    tr_phase2 = DataLoader(tr_ds, batch_size=16, shuffle=True,
                           drop_last=True, num_workers=0)
    opt2 = torch.optim.AdamW([
        {"params": [p for n, p in model.wavlm.named_parameters()
                    if "lora_" not in n and p.requires_grad], "lr": 5e-7},
        {"params": [p for n, p in model.wavlm.named_parameters()
                    if "lora_" in n],                         "lr": 5e-6},
        {"params": model.layer_weights,                       "lr": 1e-3},
        {"params": model.attn_pool.parameters(),              "lr": 2e-5},
        {"params": model.classifier.parameters(),             "lr": 2e-5},
    ], weight_decay=0.01)
    smp  = make_sampler()  # still used for SWA BN pass
    sch2 = get_cosine_schedule_with_warmup(opt2, len(tr_phase2), len(tr_phase2) * PHASE2_EPOCHS)

    print(f"\n{'='*60}")
    print(f"  PHASE 2 — LoRA Fine-tune ({PHASE2_EPOCHS} epochs, Focal only)")
    print(f"  Batch: {BATCH_SIZE} balanced | 74 batches/epoch | SWA from Ep{SWA_START}")
    print(f"{'='*60}")

    for ep in range(1, PHASE2_EPOCHS + 1):
        model.train(); ep_loss = 0.0
        for b in tqdm(tr_phase2, desc=f"Ph2 Ep{ep:02d}", leave=False):
            with autocast("cuda"):
                loss = crit(model(b["wav"].to(device), b["prosody"].to(device)),
                            b["label"].to(device))
            opt2.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt2); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt2); scaler.update(); sch2.step()
            ep_loss += loss.item()

        if ep >= SWA_START:
            swa_model.update_parameters(model); swa_active = True

        acc, f1 = fast_eval(model, va_loader, device)
        tag = ""
        if acc > best_acc:
            best_acc = acc; torch.save(model.state_dict(), save_path); tag = "  *** BEST ***"
        print(f"Ph2 Ep {ep:02d} | Loss {ep_loss/len(tr_phase2):.3f} | "
              f"Val Acc {acc:.3f} | F1 {f1:.3f}{tag}")

    # ═══════════════════════════════════════════════════
    # SWA
    # ═══════════════════════════════════════════════════
    if swa_active:
        print("\n[SWA] Finalizing...")
        swa_model.train()
        smp = make_sampler()
        tr_loader = DataLoader(tr_ds, batch_sampler=smp, num_workers=0)
        with torch.no_grad():
            for b in tqdm(tr_loader, desc="SWA BN", leave=False):
                with autocast("cuda"):
                    swa_model(b["wav"].to(device), b["prosody"].to(device))
        swa_acc, swa_f1 = fast_eval(swa_model, va_loader, device)
        print(f"[SWA] Val Acc {swa_acc:.3f} | F1 {swa_f1:.3f}")
        if swa_acc > best_acc:
            best_acc = swa_acc
            torch.save(swa_model.state_dict(), save_path)
            print("[SWA] SWA is new best.")

    # ═══════════════════════════════════════════════════
    # TTA EVALUATION (3-way: original + 0.9x + 1.1x speed)
    # ═══════════════════════════════════════════════════
    print("\n[TTA] 3-way test-time augmentation evaluation...")
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
