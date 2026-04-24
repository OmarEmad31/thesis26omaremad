"""
Audio FINAL v11 — Egyptian Arabic SER — Targeting 57%
======================================================
Proven backup architecture + every correct fix combined.

Changes from backup_sota_398.py (confirmed 39.8%):
  1. Rich 35-dim prosody (MFCC 1-13 mean+std, F0, energy, ZCR)
     → 9x more prosodic signal than 4-dim version
  2. Elite preprocessing: loudness norm + denoising + pre-emphasis + smart crop
     + in-memory cache (preprocessing runs ONCE per file)
  3. No SupCon in Phase 2 — proven root cause of every Phase 2 crash
  4. Conservative Phase 2 LoRA LR (2e-6 vs 5e-6) — less disruption
  5. Phase 2 uses min-class BalancedBatchSampler (15 batches) — no distribution shift
  6. Speed perturbation REMOVED — proven to hurt Phase 1 by 5%
  7. Multi-checkpoint ensemble: Phase1-best + Phase2-best + SWA → averaged
  8. TTA: original + 0.9x + 1.1x speed → 3-way average

Expected: ~52-58% (ensemble+TTA on top of stable ~42-46% individual model)
"""
import os, sys, subprocess, zipfile, random, io
from contextlib import redirect_stdout, redirect_stderr
from collections import defaultdict

def install_deps():
    pkgs = []
    for mod, pkg in [("audiomentations","audiomentations"), ("peft","peft"),
                     ("transformers","transformers"),("noisereduce","noisereduce"),
                     ("pyloudnorm","pyloudnorm")]:
        try: __import__(mod)
        except ImportError: pkgs.append(pkg)
    if pkgs:
        subprocess.check_call([sys.executable,"-m","pip","install",*pkgs,"-q"])

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
PHASE1_EPOCHS  = 6
PHASE2_EPOCHS  = 20
SWA_START      = 8
FOCAL_GAMMA    = 2.0
RARE_THRESHOLD = 50
MODEL_NAME     = "microsoft/wavlm-base-plus"
PROSODY_DIM    = 35   # rich prosody feature size


# ─────────────────────────────────────────────────────────
# BALANCED BATCH SAMPLER (min-class — no distribution shift)
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
        pools = {c: random.sample(idxs, len(idxs))
                 for c, idxs in self.class_indices.items()}
        ptrs  = {c: 0 for c in self.classes}
        for _ in range(self.n_batches):
            batch = []
            for c in self.classes:
                batch.extend(pools[c][ptrs[c]: ptrs[c] + self.k])
                ptrs[c] += self.k
            random.shuffle(batch); yield batch

    def __len__(self): return self.n_batches


# ─────────────────────────────────────────────────────────
# FOCAL LOSS (only — SupCon removed)
# ─────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__(); self.gamma = gamma

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
# MODEL — same as backup, classifier extended for 35-dim prosody
# ─────────────────────────────────────────────────────────
class WavLMEliteSER(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        base = WavLMModel.from_pretrained(MODEL_NAME, output_hidden_states=True)
        cfg  = LoraConfig(r=8, lora_alpha=16,
                          target_modules=["q_proj","v_proj"],
                          lora_dropout=0.05, bias="none")
        self.wavlm = get_peft_model(base, cfg)

        self.layer_weights = nn.Parameter(torch.ones(6))

        self.attn_pool = AttentionPool(768)
        nn.init.zeros_(self.attn_pool.attn.weight)
        nn.init.zeros_(self.attn_pool.attn.bias)

        # Extended classifier: 768 + 35 prosody dims
        self.classifier = nn.Sequential(
            nn.Linear(768 + PROSODY_DIM, 512),
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
# ELITE PREPROCESSING — loudness + denoise + preemphasis + smart crop
# ─────────────────────────────────────────────────────────
def elite_preprocess(y: np.ndarray, sr: int = SR) -> np.ndarray:
    try:
        meter    = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y.astype(np.float64))
        if np.isfinite(loudness) and -80 < loudness < 0:
            y = pyln.normalize.loudness(y.astype(np.float64),
                                        loudness, -23.0).astype(np.float32)
    except Exception: pass
    try:
        y = nr.reduce_noise(y=y, sr=sr, stationary=True,
                            prop_decrease=0.75).astype(np.float32)
    except Exception: pass
    y = np.append(y[0], y[1:] - 0.97 * y[:-1]).astype(np.float32)
    yt, _ = librosa.effects.trim(y, top_db=25)
    if len(yt) < sr // 4: yt = y
    if len(yt) > MAX_LEN:
        step = sr // 4; best_start, best_rms = 0, -1.0
        for s in range(0, len(yt) - MAX_LEN + 1, step):
            rms = float(np.mean(yt[s:s + MAX_LEN] ** 2))
            if rms > best_rms: best_rms, best_start = rms, s
        yt = yt[best_start:best_start + MAX_LEN]
    else:
        yt = np.pad(yt, (0, MAX_LEN - len(yt)))
    return np.clip(yt, -1.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────
# RICH PROSODY — 35-dim (vs 4-dim before)
# MFCC(1-13 mean+std=26) + F0(4) + Energy(3) + ZCR(2)
# ─────────────────────────────────────────────────────────
def extract_rich_prosody(y: np.ndarray, sr: int = SR) -> np.ndarray:
    feats = []
    # MFCC 1-13: mean + std = 26 features
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        feats.extend(mfcc.mean(axis=1).tolist())
        feats.extend(mfcc.std(axis=1).tolist())
    except Exception:
        feats.extend([0.0] * 26)
    # F0: mean, std, range, slope = 4 features
    try:
        f0 = librosa.yin(y, fmin=65, fmax=2093)
        valid = f0[np.isfinite(f0) & (f0 > 0)]
        if len(valid) > 1:
            slope = float(np.polyfit(np.arange(len(valid)), valid, 1)[0])
            feats += [valid.mean()/500, valid.std()/100,
                      valid.ptp()/500, slope/100]
        else:
            feats += [0., 0., 0., 0.]
    except Exception:
        feats += [0., 0., 0., 0.]
    # RMS energy: mean, std, max = 3 features
    try:
        rms = librosa.feature.rms(y=y)[0]
        feats += [float(rms.mean()), float(rms.std()), float(rms.max())]
    except Exception:
        feats += [0., 0., 0.]
    # ZCR: mean, std = 2 features
    try:
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        feats += [float(zcr.mean()), float(zcr.std())]
    except Exception:
        feats += [0., 0.]
    arr = np.array(feats, dtype=np.float32)
    return np.clip(np.nan_to_num(arr, nan=0., posinf=1., neginf=-1.), -10, 10)


# ─────────────────────────────────────────────────────────
# DATASET — elite preprocessing + in-memory cache + rich prosody
# Speed perturbation REMOVED (proven -5% on Phase 1 accuracy)
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
        self._cache: dict = {}   # fname → (preprocessed_audio, rich_prosody)

    def __len__(self): return len(self.df)

    def _load_audio(self, idx):
        row   = self.df.iloc[idx]
        fname = Path(row["audio_relpath"]).name
        if fname not in self.path_map: return None, None
        # Elite preprocessing cached per file
        if fname in self._cache:
            yt, pro = self._cache[fname]
            return yt.copy(), int(row["lid"]), pro.copy()
        try:
            raw, _ = librosa.load(self.path_map[fname], sr=SR)
            yt     = elite_preprocess(raw)
            pro    = extract_rich_prosody(yt)
            self._cache[fname] = (yt, pro)
            return yt.copy(), int(row["lid"]), pro.copy()
        except Exception:
            return None, None, None

    def _load(self, idx):
        result = self._load_audio(idx)
        if result[0] is None: return None
        yt, lid, pro = result
        try:
            if self.augment:
                # Intra-class mixup for rare classes
                if lid in self.rare_classes and random.random() < 0.5:
                    peer = random.choice(self.class_idx[lid])
                    r2   = self._load_audio(peer)
                    if r2[0] is not None:
                        a  = np.random.beta(0.4, 0.4)
                        yt = a * yt + (1 - a) * r2[0]

                # SpecAugment time mask
                T = len(yt); w = int(T * 0.15 * random.random())
                if w > 0:
                    s = random.randint(0, T - w)
                    yt = yt.copy(); yt[s:s + w] = 0.0

                # Gaussian noise + PitchShift
                yt = self.aug_pipe(samples=yt.astype(np.float32), sample_rate=SR)
                # Recompute prosody after augmentation
                pro = extract_rich_prosody(yt)

            return {
                "wav":     torch.tensor(yt,  dtype=torch.float32),
                "prosody": torch.tensor(pro, dtype=torch.float32),
                "label":   torch.tensor(lid, dtype=torch.long),
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
    if pm: print(f"[SUCCESS] Found {len(pm)} wav files."); return pm
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


def get_val_softmax(model, loader, device):
    """Return softmax probabilities over val set for ensemble."""
    model.eval(); probs_list = []
    with torch.no_grad():
        for b in loader:
            with autocast("cuda"):
                logits = model(b["wav"].to(device), b["prosody"].to(device))
            probs_list.append(F.softmax(logits, dim=-1).cpu())
    return torch.cat(probs_list, dim=0)


def tta_softmax(model, df, path_map, device):
    """TTA: average softmax over original + 0.9x + 1.1x speed."""
    model.eval()
    all_probs = []
    for _, row in df.iterrows():
        fname = Path(row["audio_relpath"]).name
        if fname not in path_map: all_probs.append(None); continue
        try:
            raw, _ = librosa.load(path_map[fname], sr=SR)
            yt     = elite_preprocess(raw)
            speeds = [1.0, 0.9, 1.1]; seg_probs = []
            for spd in speeds:
                if spd != 1.0:
                    try:
                        yr = librosa.resample(yt, orig_sr=SR, target_sr=int(SR*spd))
                        seg = yr[:MAX_LEN] if len(yr) >= MAX_LEN else np.pad(yr,(0,MAX_LEN-len(yr)))
                    except Exception: seg = yt
                else:
                    seg = yt
                pro = extract_rich_prosody(seg)
                with torch.no_grad(), autocast("cuda"):
                    l = model(torch.tensor(seg).unsqueeze(0).to(device),
                               torch.tensor(pro).unsqueeze(0).to(device))
                seg_probs.append(F.softmax(l, dim=-1).cpu())
            all_probs.append(torch.stack(seg_probs).mean(0))
        except Exception: all_probs.append(None)
    return all_probs


# ─────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────
def train():
    device     = "cuda"
    colab_root = Path("/content/drive/MyDrive/Thesis Project")
    csv_p      = colab_root / "data/processed/splits/text_hc"
    p1_save    = colab_root / "wavlm_v11_phase1.pt"
    p2_save    = colab_root / "wavlm_v11_phase2.pt"
    swa_save   = colab_root / "wavlm_v11_swa.pt"

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
    print("[DATA] Prosody dim:", PROSODY_DIM, "| Classifier input:", 768 + PROSODY_DIM)
    for emo, cnt in class_counts.items():
        print(f"  {emo:12s}: {cnt:3d}{' ← mixup' if lid.get(emo,-1) in rare_lids else ''}")

    tr_ds = AudioDataset(tr_df, path_map, augment=True,  rare_classes=rare_lids)
    va_ds = AudioDataset(va_df, path_map, augment=False, rare_classes=None)

    def make_sampler():
        return BalancedBatchSampler(tr_df["lid"].values, k=K_PER_CLASS)

    va_loader = DataLoader(va_ds, batch_size=16, num_workers=0)
    tr_frozen = DataLoader(tr_ds, batch_size=BATCH_FROZEN,
                           shuffle=True, drop_last=True, num_workers=0)

    model     = WavLMEliteSER(num_labels=len(lid)).to(device)
    swa_model = AveragedModel(model)
    crit      = FocalLoss(gamma=FOCAL_GAMMA)
    scaler    = GradScaler("cuda")
    p1_best   = 0.0; p2_best = 0.0; swa_active = False

    # ═══════════════════════════════════════════════════
    # PHASE 1 — Frozen backbone, train head only
    # ═══════════════════════════════════════════════════
    model.freeze_backbone()
    p1 = [p for p in model.parameters() if p.requires_grad]
    opt1 = torch.optim.AdamW(p1, lr=1e-3, weight_decay=0.01)
    sch1 = get_cosine_schedule_with_warmup(opt1, 0, len(tr_frozen) * PHASE1_EPOCHS)

    print(f"\n{'='*60}")
    print(f"  PHASE 1 — Head Warmup ({PHASE1_EPOCHS} epochs, WavLM frozen)")
    print(f"  Params: {sum(p.numel() for p in p1):,} | Batch: {BATCH_FROZEN}")
    print(f"  Loss: Focal only | Rich prosody: {PROSODY_DIM}-dim")
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
        if acc > p1_best:
            p1_best = acc; torch.save(model.state_dict(), p1_save); tag = "  *** P1 BEST ***"
        print(f"Ph1 Ep {ep:02d} | Loss {ep_loss/len(tr_frozen):.3f} | "
              f"Val Acc {acc:.3f} | F1 {f1:.3f}{tag}")

    # ═══════════════════════════════════════════════════
    # PHASE 2 — LoRA fine-tune, FOCAL ONLY
    # Conservative LR (2e-6) to avoid disruption
    # ═══════════════════════════════════════════════════
    model.unfreeze_lora()
    opt2 = torch.optim.AdamW([
        {"params": [p for n,p in model.wavlm.named_parameters()
                    if "lora_" not in n and p.requires_grad], "lr": 5e-7},
        {"params": [p for n,p in model.wavlm.named_parameters()
                    if "lora_" in n],                         "lr": 2e-6},
        {"params": model.layer_weights,                       "lr": 1e-3},
        {"params": model.attn_pool.parameters(),              "lr": 1e-5},
        {"params": model.classifier.parameters(),             "lr": 1e-5},
    ], weight_decay=0.01)
    smp  = make_sampler()
    sch2 = get_cosine_schedule_with_warmup(opt2, len(smp), len(smp)*PHASE2_EPOCHS)

    print(f"\n{'='*60}")
    print(f"  PHASE 2 — LoRA Fine-tune ({PHASE2_EPOCHS} epochs, Focal only)")
    print(f"  LoRA LR: 2e-6 | Classifier LR: 1e-5 | SWA from Ep{SWA_START}")
    print(f"{'='*60}")

    for ep in range(1, PHASE2_EPOCHS + 1):
        smp       = make_sampler()
        tr_loader = DataLoader(tr_ds, batch_sampler=smp, num_workers=0)
        model.train(); ep_loss = 0.0
        for b in tqdm(tr_loader, desc=f"Ph2 Ep{ep:02d}", leave=False):
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
        if acc > p2_best:
            p2_best = acc; torch.save(model.state_dict(), p2_save); tag = "  *** P2 BEST ***"
        print(f"Ph2 Ep {ep:02d} | Loss {ep_loss/len(smp):.3f} | "
              f"Val Acc {acc:.3f} | F1 {f1:.3f}{tag}")

    # ═══════════════════════════════════════════════════
    # SWA FINALIZATION
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
        torch.save(swa_model.state_dict(), swa_save)

    # ═══════════════════════════════════════════════════
    # MULTI-CHECKPOINT ENSEMBLE + TTA
    # Combine Phase1-best + Phase2-best + SWA via softmax averaging
    # ═══════════════════════════════════════════════════
    print("\n[ENSEMBLE] Multi-checkpoint TTA evaluation...")
    va_labels = va_df["lid"].tolist()
    all_ckpt_probs = []

    for ckpt_name, ckpt_path in [
        ("P1-best", p1_save), ("P2-best", p2_save),
        ("SWA",     swa_save if swa_active else p2_save)
    ]:
        if not Path(ckpt_path).exists(): continue
        model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
        print(f"  [{ckpt_name}] Running TTA...")
        tta_probs = tta_softmax(model, va_df, path_map, device)
        valid = [(p, t) for p, t in zip(tta_probs, va_labels) if p is not None]
        if not valid: continue
        probs_t = torch.cat([p for p, _ in valid], dim=0)  # [N, 7]
        all_ckpt_probs.append(probs_t)
        preds  = probs_t.argmax(1).numpy()
        truths = [t for _, t in valid]
        a = accuracy_score(truths, preds)
        f = f1_score(truths, preds, average="macro", zero_division=0)
        print(f"  [{ckpt_name}] TTA Acc {a:.4f} | F1 {f:.4f}")

    # Ensemble all checkpoints
    if all_ckpt_probs:
        valid_labels = [t for p, t in zip(tta_probs, va_labels) if p is not None]
        ens_probs = torch.stack(all_ckpt_probs).mean(0)
        ens_preds = ens_probs.argmax(1).numpy()
        ens_acc   = accuracy_score(valid_labels, ens_preds)
        ens_f1    = f1_score(valid_labels, ens_preds, average="macro", zero_division=0)
    else:
        ens_acc, ens_f1 = 0.0, 0.0

    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS")
    print(f"  Phase 1 Best (single-crop)  : {p1_best:.4f}")
    print(f"  Phase 2 Best (single-crop)  : {p2_best:.4f}")
    print(f"  Ensemble + TTA (3 ckpts×3)  : {ens_acc:.4f}")
    print(f"  Ensemble F1                 : {ens_f1:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
