"""
Audio v10 — emotion2vec Feature Extractor + MLP Classifier
===========================================================
Strategy: precompute ALL audio embeddings with emotion2vec_base_finetuned
           (specifically trained for SER). Train only the MLP head.

Why this beats WavLM fine-tuning for our small dataset:
  - emotion2vec was pre-trained with emotion-specific self-supervised objectives
    → features already cluster by emotion BEFORE any fine-tuning
  - No backbone fine-tuning = no LoRA instability, no Phase 2 crashes
  - Precomputed embeddings = huge batches (256+), many epochs (150), very fast
  - Feature-space MixUp + embedding noise → effective augmentation without audio ops

Expected: 45-58% (vs WavLM ceiling ~40%)
"""
import os, sys, subprocess, zipfile, random, io
from contextlib import redirect_stdout, redirect_stderr
from collections import defaultdict
from pathlib import Path

# ─────────────────────────────────────────────────────────
# DEPS
# ─────────────────────────────────────────────────────────
def install_deps():
    pkgs = []
    for mod, pkg in [("funasr", "funasr"), ("modelscope", "modelscope"),
                     ("audiomentations", "audiomentations"),
                     ("noisereduce", "noisereduce"), ("pyloudnorm", "pyloudnorm")]:
        try:   __import__(mod)
        except ImportError: pkgs.append(pkg)
    if pkgs:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *pkgs, "-q"])

if os.path.exists("/content"):
    install_deps()

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import pandas as pd, numpy as np, librosa
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import noisereduce as nr
import pyloudnorm as pyln
import logging
# Silence FunASR / ModelScope verbose INFO output
for _log in ["funasr", "modelscope", "modelscope.utils",
             "modelscope.hub", "urllib3", "asyncio"]:
    logging.getLogger(_log).setLevel(logging.ERROR)

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
SR             = 16000
MAX_LEN        = 80000        # 5 seconds
NUM_EPOCHS     = 150
BATCH_SIZE     = 64
LR             = 3e-4
FOCAL_GAMMA    = 2.0
RARE_THRESHOLD = 50
K_PER_CLASS    = 2
E2V_MODEL      = "iic/emotion2vec_base_finetuned"


# ─────────────────────────────────────────────────────────
# ELITE PREPROCESSING (same as v9c)
# ─────────────────────────────────────────────────────────
def elite_preprocess(y: np.ndarray, sr: int = SR) -> np.ndarray:
    try:
        meter    = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y.astype(np.float64))
        if np.isfinite(loudness) and -80 < loudness < 0:
            y = pyln.normalize.loudness(
                y.astype(np.float64), loudness, -23.0).astype(np.float32)
    except Exception: pass
    try:
        y = nr.reduce_noise(y=y, sr=sr, stationary=True,
                            prop_decrease=0.75).astype(np.float32)
    except Exception: pass
    y = np.append(y[0], y[1:] - 0.97 * y[:-1]).astype(np.float32)
    yt, _ = librosa.effects.trim(y, top_db=25)
    if len(yt) < sr // 4: yt = y
    if len(yt) > MAX_LEN:
        best_start, best_rms = 0, -1.0
        for s in range(0, len(yt) - MAX_LEN + 1, sr // 4):
            rms = float(np.mean(yt[s:s + MAX_LEN] ** 2))
            if rms > best_rms: best_rms, best_start = rms, s
        yt = yt[best_start:best_start + MAX_LEN]
    else:
        yt = np.pad(yt, (0, MAX_LEN - len(yt)))
    return np.clip(yt, -1.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────
# PROSODY (4-dim)
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
# emotion2vec EMBEDDING EXTRACTION
# ─────────────────────────────────────────────────────────
def load_emotion2vec(cache_dir: str):
    from funasr import AutoModel
    print(f"[E2V] Loading {E2V_MODEL} ...")
    m = AutoModel(model=E2V_MODEL, cache_dir=cache_dir, disable_update=True)
    print("[E2V] Model loaded.")
    return m


def extract_e2v_embedding(e2v_model, audio_path: str) -> np.ndarray | None:
    """Extract utterance-level emotion embedding from emotion2vec (silent)."""
    try:
        # Redirect stdout+stderr to suppress FunASR's internal tqdm/RTF output
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            logging.disable(logging.INFO)
            res = e2v_model.generate(str(audio_path), output_dir=None,
                                     granularity="utterance", extract_embedding=True)
            logging.disable(logging.NOTSET)
        if not res: return None
        r = res[0]
        for key in ["feats", "embedding", "hidden_states", "features"]:
            if key in r:
                e = np.array(r[key], dtype=np.float32)
                return e.mean(0) if e.ndim == 2 else e
    except Exception as ex:
        logging.disable(logging.NOTSET)
        print(f"  [WARN] {Path(audio_path).name}: {ex}")
    return None


def precompute_embeddings(e2v_model, df: pd.DataFrame, path_map: dict,
                          cache_path: Path) -> dict:
    """
    Run emotion2vec on all files and cache results to Drive.
    Returns dict: fname -> {'emb': [D], 'prosody': [4]}
    """
    import pickle
    if cache_path.exists():
        print(f"[CACHE] Loading precomputed embeddings from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"[E2V] Precomputing embeddings for {len(df)} samples...")
    cache = {}
    for _, row in tqdm(df.iterrows(), total=len(df)):
        fname = Path(row["audio_relpath"]).name
        if fname not in path_map: continue
        try:
            raw, _ = librosa.load(path_map[fname], sr=SR)
            yt     = elite_preprocess(raw)
            emb    = extract_e2v_embedding(e2v_model, path_map[fname])
            if emb is None: continue
            pro    = extract_prosody(yt)
            cache[fname] = {"emb": emb, "prosody": pro, "lid": int(row["lid"])}
        except Exception as ex:
            print(f"  [SKIP] {fname}: {ex}")

    print(f"[E2V] Precomputed {len(cache)}/{len(df)} files. Saving cache...")
    with open(cache_path, "wb") as f:
        import pickle; pickle.dump(cache, f)
    return cache


# ─────────────────────────────────────────────────────────
# BALANCED BATCH SAMPLER (max-class based, used in Phase 1)
# ─────────────────────────────────────────────────────────
class BalancedBatchSampler(Sampler):
    def __init__(self, labels, k=K_PER_CLASS):
        self.k = k
        self.class_indices = defaultdict(list)
        for i, lbl in enumerate(labels):
            self.class_indices[int(lbl)].append(i)
        self.classes   = sorted(self.class_indices.keys())
        max_count      = max(len(v) for v in self.class_indices.values())
        self.n_batches = max_count // k

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
            random.shuffle(batch); yield batch

    def __len__(self): return self.n_batches


# ─────────────────────────────────────────────────────────
# FOCAL LOSS
# ─────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__(); self.gamma = gamma

    def forward(self, logits, labels):
        ce  = F.cross_entropy(logits, labels, reduction="none")
        p_t = torch.exp(-ce)
        return (((1 - p_t) ** self.gamma) * ce).mean()


# ─────────────────────────────────────────────────────────
# EMBEDDING DATASET with feature-space augmentation
# ─────────────────────────────────────────────────────────
class EmbDataset(Dataset):
    def __init__(self, cache: dict, lids: list, augment=False, rare_lids=None):
        self.items     = [(fname, cache[fname]) for fname in cache]
        # Filter to only the subset we need (by lid match from dataframe)
        self.augment   = augment
        self.rare_lids = set(rare_lids or [])
        self.class_idx = defaultdict(list)
        for i, (_, d) in enumerate(self.items):
            self.class_idx[d["lid"]].append(i)

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        fname, d = self.items[idx]
        emb = d["emb"].copy().astype(np.float32)
        pro = d["prosody"].copy().astype(np.float32)
        lid = d["lid"]

        if self.augment:
            # Intra-class mixup for rare classes (feature space)
            if lid in self.rare_lids and random.random() < 0.5:
                peer_i = random.choice(self.class_idx[lid])
                peer_emb = self.items[peer_i][1]["emb"]
                a   = np.random.beta(0.4, 0.4)
                emb = a * emb + (1 - a) * peer_emb.astype(np.float32)

            # Embedding noise (like dropout in feature space)
            if random.random() < 0.5:
                emb = emb + np.random.randn(*emb.shape).astype(np.float32) * 0.02

            # Random feature masking (like SpecAugment for embeddings)
            if random.random() < 0.3:
                mask_n = int(len(emb) * 0.15)
                mask_i = np.random.choice(len(emb), mask_n, replace=False)
                emb[mask_i] = 0.0

        return {
            "emb":     torch.tensor(emb, dtype=torch.float32),
            "prosody": torch.tensor(pro, dtype=torch.float32),
            "label":   torch.tensor(lid, dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────
# MLP CLASSIFIER with residual connections
# ─────────────────────────────────────────────────────────
class EmotionMLP(nn.Module):
    def __init__(self, emb_dim: int, num_classes: int):
        super().__init__()
        inp = emb_dim + 4  # embedding + prosody

        self.stem = nn.Sequential(
            nn.Linear(inp, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.4)
        )
        # Residual block
        self.res1 = nn.Sequential(
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 512), nn.BatchNorm1d(512)
        )
        self.mid = nn.Sequential(
            nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2)
        )
        self.head = nn.Linear(256, num_classes)

    def forward(self, emb, prosody):
        x = self.stem(torch.cat([emb, prosody], dim=-1))
        x = x + self.res1(x)          # residual
        x = self.mid(x)
        return self.head(x)


# ─────────────────────────────────────────────────────────
# PATH MAP
# ─────────────────────────────────────────────────────────
def get_path_map(colab_root: Path) -> dict:
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
# EVAL
# ─────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval(); ps, ts = [], []
    with torch.no_grad():
        for b in loader:
            logits = model(b["emb"].to(device), b["prosody"].to(device))
            ps.extend(logits.argmax(1).cpu().numpy())
            ts.extend(b["label"].numpy())
    return accuracy_score(ts, ps), f1_score(ts, ps, average="macro", zero_division=0)


# ─────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────
def train():
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    colab_root = Path("/content/drive/MyDrive/Thesis Project")
    csv_p      = colab_root / "data/processed/splits/text_hc"
    e2v_cache  = Path("/content/drive/MyDrive")
    tr_cache   = e2v_cache / "e2v_train_cache.pkl"
    va_cache   = e2v_cache / "e2v_val_cache.pkl"
    save_path  = colab_root / "emotion2vec_mlp_best.pt"

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
    print("[DATA] Distribution:")
    for emo, cnt in class_counts.items():
        tag = " ← rare+mixup" if lid.get(emo, -1) in rare_lids else ""
        print(f"  {emo:12s}: {cnt:3d} ({100*cnt/len(tr_df):.1f}%){tag}")

    # Load emotion2vec and precompute embeddings
    e2v = load_emotion2vec(str(e2v_cache / "emotion2vec_model"))
    tr_cache_data = precompute_embeddings(e2v, tr_df, path_map, tr_cache)
    va_cache_data = precompute_embeddings(e2v, va_df, path_map, va_cache)

    # Filter to matching lids
    tr_lids = set(tr_df["lid"].unique())
    va_lids = set(va_df["lid"].unique())

    tr_ds = EmbDataset(tr_cache_data, tr_lids, augment=True,  rare_lids=rare_lids)
    va_ds = EmbDataset(va_cache_data, va_lids, augment=False, rare_lids=None)

    # Detect embedding dim from first sample
    emb_dim = tr_ds[0]["emb"].shape[0]
    print(f"\n[E2V] Embedding dim: {emb_dim}")

    # Balanced sampler for training (cycling all classes)
    tr_labels = [tr_cache_data[fname]["lid"] for fname, _ in tr_ds.items]
    bal_smp   = BalancedBatchSampler(tr_labels, k=K_PER_CLASS)
    tr_loader = DataLoader(tr_ds, batch_sampler=bal_smp, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=64, num_workers=0)

    print(f"[SAMPLER] {K_PER_CLASS}/class × {len(lid)} = batch {K_PER_CLASS*len(lid)} | "
          f"{len(bal_smp)} batches/epoch")

    model    = EmotionMLP(emb_dim=emb_dim, num_classes=len(lid)).to(device)
    crit     = FocalLoss(gamma=FOCAL_GAMMA)
    opt      = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sch      = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS)
    best_acc = 0.0

    print(f"\n{'='*60}")
    print(f"  emotion2vec MLP | {NUM_EPOCHS} epochs | Focal Loss | Batch {BATCH_SIZE}")
    print(f"  Augmentation: mixup + embedding noise + feature masking")
    print(f"{'='*60}")

    for ep in range(1, NUM_EPOCHS + 1):
        bal_smp   = BalancedBatchSampler(tr_labels, k=K_PER_CLASS)
        tr_loader = DataLoader(tr_ds, batch_sampler=bal_smp, num_workers=0)
        model.train(); ep_loss = 0.0

        for b in tqdm(tr_loader, desc=f"Ep {ep:03d}", leave=False):
            logits = model(b["emb"].to(device), b["prosody"].to(device))
            loss   = crit(logits, b["label"].to(device))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()

        sch.step()
        acc, f1 = evaluate(model, va_loader, device)
        tag = ""
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path); tag = "  *** BEST ***"

        if ep % 5 == 0 or ep <= 5 or tag:
            print(f"Ep {ep:03d} | Loss {ep_loss/len(bal_smp):.3f} | "
                  f"Val Acc {acc:.3f} | F1 {f1:.3f}{tag}")

    # Final best model evaluation
    model.load_state_dict(torch.load(save_path, map_location=device))
    final_acc, final_f1 = evaluate(model, va_loader, device)

    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS")
    print(f"  Best Val Acc : {best_acc:.4f}")
    print(f"  Final F1     : {final_f1:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
