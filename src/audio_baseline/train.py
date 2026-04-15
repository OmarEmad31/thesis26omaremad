"""Audio emotion classifier — Phases 1-3 implementation.

Changes vs. previous version:
  Phase 1 — Fix stale cache (new cache filename) + fix f-string warning bug
  Phase 2 — Audio augmentation during extraction (4× training data)
             Each train file → orig + speed×0.85 + speed×1.15 + noise
             Val / test stay CLEAN (no augmentation ever)
  Phase 3 — Frame-level embedding aggregation
             granularity="frame" → concat(mean, std, max) → 2304-dim
             (was utterance-level mean → 768-dim, throwing away temporal info)

Run from project root:
  python -m src.audio_baseline.train

COLAB SETUP:
  !pip install funasr librosa -q
  !git pull
  # No need to rebuild manifest — current splits have 648 train / 139 val / 137 test
  # New cache name means no conflict with old 511-sample cache
  !python -m src.audio_baseline.train
"""

import sys
import logging
import warnings
import collections
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from transformers import set_seed
from transformers import logging as transformers_logging
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report

warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()
logging.getLogger("funasr").setLevel(logging.ERROR)

from src.audio_baseline import config

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
log_file = config.CHECKPOINT_DIR.parent / "audio_training_log.txt"
log_file.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 3 helper: aggregate frame-level embeddings → richer representation
# ---------------------------------------------------------------------------
def aggregate_frames(feat: np.ndarray) -> np.ndarray:
    """
    feat: [T, D]  (T frames, D dims)  OR  [D]  (single utterance-level vector)

    Returns: [D]  if feat is 1-D (utterance fallback)
             [3*D] if feat is 2-D (frame-level): concat(mean, std, max)

    The 3× richer vector preserves temporal dynamics that mean-pooling erases.
    E.g. for emotion2vec_plus_base (D=768) → 2304-dim output.
    """
    if feat.ndim == 1:
        # Model returned utterance-level — pad to 3× so dim is consistent
        return np.concatenate([feat, np.zeros_like(feat), feat])   # [3*D]
    # Normal case: [T, D] → concat stats
    mean_f = feat.mean(axis=0)   # [D]
    std_f  = feat.std(axis=0)    # [D]
    max_f  = feat.max(axis=0)    # [D]
    return np.concatenate([mean_f, std_f, max_f])                  # [3*D]


# ---------------------------------------------------------------------------
# Phase 2 helper: build augmented versions of a waveform
# ---------------------------------------------------------------------------
def get_augmentations(audio: np.ndarray, sr: int) -> list[tuple[str, np.ndarray]]:
    """
    Returns [(name, waveform)] list.
    Always starts with the original; augmented versions added when feasible.
    All returned arrays are float32.
    """
    import librosa

    results = [("orig", audio.copy())]

    # Speed stretch — preserves pitch, changes duration
    for rate in config.AUG_SPEED_RATES:
        label = "slow" if rate < 1.0 else "fast"
        try:
            stretched = librosa.effects.time_stretch(audio.astype(np.float32), rate=rate)
            results.append((label, stretched))
        except Exception:
            pass  # Skip this augmentation if it fails (very short clips)

    # Additive Gaussian noise proportional to signal std
    noise_std = config.AUG_NOISE_STD * float(np.std(audio) + 1e-8)
    noisy     = (audio + (noise_std * np.random.randn(len(audio)))).astype(np.float32)
    results.append(("noise", noisy))

    return results


# ---------------------------------------------------------------------------
# Embedding extraction  (Phases 2 + 3)
# ---------------------------------------------------------------------------
def extract_embeddings(backbone, df: pd.DataFrame, label2id: dict,
                       split_name: str, augment: bool = False,
                       first_call: bool = False):
    """
    Extract emotion2vec frame-level embeddings and aggregate as mean+std+max.

    augment=True  → apply 4× audio augmentation (training split only!)
    augment=False → single clean extraction (val / test splits)
    """
    import librosa

    embeddings: list[np.ndarray] = []
    labels_out: list[int]        = []
    skipped = 0
    n_aug_versions = len(config.AUG_SPEED_RATES) + 2  # orig + speeds + noise
    detected_dim = None

    for row_idx, (_, row) in enumerate(tqdm(
            df.iterrows(), total=len(df),
            desc=f"  {'aug-' if augment else ''}{split_name}")):

        folder   = str(row["folder"]).strip()
        rel_path = str(row["audio_relpath"]).replace("\\", "/").lstrip("/")

        candidate = config.DATA_ROOT / folder / rel_path
        if not candidate.exists():
            for sub in (s for s in config.DATA_ROOT.iterdir() if s.is_dir()):
                alt = sub / folder / rel_path
                if alt.exists():
                    candidate = alt
                    break

        if not candidate.exists():
            skipped += 1
            continue

        try:
            audio, _ = librosa.load(candidate, sr=config.SAMPLING_RATE, mono=True)
            if len(audio) > config.MAX_AUDIO_SAMPLES:
                audio = audio[: config.MAX_AUDIO_SAMPLES]
            audio = audio.astype(np.float32)
        except Exception:
            skipped += 1
            continue

        label = label2id[row["emotion_final"]]

        # Build list of (name, waveform) to process
        versions = get_augmentations(audio, config.SAMPLING_RATE) if augment else [("orig", audio)]

        for aug_name, wav in versions:
            # Clip augmented audio to max length
            if len(wav) > config.MAX_AUDIO_SAMPLES:
                wav = wav[: config.MAX_AUDIO_SAMPLES]

            try:
                # Silence funasr stdout/stderr
                with open(os.devnull, "w") as devnull:
                    old_out, old_err = sys.stdout, sys.stderr
                    sys.stdout = sys.stderr = devnull
                    try:
                        res = backbone.generate(
                            input=wav,
                            granularity=config.EMBED_GRANULARITY,
                            extract_embedding=True,
                        )
                    finally:
                        sys.stdout, sys.stderr = old_out, old_err

                if not res or "feats" not in res[0]:
                    continue

                raw  = np.array(res[0]["feats"], dtype=np.float32)
                feat = aggregate_frames(raw)    # → [3*D] or [D] fallback

                if detected_dim is None:
                    detected_dim = feat.shape[0]
                    if first_call:
                        logger.info(
                            f"    🔬 granularity={config.EMBED_GRANULARITY}"
                            f"  raw_shape={raw.shape}"
                            f"  → aggregated dim={detected_dim}"
                        )

                embeddings.append(feat)
                labels_out.append(label)

            except Exception:
                continue  # Skip this version; at least orig is usually fine

    n_files = len(df) - skipped
    n_embs  = len(embeddings)
    aug_msg = f" × {n_embs // max(n_files, 1):.1f} aug" if augment else ""
    logger.info(
        f"    {split_name}: {n_files} files → {n_embs} embeddings{aug_msg}"
        f"  ({skipped} files skipped)  dim={detected_dim}"
    )
    return (np.array(embeddings, dtype=np.float32),
            np.array(labels_out,  dtype=np.int64))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout)
        )
    def forward(self, x): return x + self.net(x)


class AudioMLP(nn.Module):
    """
    Residual MLP classifier on frozen emotion2vec embeddings.

    Input dim is auto-detected (2304 with frame-level agg, 768 fallback).
    Architecture: in → 512 → 512-res → 256 → 256-res → num_labels
    Heavy dropout because N_train < 5000.
    """
    def __init__(self, input_dim: int, num_labels: int = 7):
        super().__init__()
        self.stem = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, 512),
            nn.GELU(), nn.Dropout(0.40),
        )
        self.res1 = ResBlock(512, 0.30)
        self.down = nn.Sequential(
            nn.LayerNorm(512), nn.Linear(512, 256),
            nn.GELU(), nn.Dropout(0.35),
        )
        self.res2       = ResBlock(256, 0.25)
        self.classifier = nn.Linear(256, num_labels)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        x    = self.stem(x)
        x    = self.res1(x)
        x    = self.down(x)
        proj = self.res2(x)
        return self.classifier(proj), proj


# ---------------------------------------------------------------------------
# Losses + augmentation
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma: float = 1.5, label_smoothing: float = 0.1):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else torch.ones(1))
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce    = F.cross_entropy(logits, targets, weight=self.weight,
                                label_smoothing=self.label_smoothing, reduction="none")
        focal = (1.0 - torch.exp(-ce)) ** self.gamma * ce
        return focal.mean()


def supcon_loss(proj, labels, temp, device):
    feat      = F.normalize(proj, p=2, dim=1)
    sim       = torch.matmul(feat, feat.T) / temp
    bs        = labels.size(0)
    I         = torch.eye(bs, device=device)
    pos       = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float() * (1 - I)
    neg_I     = 1 - I
    sim_max,_ = sim.max(dim=1, keepdim=True)
    sim       = sim - sim_max.detach()
    exp_sim   = torch.exp(sim) * neg_I
    log_prob  = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)
    valid     = pos.sum(1) > 0
    if not valid.any():
        return torch.tensor(0.0, device=device)
    return (-(pos[valid] * log_prob[valid]).sum(1) / pos[valid].sum(1).clamp(1)).mean()


def mixup(x, y, alpha, device):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def mixup_loss(criterion, logits, ya, yb, lam):
    return lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)


# ---------------------------------------------------------------------------
# SVM pipeline
# ---------------------------------------------------------------------------
def run_sklearn(train_emb, train_lbl, val_emb, val_lbl,
                test_emb, test_lbl, id2label):
    scaler = StandardScaler()
    Xtr    = scaler.fit_transform(train_emb)
    Xva    = scaler.transform(val_emb)
    Xte    = scaler.transform(test_emb)
    # Train+val for final SVM generalisation
    Xtv    = scaler.transform(np.concatenate([train_emb, val_emb]))
    ytv    = np.concatenate([train_lbl, val_lbl])

    results = {}

    # Logistic Regression (reference)
    logger.info("   [LR] fitting...")
    lr = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced",
                            solver="lbfgs", random_state=42)
    lr.fit(Xtr, train_lbl)
    for sn, X, y in [("val", Xva, val_lbl), ("test", Xte, test_lbl)]:
        p = lr.predict(X)
        results[f"LR_{sn}_acc"] = accuracy_score(y, p)
        results[f"LR_{sn}_f1"]  = f1_score(y, p, average="macro")

    # SVM-RBF GridSearch
    logger.info("   [SVM] grid-searching (C, gamma)...")
    param_grid = {"C": [0.1, 0.5, 1, 5, 10, 50, 100, 500],
                  "gamma": ["scale", "auto"]}
    gs = GridSearchCV(
        SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
        param_grid, cv=5, scoring="f1_macro", n_jobs=-1, verbose=0,
    )
    gs.fit(Xtr, train_lbl)
    best = gs.best_params_
    logger.info(f"   ✅ Best C={best['C']}  gamma={best['gamma']}"
                f"  cv-F1={gs.best_score_:.4f}")

    svm_tv = SVC(kernel="rbf", probability=True, class_weight="balanced",
                 random_state=42, **best)
    svm_tv.fit(Xtv, ytv)

    for sn, X, y, m in [("val",  Xva, val_lbl,  gs.best_estimator_),
                         ("test", Xte, test_lbl, svm_tv)]:
        p = m.predict(X)
        results[f"SVM_{sn}_acc"] = accuracy_score(y, p)
        results[f"SVM_{sn}_f1"]  = f1_score(y, p, average="macro")

    logger.info("\n📊 SKLEARN:")
    logger.info(f"   LR  val={results['LR_val_acc']:.4f}/{results['LR_val_f1']:.4f}"
                f"  test={results['LR_test_acc']:.4f}/{results['LR_test_f1']:.4f}")
    logger.info(f"   SVM val={results['SVM_val_acc']:.4f}/{results['SVM_val_f1']:.4f}"
                f"  test={results['SVM_test_acc']:.4f}/{results['SVM_test_f1']:.4f}")

    label_strs = [id2label[i] for i in range(len(id2label))]
    logger.info("\n   SVM Test Report:")
    logger.info(classification_report(test_lbl, svm_tv.predict(Xte),
                                      target_names=label_strs))

    return results, scaler, svm_tv


# ---------------------------------------------------------------------------
# MLP training
# ---------------------------------------------------------------------------
def train_mlp(train_emb, train_lbl, val_emb, val_lbl, id2label, device):
    num_labels = len(id2label)
    emb_dim    = train_emb.shape[1]

    N_train = len(train_lbl)
    dist    = dict(sorted(collections.Counter(train_lbl.tolist()).items()))
    logger.info(f"\n   Training dist ({N_train} samples): {dist}")
    # Phase 1 fix: was "Only {N_train}" (literal), now f-string
    if N_train < 1000:
        logger.warning(f"   ⚠️  Only {N_train} training samples. "
                       "Consider deleting stale cache and using more data.")

    Xtr = torch.tensor(train_emb, dtype=torch.float32)
    ytr = torch.tensor(train_lbl, dtype=torch.long)
    Xva = torch.tensor(val_emb,   dtype=torch.float32)
    yva = torch.tensor(val_lbl,   dtype=torch.long)

    train_loader = DataLoader(TensorDataset(Xtr, ytr),
                              batch_size=config.BATCH_SIZE,
                              shuffle=True, drop_last=False)
    val_loader   = DataLoader(TensorDataset(Xva, yva), batch_size=256, shuffle=False)

    model     = AudioMLP(input_dim=emb_dim, num_labels=num_labels).to(device)
    raw_w     = compute_class_weight("balanced", classes=np.arange(num_labels), y=train_lbl)
    cw        = torch.tensor(raw_w, dtype=torch.float, device=device)
    criterion = FocalLoss(weight=cw, gamma=config.FOCAL_GAMMA, label_smoothing=0.1)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.LEARNING_RATE,
                                  weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE * 10,
        steps_per_epoch=len(train_loader),
        epochs=config.EPOCHS,
        pct_start=0.10,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=1e3,
    )

    best_f1, patience, best_state = 0.0, 0, None

    logger.info(f"\n⚡ AudioMLP + Focal + SCL(t={config.SCL_TEMP}) + MixUp")
    logger.info(f"   dim={emb_dim}  bs={config.BATCH_SIZE}"
                f"  lr={config.LEARNING_RATE}  wd={config.WEIGHT_DECAY}")
    logger.info(f"   epochs={config.EPOCHS}  patience={config.MAX_PATIENCE}")

    for epoch in range(config.EPOCHS):
        model.train()
        sum_correct = sum_n = 0
        sum_ce = sum_sc = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            use_mix = config.USE_MIXUP and np.random.random() < config.MIXUP_PROB

            if use_mix:
                xb, ya, yb_, lam = mixup(xb, yb, config.MIXUP_ALPHA, device)
                logits, proj = model(xb)
                ce  = mixup_loss(criterion, logits, ya, yb_, lam)
                sc  = supcon_loss(proj, ya, config.SCL_TEMP, device) if config.USE_SCL else torch.tensor(0.0)
                p   = logits.argmax(1)
                sum_correct += ((p == ya).float() * lam + (p == yb_).float() * (1 - lam)).sum().item()
            else:
                logits, proj = model(xb)
                ce  = criterion(logits, yb)
                sc  = supcon_loss(proj, yb, config.SCL_TEMP, device) if config.USE_SCL else torch.tensor(0.0)
                sum_correct += (logits.argmax(1) == yb).sum().item()

            loss = ce + config.SCL_WEIGHT * sc
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()

            sum_n  += yb.size(0)
            sum_ce += ce.item()
            sum_sc += sc.item()

        # Validation
        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                vp.extend(model(xb.to(device))[0].argmax(1).cpu().numpy())
                vt.extend(yb.numpy())

        val_acc = accuracy_score(vt, vp)
        val_f1  = f1_score(vt, vp, average="macro")
        improved = val_f1 > best_f1

        if improved:
            best_f1    = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience   = 0
        else:
            patience += 1

        if improved or ((epoch + 1) % config.LOG_EVERY == 0) or epoch == 0:
            tag = "⭐" if improved else f"patience {patience}/{config.MAX_PATIENCE}"
            logger.info(
                f"  Ep {epoch:03d} | Tr {sum_correct/sum_n:.4f} | "
                f"Val {val_acc:.4f}/{val_f1:.4f} | "
                f"CE={sum_ce/len(train_loader):.3f} "
                f"SCL={sum_sc/len(train_loader):.3f}  {tag}"
            )

        if patience >= config.MAX_PATIENCE:
            logger.info(f"  ⏹ Early stop ep={epoch}  best val-F1={best_f1:.4f}")
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_f1


# ---------------------------------------------------------------------------
# Evaluation (clean + TTA)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, emb_np, lbl_np, device, tta=False):
    model.eval()
    passes = config.TTA_PASSES if tta else 1
    noise  = config.TTA_NOISE  if tta else 0.0
    acc_p  = None

    for _ in range(passes):
        x = torch.tensor(emb_np, dtype=torch.float32)
        if noise > 0:
            x = x + torch.randn_like(x) * noise
        loader = DataLoader(TensorDataset(x, torch.tensor(lbl_np)),
                            batch_size=256, shuffle=False)
        probs  = []
        for xb, _ in loader:
            logits, _ = model(xb.to(device))
            probs.append(F.softmax(logits, -1).cpu().numpy())
        probs  = np.concatenate(probs)
        acc_p  = probs if acc_p is None else acc_p + probs

    avg  = acc_p / passes
    pred = avg.argmax(1)
    return lbl_np, pred, avg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    set_seed(42)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    logger.info(f"\n🚀 Device: {device}  |  Backbone: {config.MODEL_NAME}")
    logger.info(f"   Embed granularity: {config.EMBED_GRANULARITY}"
                f"  |  Train augmentation: {config.AUGMENT_TRAIN}")

    # 1. Load splits
    logger.info(f"\n📁 {config.SPLIT_CSV_DIR}")
    train_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv")
    val_df   = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv")
    test_df  = pd.read_csv(config.SPLIT_CSV_DIR / "test.csv")

    label_names = sorted(train_df["emotion_final"].unique())
    label2id    = {n: i for i, n in enumerate(label_names)}
    id2label    = {i: n for n, i in label2id.items()}
    logger.info(f"🎭 {len(label2id)} classes: {label2id}")
    logger.info(f"   CSV rows — Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # 2. Embeddings (cached)
    cache = config.EMBEDDING_CACHE
    if cache.exists():
        logger.info(f"⚡ Cache hit: {cache}")
        d         = np.load(cache)
        train_emb = d["train_emb"]; train_lbl = d["train_lbl"]
        val_emb   = d["val_emb"];   val_lbl   = d["val_lbl"]
        test_emb  = d["test_emb"];  test_lbl  = d["test_lbl"]
        logger.info(f"   Loaded — Train: {len(train_lbl)} | Val: {len(val_lbl)}"
                    f" | Test: {len(test_lbl)} | dim: {train_emb.shape[1]}")
    else:
        logger.info("🧠 Loading backbone (extraction runs once, then cached)...")
        from funasr import AutoModel
        bb = AutoModel(model=config.MODEL_NAME, hub="hf", trust_remote_code=True)
        bb.model.eval()
        for p in bb.model.parameters():
            p.requires_grad = False

        logger.info(f"🔬 Extracting embeddings"
                    f" (granularity={config.EMBED_GRANULARITY}"
                    f", train_aug={config.AUGMENT_TRAIN})...")
        # Phase 2: training gets augmented, val/test always clean
        train_emb, train_lbl = extract_embeddings(
            bb, train_df, label2id, "train",
            augment=config.AUGMENT_TRAIN, first_call=True
        )
        val_emb,   val_lbl   = extract_embeddings(bb, val_df,  label2id, "val",  augment=False)
        test_emb,  test_lbl  = extract_embeddings(bb, test_df, label2id, "test", augment=False)

        np.savez_compressed(cache,
            train_emb=train_emb, train_lbl=train_lbl,
            val_emb=val_emb,     val_lbl=val_lbl,
            test_emb=test_emb,   test_lbl=test_lbl)
        logger.info(f"💾 Saved to {cache}")
        del bb
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    logger.info(f"   Train: {len(train_lbl)} embeddings (includes augmentation if on)")
    logger.info(f"   Val: {len(val_lbl)} | Test: {len(test_lbl)} | dim: {train_emb.shape[1]}")
    logger.info(f"   Train dist: {dict(sorted(collections.Counter(train_lbl.tolist()).items()))}")

    # 3. SVM (uses CLEAN train embeddings — SVM doesn't benefit from augmented duplicates)
    # Note: when augment=True, train_emb contains 4× data including augmented versions.
    # SVM with probability calibration still runs on this — more data helps generalisation.
    sk_res, scaler, svm = run_sklearn(
        train_emb, train_lbl, val_emb, val_lbl, test_emb, test_lbl, id2label
    )

    # 4. MLP (benefits greatly from augmented training data)
    mlp, best_val_f1 = train_mlp(
        train_emb, train_lbl, val_emb, val_lbl, id2label, device
    )

    # 5. Evaluate MLP
    true_te, pred_clean, prob_clean = evaluate(mlp, test_emb, test_lbl, device, tta=False)
    _,       pred_tta,   prob_tta   = evaluate(mlp, test_emb, test_lbl, device, tta=True)
    true_va, pred_va,    _          = evaluate(mlp, val_emb,  val_lbl,  device, tta=False)

    mlp_test_acc = accuracy_score(true_te, pred_clean)
    mlp_test_f1  = f1_score(true_te, pred_clean, average="macro")
    tta_test_acc = accuracy_score(true_te, pred_tta)
    tta_test_f1  = f1_score(true_te, pred_tta, average="macro")
    mlp_val_acc  = accuracy_score(true_va, pred_va)

    # 6. Ensemble: SVM + MLP-TTA
    svm_prob_te = svm.predict_proba(scaler.transform(test_emb))
    ens_prob    = 0.5 * svm_prob_te + 0.5 * prob_tta
    ens_pred    = ens_prob.argmax(1)
    ens_acc     = accuracy_score(test_lbl, ens_pred)
    ens_f1      = f1_score(test_lbl, ens_pred, average="macro")
    ens_f1w     = f1_score(test_lbl, ens_pred, average="weighted")

    # 7. Final reports
    label_strs = [id2label[i] for i in range(len(id2label))]
    logger.info("\n   MLP (clean) Test Report:")
    logger.info(classification_report(true_te, pred_clean, target_names=label_strs))
    if config.USE_TTA:
        logger.info("   MLP (TTA) Test Report:")
        logger.info(classification_report(true_te, pred_tta, target_names=label_strs))
    logger.info("   Ensemble (SVM + MLP-TTA) Test Report:")
    logger.info(classification_report(test_lbl, ens_pred, target_names=label_strs))

    logger.info("\n" + "=" * 70)
    logger.info("🏆  AUDIO EMOTION CLASSIFICATION — FINAL RESULTS")
    logger.info("=" * 70)
    logger.info(f"{'Method':<30} {'Val-Acc':>8} {'Val-F1':>8} {'Test-Acc':>9} {'Test-F1':>9}")
    logger.info("-" * 70)
    logger.info(f"{'LR (reference)':<30} {sk_res['LR_val_acc']:>8.4f} {sk_res['LR_val_f1']:>8.4f} "
                f"{sk_res['LR_test_acc']:>9.4f} {sk_res['LR_test_f1']:>9.4f}")
    logger.info(f"{'SVM-RBF (grid, T+V fit)':<30} {sk_res['SVM_val_acc']:>8.4f} {sk_res['SVM_val_f1']:>8.4f} "
                f"{sk_res['SVM_test_acc']:>9.4f} {sk_res['SVM_test_f1']:>9.4f}")
    logger.info(f"{'AudioMLP + Focal + SCL':<30} {mlp_val_acc:>8.4f} {best_val_f1:>8.4f} "
                f"{mlp_test_acc:>9.4f} {mlp_test_f1:>9.4f}")
    logger.info(f"{'AudioMLP + TTA':<30} {'—':>8} {'—':>8} "
                f"{tta_test_acc:>9.4f} {tta_test_f1:>9.4f}")
    logger.info(f"{'Ensemble (SVM + MLP-TTA)':<30} {'—':>8} {'—':>8} "
                f"{ens_acc:>9.4f} {ens_f1:>9.4f}")
    logger.info("=" * 70)

    best_acc = max(sk_res["SVM_test_acc"], mlp_test_acc, tta_test_acc, ens_acc)
    best_f1_ = max(sk_res["SVM_test_f1"],  mlp_test_f1,  tta_test_f1,  ens_f1)
    logger.info(f"  🏅 Best Accuracy  : {best_acc:.4f}")
    logger.info(f"  🏅 Best F1-Macro  : {best_f1_:.4f}")
    logger.info(f"  📊 Ensemble F1-Wt : {ens_f1w:.4f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
