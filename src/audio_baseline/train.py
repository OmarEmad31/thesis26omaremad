"""Train emotion classifier on emotion2vec embeddings.

Why the old model hit 42% ceiling:
  - Class imbalance → model predicted majority only → F1-macro ~22%
  - patience=25 → early stopping fired before the model properly learned
  - SCL temp=0.3 → loss signal too weak, clusters barely moved
  - No oversampling → minority emotions starved of examples

This version fixes all of that:
  1. SVM-RBF with GridSearchCV                        (strong baseline)
  2. Deep Residual MLP
       + Focal Loss (handles imbalance at loss level)
       + Supervised Contrastive Loss (SCL, temp=0.07)
       + MixUp augmentation (70% of batches)
       + Gaussian noise injection
  3. Random oversampling → balanced training distribution
  4. Test-Time Augmentation (TTA, 8 passes)
  5. Ensemble: SVM + MLP averaged probabilities

Run from project root:
  python -m src.audio_baseline.train
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
from sklearn.preprocessing import StandardScaler, normalize as sk_normalize
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
# Data: random oversampling of minority classes
# ---------------------------------------------------------------------------
def balanced_oversample(emb: np.ndarray, labels: np.ndarray) -> tuple:
    """
    Under-represented classes are randomly duplicated (with replacement)
    until every class has as many samples as the majority class.
    Done in embedding space — no new audio needed.
    """
    rng    = np.random.default_rng(42)
    counts = collections.Counter(labels.tolist())
    max_n  = max(counts.values())
    parts_e, parts_l = [emb], [labels]
    for cls, n in counts.items():
        deficit = max_n - n
        if deficit > 0:
            mask = labels == cls
            idx  = rng.choice(mask.sum(), deficit, replace=True)
            parts_e.append(emb[mask][idx])
            parts_l.append(np.full(deficit, cls, dtype=np.int64))
    emb_out = np.concatenate(parts_e, axis=0)
    lbl_out = np.concatenate(parts_l, axis=0)
    # Shuffle so classes are interleaved
    perm    = rng.permutation(len(lbl_out))
    return emb_out[perm], lbl_out[perm]


# ---------------------------------------------------------------------------
# Model: Deep Residual MLP with cosine classifier
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    """Pre-norm residual: LayerNorm → Linear → GELU → Dropout → add."""
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.norm  = nn.LayerNorm(dim)
        self.lin   = nn.Linear(dim, dim)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.act(self.lin(self.norm(x))))


class DeepMLP(nn.Module):
    """
    Residual MLP on frozen emotion2vec embeddings.

    Architecture:
      [B, in_dim]
        LayerNorm → Linear(in→512) → GELU → Dropout(0.35)
        ResBlock(512, 0.20)
        LayerNorm → Linear(512→256) → GELU → Dropout(0.25)
        ResBlock(256, 0.15)
        ↓ proj [B, 256]  → L2-normalize → used for SCL
        Linear(256 → num_labels)
    """
    def __init__(self, input_dim: int = 768, num_labels: int = 7):
        super().__init__()
        self.stem = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.35),
        )
        self.res1 = ResBlock(512, 0.20)
        self.down = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.25),
        )
        self.res2       = ResBlock(256, 0.15)
        self.classifier = nn.Linear(256, num_labels)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """Returns (logits [B, C], proj [B, 256]) where proj is L2-normalised."""
        x    = self.stem(x)
        x    = self.res1(x)
        x    = self.down(x)
        proj = self.res2(x)                           # [B, 256]
        proj_n = F.normalize(proj, p=2, dim=1)        # L2-norm for SCL
        logits = self.classifier(proj)                # [B, num_labels]
        return logits, proj_n


# ---------------------------------------------------------------------------
# Focal Loss  (down-weights easy / majority-class predictions)
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    FL(p_t) = -(1 - p_t)^γ · log(p_t)
    γ=2 is the standard value from the original FPN paper.
    Combined with class weights and label smoothing.
    """
    def __init__(self, weight: torch.Tensor = None,
                 gamma: float = 2.0, label_smoothing: float = 0.1):
        super().__init__()
        self.weight          = weight
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce    = F.cross_entropy(logits, targets,
                                weight=self.weight,
                                label_smoothing=self.label_smoothing,
                                reduction="none")
        p_t   = torch.exp(-ce)                        # predicted prob for correct class
        focal = ((1.0 - p_t) ** self.gamma) * ce
        return focal.mean()


# ---------------------------------------------------------------------------
# Supervised Contrastive Loss  (Khosla et al., 2020)
# ---------------------------------------------------------------------------
def scl_loss(proj: torch.Tensor, labels: torch.Tensor,
             temp: float, device: torch.device) -> torch.Tensor:
    """proj must already be L2-normalised; temp=0.07 is the SupCon default."""
    sim       = torch.matmul(proj, proj.T) / temp

    bs        = labels.size(0)
    pos_mask  = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
    self_mask = 1.0 - torch.eye(bs, device=device)
    pos_mask  = pos_mask * self_mask

    max_sim, _ = sim.max(dim=1, keepdim=True)
    sim        = sim - max_sim.detach()
    exp_sim    = torch.exp(sim) * self_mask
    log_prob   = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    valid = pos_mask.sum(1) > 0
    if not valid.any():
        return torch.tensor(0.0, device=device)
    loss = -(pos_mask[valid] * log_prob[valid]).sum(1) / (pos_mask[valid].sum(1) + 1e-8)
    return loss.mean()


# ---------------------------------------------------------------------------
# MixUp (embedding space)
# ---------------------------------------------------------------------------
def mixup_batch(x, y, alpha, device):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=device)
    return lam * x + (1.0 - lam) * x[idx], y, y[idx], lam


def mixup_criterion(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------
def extract_embeddings(backbone, df: pd.DataFrame, label2id: dict,
                       split_name: str, first_call: bool = False):
    import librosa

    embeddings, labels_out = [], []
    skipped = 0
    detected_dim = None

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  Extracting {split_name}"):
        folder   = str(row["folder"]).strip()
        rel_path = str(row["audio_relpath"]).replace("\\", "/").lstrip("/")

        candidate = config.DATA_ROOT / folder / rel_path
        if not candidate.exists():
            try:
                for sub in config.DATA_ROOT.iterdir():
                    if sub.is_dir():
                        alt = sub / folder / rel_path
                        if alt.exists():
                            candidate = alt
                            break
            except Exception:
                pass

        if not candidate.exists():
            skipped += 1
            continue

        try:
            audio, _ = librosa.load(candidate, sr=config.SAMPLING_RATE, mono=True)
            if len(audio) > config.MAX_AUDIO_SAMPLES:
                audio = audio[: config.MAX_AUDIO_SAMPLES]

            with open(os.devnull, "w") as devnull:
                old_stdout, old_stderr = sys.stdout, sys.stderr
                sys.stdout = sys.stderr = devnull
                try:
                    res = backbone.generate(
                        input=audio, granularity="utterance", extract_embedding=True
                    )
                finally:
                    sys.stdout, sys.stderr = old_stdout, old_stderr

            if not res or "feats" not in res[0]:
                skipped += 1
                continue

            feat = np.array(res[0]["feats"])
            if feat.ndim == 2:
                feat = feat.mean(axis=0)
            feat = feat.flatten()

            if detected_dim is None:
                detected_dim = feat.shape[0]
                if first_call:
                    logger.info(f"    🔬 Embedding dim={feat.shape}  "
                                f"mean={feat.mean():.4f}  std={feat.std():.4f}")

            embeddings.append(feat.astype(np.float32))
            labels_out.append(label2id[row["emotion_final"]])

        except Exception:
            skipped += 1
            continue

    logger.info(f"    {split_name}: {len(embeddings)} extracted, {skipped} skipped, dim={detected_dim}")
    return np.array(embeddings, dtype=np.float32), np.array(labels_out, dtype=np.int64)


# ---------------------------------------------------------------------------
# SVM + GridSearch
# ---------------------------------------------------------------------------
def run_sklearn(train_emb, train_lbl, val_emb, val_lbl,
                test_emb, test_lbl, id2label):
    # L2-normalize first (emotion2vec embeddings live on a hypersphere)
    X_train = sk_normalize(train_emb, norm="l2")
    X_val   = sk_normalize(val_emb,   norm="l2")
    X_test  = sk_normalize(test_emb,  norm="l2")
    X_tv    = sk_normalize(np.concatenate([train_emb, val_emb]), norm="l2")
    y_tv    = np.concatenate([train_lbl, val_lbl])

    results = {}

    # 1. Logistic Regression (reference)
    logger.info("   Fitting Logistic Regression...")
    lr = LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs",
                            class_weight="balanced", random_state=42)
    lr.fit(X_train, train_lbl)
    for sn, X, y in [("val", X_val, val_lbl), ("test", X_test, test_lbl)]:
        p = lr.predict(X)
        results[f"LR_{sn}_acc"] = accuracy_score(y, p)
        results[f"LR_{sn}_f1"]  = f1_score(y, p, average="macro")

    # 2. SVM-RBF GridSearch
    logger.info("   Grid-searching SVM (C, gamma) — takes a few minutes...")
    param_grid = {
        "C":     [0.5, 1, 5, 10, 50, 100, 200],
        "gamma": ["scale", "auto"],
    }
    grid = GridSearchCV(
        SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
        param_grid, cv=5, scoring="f1_macro", n_jobs=-1, verbose=0,
    )
    grid.fit(X_train, train_lbl)
    logger.info(f"   ✅ Best SVM: C={grid.best_params_['C']}, "
                f"gamma={grid.best_params_['gamma']}  "
                f"(cv F1={grid.best_score_:.4f})")

    # Val score from grid CV model, test score from TV-retrained model
    svm_tv = SVC(kernel="rbf", probability=True, class_weight="balanced",
                 random_state=42, **grid.best_params_)
    svm_tv.fit(X_tv, y_tv)

    for sn, X, y, m in [("val",  X_val,  val_lbl,  grid.best_estimator_),
                         ("test", X_test, test_lbl, svm_tv)]:
        p = m.predict(X)
        results[f"SVM_{sn}_acc"] = accuracy_score(y, p)
        results[f"SVM_{sn}_f1"]  = f1_score(y, p, average="macro")

    logger.info("\n📊 SKLEARN RESULTS:")
    logger.info(f"   LR  — Val Acc {results['LR_val_acc']:.4f}  F1 {results['LR_val_f1']:.4f} | "
                f"Test Acc {results['LR_test_acc']:.4f}  F1 {results['LR_test_f1']:.4f}")
    logger.info(f"   SVM — Val Acc {results['SVM_val_acc']:.4f}  F1 {results['SVM_val_f1']:.4f} | "
                f"Test Acc {results['SVM_test_acc']:.4f}  F1 {results['SVM_test_f1']:.4f}")

    label_strs = [id2label[i] for i in range(len(id2label))]
    logger.info("\n   SVM Per-class Report (Test):")
    logger.info(classification_report(test_lbl, svm_tv.predict(X_test),
                                      target_names=label_strs))

    return results, svm_tv, X_tv, X_test


# ---------------------------------------------------------------------------
# Deep MLP training
# ---------------------------------------------------------------------------
def train_deep_mlp(train_emb, train_lbl, val_emb, val_lbl,
                   test_emb, test_lbl, id2label, device):
    num_labels = len(id2label)
    emb_dim    = train_emb.shape[1]

    # --- Oversample training data so every class is equally represented ---
    if config.USE_OVERSAMPLING:
        train_emb_os, train_lbl_os = balanced_oversample(train_emb, train_lbl)
        before = collections.Counter(train_lbl.tolist())
        after  = collections.Counter(train_lbl_os.tolist())
        logger.info(f"   Oversampling: {sum(before.values())} → {sum(after.values())} samples")
        logger.info(f"   Before: {dict(sorted(before.items()))}")
        logger.info(f"   After : {dict(sorted(after.items()))}")
    else:
        train_emb_os, train_lbl_os = train_emb, train_lbl

    # Tensors
    Xtr = torch.tensor(train_emb_os, dtype=torch.float32)
    ytr = torch.tensor(train_lbl_os, dtype=torch.long)
    Xv  = torch.tensor(val_emb,      dtype=torch.float32)
    yv  = torch.tensor(val_lbl,      dtype=torch.long)

    train_loader = DataLoader(TensorDataset(Xtr, ytr),
                              batch_size=config.BATCH_SIZE,
                              shuffle=True, drop_last=False)
    val_loader   = DataLoader(TensorDataset(Xv, yv),
                              batch_size=256, shuffle=False)

    model = DeepMLP(input_dim=emb_dim, num_labels=num_labels).to(device)

    # Class weights computed from ORIGINAL (not oversampled) distribution
    weights   = compute_class_weight("balanced",
                                     classes=np.arange(num_labels), y=train_lbl)
    cw        = torch.tensor(weights, dtype=torch.float).to(device)
    criterion = FocalLoss(weight=cw, gamma=config.FOCAL_GAMMA, label_smoothing=0.1)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.LEARNING_RATE,
                                  weight_decay=config.WEIGHT_DECAY)

    # CosineAnnealingWarmRestarts: restarts help escape local minima.
    # T_0=50 → first restart at epoch 50, then 100, 200 (T_mult=2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )

    best_f1    = 0.0
    patience   = 0
    best_state = None

    logger.info(f"\n⚡ Training Deep Residual MLP + SCL + Focal + MixUp + Noise")
    logger.info(f"   dim={emb_dim}  batch={config.BATCH_SIZE}  oversample=on")
    logger.info(f"   lr={config.LEARNING_RATE}  wd={config.WEIGHT_DECAY}")
    logger.info(f"   scl_w={config.SCL_WEIGHT}  scl_t={config.SCL_TEMP}  focal_γ={config.FOCAL_GAMMA}")
    logger.info(f"   mixup={config.MIXUP_PROB:.0%} of batches  noise_std={config.NOISE_STD}")
    logger.info(f"   epochs={config.EPOCHS}  patience={config.MAX_PATIENCE}")

    for epoch in range(config.EPOCHS):
        model.train()
        t_correct = t_total = 0
        t_ce = t_scl = 0.0

        for emb_b, lab_b in train_loader:
            emb_b, lab_b = emb_b.to(device), lab_b.to(device)

            # Gaussian noise injection
            if config.NOISE_STD > 0:
                emb_b = emb_b + torch.randn_like(emb_b) * config.NOISE_STD

            use_mixup = config.USE_MIXUP and (np.random.random() < config.MIXUP_PROB)
            if use_mixup:
                emb_b, y_a, y_b, lam = mixup_batch(emb_b, lab_b, config.MIXUP_ALPHA, device)
                logits, proj = model(emb_b)
                ce  = mixup_criterion(criterion, logits, y_a, y_b, lam)
                sc  = scl_loss(proj, y_a, config.SCL_TEMP, device) if config.USE_SCL else torch.tensor(0.0)
                p   = logits.argmax(1)
                t_correct += ((p == y_a).float() * lam + (p == y_b).float() * (1.0 - lam)).sum().item()
            else:
                logits, proj = model(emb_b)
                ce  = criterion(logits, lab_b)
                sc  = scl_loss(proj, lab_b, config.SCL_TEMP, device) if config.USE_SCL else torch.tensor(0.0)
                t_correct += (logits.argmax(1) == lab_b).sum().item()

            loss = ce + config.SCL_WEIGHT * sc

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            t_total += lab_b.size(0)
            t_ce    += ce.item()
            t_scl   += sc.item()

        scheduler.step()

        # Validation (no noise, no mixup)
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for emb_b, lab_b in val_loader:
                logits, _ = model(emb_b.to(device))
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_true.extend(lab_b.numpy())

        val_acc = accuracy_score(all_true, all_preds)
        val_f1  = f1_score(all_true, all_preds, average="macro")

        improved = val_f1 > best_f1
        if improved:
            best_f1    = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience   = 0
        else:
            patience += 1

        if (epoch + 1) % config.LOG_EVERY == 0 or epoch == 0 or improved:
            star = "⭐" if improved else f"(patience {patience}/{config.MAX_PATIENCE})"
            logger.info(
                f"  Ep {epoch:03d} | Tr-Acc {t_correct/t_total:.4f} | "
                f"Val-Acc {val_acc:.4f} | Val-F1 {val_f1:.4f} {star}  "
                f"[CE={t_ce/len(train_loader):.3f} SCL={t_scl/len(train_loader):.3f}]"
            )

        if patience >= config.MAX_PATIENCE:
            logger.info(f"  ⏹️  Early stopping at epoch {epoch} (best val F1={best_f1:.4f})")
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_f1


# ---------------------------------------------------------------------------
# Evaluation: clean + TTA
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_mlp(model, emb_np: np.ndarray, lbl_np: np.ndarray,
                 device, tta: bool = False):
    """
    Returns (true_labels, predictions, probabilities).
    If tta=True: averages softmax over TTA_PASSES noisy forward passes.
    """
    model.eval()
    n_passes = config.TTA_PASSES if tta else 1
    noise    = config.TTA_NOISE  if tta else 0.0

    accum_probs = None
    for _ in range(n_passes):
        x = torch.tensor(emb_np, dtype=torch.float32)
        if noise > 0:
            x = x + torch.randn_like(x) * noise
        loader = DataLoader(TensorDataset(x, torch.tensor(lbl_np)),
                            batch_size=256, shuffle=False)
        pass_probs = []
        for emb_b, _ in loader:
            logits, _ = model(emb_b.to(device))
            pass_probs.append(F.softmax(logits, dim=-1).cpu().numpy())
        pass_probs = np.concatenate(pass_probs, axis=0)   # [N, C]

        accum_probs = pass_probs if accum_probs is None else accum_probs + pass_probs

    avg_probs = accum_probs / n_passes
    preds     = avg_probs.argmax(axis=1)
    return lbl_np, preds, avg_probs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    set_seed(42)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    logger.info(f"🚀 Device: {device}")
    logger.info(f"🧠 Backbone: {config.MODEL_NAME}")

    # 1. Load splits
    logger.info(f"\n📁 Loading splits from {config.SPLIT_CSV_DIR.name}...")
    train_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv")
    val_df   = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv")
    test_df  = pd.read_csv(config.SPLIT_CSV_DIR / "test.csv")

    label_names = sorted(train_df["emotion_final"].unique())
    label2id    = {n: i for i, n in enumerate(label_names)}
    id2label    = {i: n for n, i in label2id.items()}
    logger.info(f"🎭 Emotions ({len(label2id)}): {label2id}")
    logger.info(f"   Train {len(train_df)} | Val {len(val_df)} | Test {len(test_df)}")

    # 2. Embedding extraction (cached to disk)
    cache = config.EMBEDDING_CACHE
    if cache.exists():
        logger.info(f"⚡ Loading cached embeddings from {cache}...")
        data      = np.load(cache)
        train_emb = data["train_emb"];  train_lbl = data["train_lbl"]
        val_emb   = data["val_emb"];    val_lbl   = data["val_lbl"]
        test_emb  = data["test_emb"];   test_lbl  = data["test_lbl"]
    else:
        logger.info("🧠 Loading emotion2vec backbone...")
        from funasr import AutoModel
        backbone = AutoModel(model=config.MODEL_NAME, hub="hf", trust_remote_code=True)
        backbone.model.eval()
        for p in backbone.model.parameters():
            p.requires_grad = False

        logger.info("🔬 Extracting embeddings (cached after first run)...")
        train_emb, train_lbl = extract_embeddings(backbone, train_df, label2id, "train", first_call=True)
        val_emb,   val_lbl   = extract_embeddings(backbone, val_df,   label2id, "val")
        test_emb,  test_lbl  = extract_embeddings(backbone, test_df,  label2id, "test")

        np.savez_compressed(cache,
            train_emb=train_emb, train_lbl=train_lbl,
            val_emb=val_emb,     val_lbl=val_lbl,
            test_emb=test_emb,   test_lbl=test_lbl)
        logger.info(f"💾 Cached to {cache}")
        del backbone
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    logger.info(f"   Embedding dim: {train_emb.shape[1]}")
    logger.info(f"   Train {len(train_lbl)} | Val {len(val_lbl)} | Test {len(test_lbl)}")
    logger.info(f"   Train class dist: {dict(sorted(collections.Counter(train_lbl.tolist()).items()))}")
    logger.info(f"   Val   class dist: {dict(sorted(collections.Counter(val_lbl.tolist()).items()))}")

    # 3. SVM with grid search
    sklearn_results, svm_model, X_tv_norm, X_test_norm = run_sklearn(
        train_emb, train_lbl, val_emb, val_lbl, test_emb, test_lbl, id2label
    )

    # 4. Deep MLP + Focal + SCL + MixUp + oversampling
    mlp_model, mlp_best_val_f1 = train_deep_mlp(
        train_emb, train_lbl, val_emb, val_lbl, test_emb, test_lbl, id2label, device
    )

    # 5. MLP evaluation (clean + TTA)
    true_te, preds_clean, probs_clean = evaluate_mlp(mlp_model, test_emb, test_lbl, device, tta=False)
    _,       preds_tta,   probs_tta   = evaluate_mlp(mlp_model, test_emb, test_lbl, device, tta=True)
    true_v,  preds_v,     _           = evaluate_mlp(mlp_model, val_emb,  val_lbl,  device, tta=False)

    mlp_test_acc  = accuracy_score(true_te, preds_clean)
    mlp_test_f1   = f1_score(true_te, preds_clean, average="macro")
    mlp_test_f1w  = f1_score(true_te, preds_clean, average="weighted")
    tta_test_acc  = accuracy_score(true_te, preds_tta)
    tta_test_f1   = f1_score(true_te, preds_tta, average="macro")
    mlp_val_acc   = accuracy_score(true_v, preds_v)

    # 6. Ensemble: SVM probs + MLP-TTA probs
    svm_probs_te = svm_model.predict_proba(X_test_norm)
    ens_probs    = 0.5 * svm_probs_te + 0.5 * probs_tta
    ens_preds    = ens_probs.argmax(axis=1)
    ens_acc      = accuracy_score(test_lbl, ens_preds)
    ens_f1       = f1_score(test_lbl, ens_preds, average="macro")
    ens_f1w      = f1_score(test_lbl, ens_preds, average="weighted")

    # 7. Reports
    label_strs = [id2label[i] for i in range(len(id2label))]

    logger.info("\n   Deep MLP Per-class Report (Test, clean):")
    logger.info(classification_report(true_te, preds_clean, target_names=label_strs))

    if config.USE_TTA:
        logger.info("\n   Deep MLP Per-class Report (Test, TTA):")
        logger.info(classification_report(true_te, preds_tta, target_names=label_strs))

    logger.info("\n   Ensemble (SVM + MLP-TTA) Per-class Report (Test):")
    logger.info(classification_report(test_lbl, ens_preds, target_names=label_strs))

    logger.info("\n" + "=" * 68)
    logger.info("🏆  FINAL REPORT — AUDIO EMOTION CLASSIFICATION")
    logger.info("=" * 68)
    logger.info(f"{'Method':<32} {'Val Acc':>8} {'Val F1':>8} {'Test Acc':>9} {'Test F1':>9}")
    logger.info("-" * 68)
    logger.info(f"{'LR (reference)':<32} {sklearn_results['LR_val_acc']:>8.4f} "
                f"{sklearn_results['LR_val_f1']:>8.4f} "
                f"{sklearn_results['LR_test_acc']:>9.4f} {sklearn_results['LR_test_f1']:>9.4f}")
    logger.info(f"{'SVM-RBF (grid, TV-trained)':<32} {sklearn_results['SVM_val_acc']:>8.4f} "
                f"{sklearn_results['SVM_val_f1']:>8.4f} "
                f"{sklearn_results['SVM_test_acc']:>9.4f} {sklearn_results['SVM_test_f1']:>9.4f}")
    logger.info(f"{'Deep MLP + SCL + Focal':<32} {mlp_val_acc:>8.4f} "
                f"{mlp_best_val_f1:>8.4f} "
                f"{mlp_test_acc:>9.4f} {mlp_test_f1:>9.4f}")
    logger.info(f"{'Deep MLP + TTA':<32} {'—':>8} {'—':>8} "
                f"{tta_test_acc:>9.4f} {tta_test_f1:>9.4f}")
    logger.info(f"{'Ensemble (SVM + MLP-TTA)':<32} {'—':>8} {'—':>8} "
                f"{ens_acc:>9.4f} {ens_f1:>9.4f}")
    logger.info("=" * 68)

    best_acc = max(sklearn_results['SVM_test_acc'], mlp_test_acc, tta_test_acc, ens_acc)
    best_f1  = max(sklearn_results['SVM_test_f1'],  mlp_test_f1,  tta_test_f1,  ens_f1)
    logger.info(f"  🏅 Best Test Accuracy : {best_acc:.4f}")
    logger.info(f"  🏅 Best Test F1-Macro : {best_f1:.4f}")
    logger.info(f"  📊 Ensemble F1-Weighted: {ens_f1w:.4f}")
    logger.info("=" * 68)


if __name__ == "__main__":
    main()
