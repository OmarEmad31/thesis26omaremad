"""Audio emotion classifier — complete, correct implementation.

ROOT CAUSE OF PREVIOUS 28-42% CEILING
  1. Manifest was built on Windows → only 718/5700 files marked as audio_exists=1
     FIX → rebuild manifest on Colab before running (see README at bottom)
  2. L2-normalize broke SVM (reverted to StandardScaler which gave 42%)
  3. emotion2vec_plus_large WORSE than base on this dataset (reverted)
  4. SCL temp=0.07 too aggressive, barely converging (now 0.20)
  5. Oversampling 511 → 1036 duplicates → train 72%, val 29% (removed)

Pipeline:
  1. Load emotion2vec_plus_base backbone → extract 768-dim utterance embeddings
  2. SVM-RBF with GridSearchCV (proven best for high-dim, small-N data)
  3. MLP with Focal Loss + SCL(temp=0.2) + MixUp      (thesis SCL contribution)
  4. Ensemble: SVM + MLP probability averaging
  5. Final report

Run from project root:
  python -m src.audio_baseline.train

COLAB SETUP (do BEFORE running train):
  # Step 1 — rebuild manifest with actual Colab paths (detects all 5700 files)
  !python scripts/build_manifest.py
  # Step 2 — rebuild splits (gives ~3500 train / 750 val / 750 test)
  !python scripts/split_dataset.py
  # Step 3 — delete old embedding cache (different backbone / more files)
  import os; os.remove("/content/audio_embeddings.npz") if os.path.exists("/content/audio_embeddings.npz") else None
  # Step 4 — train
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
# Model
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    """Pre-norm residual: LayerNorm → Linear → GELU → Dropout → add."""
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return x + self.net(x)


class AudioMLP(nn.Module):
    """
    Lightweight but expressive classifier on top of frozen emotion2vec embeddings.

    Architecture  [in=768]:
      LN → Linear(768→512) → GELU → Dropout(0.40)
      ResBlock(512, 0.30)
      LN → Linear(512→256) → GELU → Dropout(0.35)
      ResBlock(256, 0.25)       ← proj head for SCL
      Linear(256 → num_labels)

    Dropout is intentionally heavy because N_train < 5000.
    """
    def __init__(self, input_dim: int = 768, num_labels: int = 7):
        super().__init__()
        self.stem = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.40),
        )
        self.res1 = ResBlock(512, 0.30)
        self.down = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.35),
        )
        self.res2       = ResBlock(256, 0.25)
        self.classifier = nn.Linear(256, num_labels)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """Returns (logits [B, C], proj [B, 256])."""
        x    = self.stem(x)
        x    = self.res1(x)
        x    = self.down(x)
        proj = self.res2(x)         # raw 256-dim projection
        logits = self.classifier(proj)
        return logits, proj


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """FL(p_t) = -(1-p_t)^γ · log(p_t).  Combined with class weights."""
    def __init__(self, weight=None, gamma: float = 1.5, label_smoothing: float = 0.1):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None
                             else torch.ones(1))
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce    = F.cross_entropy(logits, targets,
                                weight=self.weight,
                                label_smoothing=self.label_smoothing,
                                reduction="none")
        p_t   = torch.exp(-ce)
        focal = (1.0 - p_t) ** self.gamma * ce
        return focal.mean()


# ---------------------------------------------------------------------------
# Supervised Contrastive Loss   (Khosla et al., 2020)
# temp=0.20: looser than the original 0.07 — needed for small N and few positives
# ---------------------------------------------------------------------------
def supcon_loss(proj: torch.Tensor, labels: torch.Tensor,
                temp: float, device: torch.device) -> torch.Tensor:
    """
    proj : [B, D]  — raw projection vectors (will be L2-normalised inside)
    labels: [B]    — integer class indices
    """
    feat   = F.normalize(proj, p=2, dim=1)          # [B, D] unit sphere
    sim    = torch.matmul(feat, feat.T) / temp       # [B, B]

    bs     = labels.size(0)
    I      = torch.eye(bs, device=device)
    pos    = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float() * (1 - I)
    neg_I  = 1 - I                                  # all pairs except self

    # Stability: subtract row max
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim        = sim - sim_max.detach()

    exp_sim  = torch.exp(sim) * neg_I
    log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)

    valid = pos.sum(1) > 0
    if not valid.any():
        return torch.tensor(0.0, device=device)

    loss = -(pos[valid] * log_prob[valid]).sum(1) / pos[valid].sum(1).clamp(min=1)
    return loss.mean()


# ---------------------------------------------------------------------------
# MixUp
# ---------------------------------------------------------------------------
def mixup(x, y, alpha, device):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def mixup_loss(criterion, logits, ya, yb, lam):
    return lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------
def extract_embeddings(backbone, df: pd.DataFrame, label2id: dict,
                       split_name: str, first_call: bool = False):
    import librosa
    embeddings, labels_out, skipped = [], [], 0
    detected_dim = None

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {split_name}"):
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

            # Silence funasr's own stdout/stderr
            with open(os.devnull, "w") as devnull:
                old_out, old_err = sys.stdout, sys.stderr
                sys.stdout = sys.stderr = devnull
                try:
                    res = backbone.generate(
                        input=audio, granularity="utterance", extract_embedding=True
                    )
                finally:
                    sys.stdout, sys.stderr = old_out, old_err

            if not res or "feats" not in res[0]:
                skipped += 1
                continue

            feat = np.array(res[0]["feats"])
            if feat.ndim == 2:
                feat = feat.mean(axis=0)
            feat = feat.flatten().astype(np.float32)

            if detected_dim is None:
                detected_dim = feat.shape[0]
                if first_call:
                    logger.info(f"    🔬 Embedding dim={detected_dim}"
                                f"  mean={feat.mean():.4f}  std={feat.std():.4f}")

            embeddings.append(feat)
            labels_out.append(label2id[row["emotion_final"]])

        except Exception:
            skipped += 1

    logger.info(f"    {split_name}: {len(embeddings)} ok | {skipped} skipped | dim={detected_dim}")
    return (np.array(embeddings, dtype=np.float32),
            np.array(labels_out,  dtype=np.int64))


# ---------------------------------------------------------------------------
# SVM with GridSearchCV  (StandardScaler — DO NOT use L2-normalize for RBF)
# ---------------------------------------------------------------------------
def run_sklearn(train_emb, train_lbl, val_emb, val_lbl,
                test_emb, test_lbl, id2label):
    """
    StandardScaler + SVM-RBF with grid search.
    Final model is trained on train+val (more data → better generalisation).
    """
    scaler = StandardScaler()
    Xtr    = scaler.fit_transform(train_emb)
    Xva    = scaler.transform(val_emb)
    Xte    = scaler.transform(test_emb)
    # Train+val for final SVM
    Xtv    = scaler.transform(np.concatenate([train_emb, val_emb]))
    ytv    = np.concatenate([train_lbl, val_lbl])

    results = {}

    # 1. Logistic Regression (quick reference)
    logger.info("   [LR] fitting...")
    lr = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced",
                            solver="lbfgs", random_state=42)
    lr.fit(Xtr, train_lbl)
    for sn, X, y in [("val", Xva, val_lbl), ("test", Xte, test_lbl)]:
        p = lr.predict(X)
        results[f"LR_{sn}_acc"] = accuracy_score(y, p)
        results[f"LR_{sn}_f1"]  = f1_score(y, p, average="macro")

    # 2. SVM-RBF — GridSearchCV on training fold, then retrain on TV
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

    # Re-fit on train+val for best test accuracy
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
    dist    = collections.Counter(train_lbl.tolist())
    logger.info(f"\n   Training dist ({N_train} samples): {dict(sorted(dist.items()))}")
    if N_train < 1000:
        logger.warning("   ⚠️  Only {N_train} training samples detected!"
                       " Run build_manifest.py + split_dataset.py on Colab"
                       " first to unlock all ~5000 files.")

    Xtr = torch.tensor(train_emb, dtype=torch.float32)
    ytr = torch.tensor(train_lbl, dtype=torch.long)
    Xva = torch.tensor(val_emb,   dtype=torch.float32)
    yva = torch.tensor(val_lbl,   dtype=torch.long)

    # NOTE: No oversampling — class-weighted loss handles imbalance cleanly.
    # Oversampling with this dataset size causes memorisation (train≫val).
    train_loader = DataLoader(TensorDataset(Xtr, ytr),
                              batch_size=config.BATCH_SIZE,
                              shuffle=True, drop_last=False)
    val_loader   = DataLoader(TensorDataset(Xva, yva),
                              batch_size=256, shuffle=False)

    model = AudioMLP(input_dim=emb_dim, num_labels=num_labels).to(device)

    # Class weights (computed from real train dist, not oversampled)
    raw_weights = compute_class_weight("balanced",
                                       classes=np.arange(num_labels),
                                       y=train_lbl)
    cw     = torch.tensor(raw_weights, dtype=torch.float, device=device)
    criterion = FocalLoss(weight=cw, gamma=config.FOCAL_GAMMA, label_smoothing=0.1)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.LEARNING_RATE,
                                  weight_decay=config.WEIGHT_DECAY)

    # OneCycleLR: warm-up (10%) then cosine anneal — standard best practice
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

    best_f1    = 0.0
    patience   = 0
    best_state = None

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

            if config.USE_MIXUP and np.random.random() < config.MIXUP_PROB:
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
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            sum_n  += yb.size(0)
            sum_ce += ce.item()
            sum_sc += sc.item()

        # Validation (clean, no augmentation)
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

        log_now = improved or ((epoch + 1) % config.LOG_EVERY == 0) or epoch == 0
        if log_now:
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
    passes  = config.TTA_PASSES if tta else 1
    noise   = config.TTA_NOISE  if tta else 0.0
    acc_p   = None

    for _ in range(passes):
        x = torch.tensor(emb_np, dtype=torch.float32)
        if noise > 0:
            x = x + torch.randn_like(x) * noise
        loader = DataLoader(TensorDataset(x, torch.tensor(lbl_np)),
                            batch_size=256, shuffle=False)
        probs_p = []
        for xb, _ in loader:
            logits, _ = model(xb.to(device))
            probs_p.append(F.softmax(logits, -1).cpu().numpy())
        probs_p = np.concatenate(probs_p)
        acc_p   = probs_p if acc_p is None else acc_p + probs_p

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

    # 1. Load splits
    logger.info(f"📁 {config.SPLIT_CSV_DIR}")
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
    else:
        logger.info("🧠 Loading backbone (runs once, then cached)...")
        from funasr import AutoModel
        bb = AutoModel(model=config.MODEL_NAME, hub="hf", trust_remote_code=True)
        bb.model.eval()
        for p in bb.model.parameters():
            p.requires_grad = False

        train_emb, train_lbl = extract_embeddings(bb, train_df, label2id, "train", True)
        val_emb,   val_lbl   = extract_embeddings(bb, val_df,   label2id, "val")
        test_emb,  test_lbl  = extract_embeddings(bb, test_df,  label2id, "test")

        np.savez_compressed(cache,
            train_emb=train_emb, train_lbl=train_lbl,
            val_emb=val_emb,     val_lbl=val_lbl,
            test_emb=test_emb,   test_lbl=test_lbl)
        logger.info(f"💾 Saved to {cache}")
        del bb
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(f"   Embeddings — dim={train_emb.shape[1]} | "
                f"Train={len(train_lbl)} | Val={len(val_lbl)} | Test={len(test_lbl)}")
    logger.info(f"   Train dist: {dict(sorted(collections.Counter(train_lbl.tolist()).items()))}")

    # 3. SVM
    sk_res, scaler, svm = run_sklearn(
        train_emb, train_lbl, val_emb, val_lbl, test_emb, test_lbl, id2label
    )

    # 4. MLP
    mlp, best_val_f1 = train_mlp(
        train_emb, train_lbl, val_emb, val_lbl, id2label, device
    )

    # 5. Evaluate MLP (clean + TTA)
    true_te, pred_clean, prob_clean = evaluate(mlp, test_emb, test_lbl, device, tta=False)
    _,       pred_tta,   prob_tta   = evaluate(mlp, test_emb, test_lbl, device, tta=True)
    true_va, pred_va,    _          = evaluate(mlp, val_emb,  val_lbl,  device, tta=False)

    mlp_test_acc = accuracy_score(true_te, pred_clean)
    mlp_test_f1  = f1_score(true_te, pred_clean, average="macro")
    tta_test_acc = accuracy_score(true_te, pred_tta)
    tta_test_f1  = f1_score(true_te, pred_tta,   average="macro")
    mlp_val_acc  = accuracy_score(true_va, pred_va)

    # 6. Ensemble: SVM probs + MLP-TTA probs
    # SVM uses StandardScaler; re-apply the same scaler
    svm_prob_te = svm.predict_proba(scaler.transform(test_emb))
    ens_prob    = 0.5 * svm_prob_te + 0.5 * prob_tta
    ens_pred    = ens_prob.argmax(1)
    ens_acc     = accuracy_score(test_lbl, ens_pred)
    ens_f1      = f1_score(test_lbl, ens_pred, average="macro")
    ens_f1w     = f1_score(test_lbl, ens_pred, average="weighted")

    # 7. Reports
    label_strs = [id2label[i] for i in range(len(id2label))]

    logger.info("\n   MLP (clean) Test Report:")
    logger.info(classification_report(true_te, pred_clean, target_names=label_strs))
    if config.USE_TTA:
        logger.info("   MLP (TTA) Test Report:")
        logger.info(classification_report(true_te, pred_tta, target_names=label_strs))
    logger.info("   Ensemble Test Report:")
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
    best_f1  = max(sk_res["SVM_test_f1"],  mlp_test_f1,  tta_test_f1,  ens_f1)
    logger.info(f"  🏅 Best Accuracy  : {best_acc:.4f}")
    logger.info(f"  🏅 Best F1-Macro  : {best_f1:.4f}")
    logger.info(f"  📊 Ensemble F1-Wt : {ens_f1w:.4f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
