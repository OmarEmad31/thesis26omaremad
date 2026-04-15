"""Audio Emotion Classification — All 6 Phases + SWA + Val-F1 Weighted Ensemble.

PHASES IMPLEMENTED:
  Phase 1: Fixed f-string bug + new cache (audio_emb_v6.npz) — never loads stale data
  Phase 2: Audio augmentation during MLP extraction (4× training samples: 648→2592)
  Phase 3: Frame-level embedding aggregation concat(mean,std,max) = 2304-dim for MLP
  Phase 4: Expanded ensemble — SVM + RF + GBM + KNN + MLP, weighted by val-F1
  Phase 5: SMOTE (no external deps) — synthesises minority class samples for MLP
  Phase 6: Separate SCL projection head — decouples representation from classification

EXTRAS:
  SWA   — stochastic weight averaging (last 30% of training) for better generalisation
  Dual  — SVM/RF/GBM/KNN use utterance 768-dim clean, MLP uses frame 2304-dim augmented

COLAB:
  !pip install funasr librosa -q
  !git pull
  import os; [os.remove(p) for p in ["/content/audio_emb_v6.npz",
    "/content/audio_embeddings_dual.npz", "/content/audio_embeddings_frame_aug.npz",
    "/content/audio_embeddings.npz"] if os.path.exists(p)]
  !python -m src.audio_baseline.train
  # Extraction: ~10 min (SVM clean) + ~40 min (MLP augmented), then cached
"""

import sys, os, logging, warnings, collections
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from transformers import set_seed
from transformers import logging as hf_log
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report

warnings.filterwarnings("ignore")
hf_log.set_verbosity_error()
logging.getLogger("funasr").setLevel(logging.ERROR)

from src.audio_baseline import config

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
log_file = config.CHECKPOINT_DIR.parent / "audio_training_log.txt"
log_file.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format="%(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ===========================================================================
# PHASE 3 — Frame aggregation
# ===========================================================================
def aggregate_frames(feat: np.ndarray) -> np.ndarray:
    """[T,D] → concat(mean, std, max) = [3*D].  [D] → padded to [3*D]."""
    if feat.ndim == 1:
        return np.concatenate([feat, np.zeros_like(feat), feat])
    return np.concatenate([feat.mean(0), feat.std(0), feat.max(0)])


# ===========================================================================
# PHASE 2 — Audio augmentations
# ===========================================================================
def get_augmentations(audio: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """4 versions: orig + speed-slow + speed-fast + noise.  Always starts with orig."""
    import librosa
    out = [("orig", audio.copy())]
    for rate in config.AUG_SPEED_RATES:
        try:
            out.append((f"spd{rate}",
                        librosa.effects.time_stretch(audio.astype(np.float32), rate=rate)))
        except Exception:
            pass
    noise = config.AUG_NOISE_STD * float(np.std(audio) + 1e-8)
    out.append(("noise", (audio + noise * np.random.randn(len(audio))).astype(np.float32)))
    return out


# ===========================================================================
# PHASE 5 — SMOTE (no external deps, pure numpy/sklearn)
# ===========================================================================
def simple_smote(X: np.ndarray, y: np.ndarray,
                 target_per_class: int, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic samples for minority classes via KNN interpolation.
    Each synthetic sample = convex combination of a real sample and one of its k neighbours.
    Only generates samples for classes with N < target_per_class.
    """
    classes = np.unique(y)
    Xs, ys  = [X], [y]
    for cls in classes:
        mask   = y == cls
        X_cls  = X[mask]
        n_cls  = X_cls.shape[0]
        needed = target_per_class - n_cls
        if needed <= 0 or n_cls < 2:
            continue
        kk = min(k, n_cls - 1)
        nn = NearestNeighbors(n_neighbors=kk + 1, metric="euclidean", n_jobs=-1)
        nn.fit(X_cls)
        _, idx = nn.kneighbors(X_cls)   # [N, k+1]; col 0 is self
        for _ in range(needed):
            i     = np.random.randint(n_cls)
            j     = idx[i, 1 + np.random.randint(kk)]
            lam   = np.random.random()
            synth = (lam * X_cls[i] + (1 - lam) * X_cls[j]).astype(np.float32)
            Xs.append(synth[None])
            ys.append(np.array([cls]))
    return np.vstack(Xs), np.concatenate(ys)


# ===========================================================================
# Embedding extraction (dual: utterance for sklearn, frame for MLP)
# ===========================================================================
def extract_embeddings(backbone, df: pd.DataFrame, label2id: dict,
                       split_name: str,
                       granularity: str = "utterance",
                       augment: bool = False,
                       first_call: bool = False):
    import librosa
    embs, labs, skipped = [], [], 0
    dim = None

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"  [{granularity[:3]}{'*' if augment else ' '}] {split_name}"):
        folder   = str(row["folder"]).strip()
        rel_path = str(row["audio_relpath"]).replace("\\", "/").lstrip("/")

        candidate = config.DATA_ROOT / folder / rel_path
        if not candidate.exists():
            for sub in (s for s in config.DATA_ROOT.iterdir() if s.is_dir()):
                alt = sub / folder / rel_path
                if alt.exists(): candidate = alt; break

        if not candidate.exists():
            skipped += 1; continue

        try:
            audio, _ = librosa.load(candidate, sr=config.SAMPLING_RATE, mono=True)
            if len(audio) > config.MAX_AUDIO_SAMPLES:
                audio = audio[: config.MAX_AUDIO_SAMPLES]
            audio = audio.astype(np.float32)
        except Exception:
            skipped += 1; continue

        label    = label2id[row["emotion_final"]]
        versions = get_augmentations(audio) if augment else [("orig", audio)]

        for _, wav in versions:
            if len(wav) > config.MAX_AUDIO_SAMPLES:
                wav = wav[: config.MAX_AUDIO_SAMPLES]
            try:
                with open(os.devnull, "w") as dn:
                    old_out, old_err = sys.stdout, sys.stderr
                    sys.stdout = sys.stderr = dn
                    try:
                        res = backbone.generate(input=wav, granularity=granularity,
                                                extract_embedding=True)
                    finally:
                        sys.stdout, sys.stderr = old_out, old_err

                if not res or "feats" not in res[0]: continue
                raw  = np.array(res[0]["feats"], dtype=np.float32)
                feat = aggregate_frames(raw) if granularity == "frame" else raw.flatten()
                if dim is None:
                    dim = feat.shape[0]
                    if first_call:
                        logger.info(f"    🔬 gran={granularity} raw={raw.shape} → dim={dim}")
                embs.append(feat); labs.append(label)
            except Exception:
                continue

    logger.info(f"    {split_name}: {len(df)-skipped} files → {len(embs)} embs"
                f"  ({skipped} skip)  dim={dim}")
    return np.array(embs, dtype=np.float32), np.array(labs, dtype=np.int64)


# ===========================================================================
# PHASE 4 — Expanded sklearn classifiers + val-F1 weighted ensemble
# ===========================================================================
def run_sklearn(svm_tr_emb, svm_tr_lbl, svm_va_emb, svm_va_lbl,
                svm_te_emb, svm_te_lbl, id2label):
    """
    Trains 5 classifiers on utterance-level 768-dim clean data.
    Returns val-F1 weighted ensemble probabilities for the test set.
    """
    num_cls = len(id2label)
    scaler  = StandardScaler()
    Xtr = scaler.fit_transform(svm_tr_emb)
    Xva = scaler.transform(svm_va_emb)
    Xte = scaler.transform(svm_te_emb)
    Xtv = scaler.transform(np.concatenate([svm_tr_emb, svm_va_emb]))
    ytv = np.concatenate([svm_tr_lbl, svm_va_lbl])

    results, val_f1s, te_probs = {}, {}, {}

    # ── LR reference ──────────────────────────────────────────────────────
    logger.info("   [LR] fitting...")
    lr = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced",
                            solver="lbfgs", random_state=42)
    lr.fit(Xtr, svm_tr_lbl)
    for sn, X, y in [("val", Xva, svm_va_lbl), ("test", Xte, svm_te_lbl)]:
        p = lr.predict(X)
        results[f"LR_{sn}_acc"] = accuracy_score(y, p)
        results[f"LR_{sn}_f1"]  = f1_score(y, p, average="macro")

    # ── SVM-RBF (12 fits: 4C × 1γ × 3cv) ─────────────────────────────────
    logger.info("   [SVM] grid C∈{1,10,100,500} γ=scale (12 fits)...")
    gs = GridSearchCV(
        SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
        {"C": [1, 10, 100, 500], "gamma": ["scale"]},
        cv=3, scoring="f1_macro", n_jobs=-1, verbose=0)
    gs.fit(Xtr, svm_tr_lbl)
    best = gs.best_params_
    logger.info(f"   ✅ SVM best C={best['C']}  cv-F1={gs.best_score_:.4f}")
    svm_tv = SVC(kernel="rbf", probability=True, class_weight="balanced",
                 random_state=42, **best)
    svm_tv.fit(Xtv, ytv)
    svm_vf = f1_score(svm_va_lbl, gs.best_estimator_.predict(Xva), average="macro")
    val_f1s["svm"] = svm_vf
    te_probs["svm"] = svm_tv.predict_proba(Xte)
    for sn, X, y, m in [("val", Xva, svm_va_lbl, gs.best_estimator_),
                         ("test", Xte, svm_te_lbl, svm_tv)]:
        p = m.predict(X)
        results[f"SVM_{sn}_acc"] = accuracy_score(y, p)
        results[f"SVM_{sn}_f1"]  = f1_score(y, p, average="macro")

    # ── Random Forest ──────────────────────────────────────────────────────
    logger.info("   [RF] fitting (200 trees)...")
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                max_depth=None, min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf.fit(Xtr, svm_tr_lbl)
    val_f1s["rf"]   = f1_score(svm_va_lbl, rf.predict(Xva), average="macro")
    te_probs["rf"]  = rf.predict_proba(Xte)
    results["RF_val_acc"]  = accuracy_score(svm_va_lbl, rf.predict(Xva))
    results["RF_val_f1"]   = val_f1s["rf"]
    results["RF_test_acc"] = accuracy_score(svm_te_lbl, rf.predict(Xte))
    results["RF_test_f1"]  = f1_score(svm_te_lbl, rf.predict(Xte), average="macro")
    logger.info(f"   ✅ RF val-F1={val_f1s['rf']:.4f}  test-F1={results['RF_test_f1']:.4f}")

    # ── Gradient Boosting ─────────────────────────────────────────────────
    logger.info("   [GBM] fitting (100 estimators)...")
    gbm = GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                     learning_rate=0.1, subsample=0.8, random_state=42)
    gbm.fit(Xtr, svm_tr_lbl)
    val_f1s["gbm"]   = f1_score(svm_va_lbl, gbm.predict(Xva), average="macro")
    te_probs["gbm"]  = gbm.predict_proba(Xte)
    results["GBM_val_acc"]  = accuracy_score(svm_va_lbl, gbm.predict(Xva))
    results["GBM_val_f1"]   = val_f1s["gbm"]
    results["GBM_test_acc"] = accuracy_score(svm_te_lbl, gbm.predict(Xte))
    results["GBM_test_f1"]  = f1_score(svm_te_lbl, gbm.predict(Xte), average="macro")
    logger.info(f"   ✅ GBM val-F1={val_f1s['gbm']:.4f}  test-F1={results['GBM_test_f1']:.4f}")

    # ── KNN (cosine, k=7) ──────────────────────────────────────────────────
    logger.info("   [KNN] fitting (k=7, cosine)...")
    knn = KNeighborsClassifier(n_neighbors=7, metric="cosine", n_jobs=-1)
    knn.fit(Xtr, svm_tr_lbl)
    val_f1s["knn"]   = f1_score(svm_va_lbl, knn.predict(Xva), average="macro")
    te_probs["knn"]  = knn.predict_proba(Xte)
    results["KNN_val_acc"]  = accuracy_score(svm_va_lbl, knn.predict(Xva))
    results["KNN_val_f1"]   = val_f1s["knn"]
    results["KNN_test_acc"] = accuracy_score(svm_te_lbl, knn.predict(Xte))
    results["KNN_test_f1"]  = f1_score(svm_te_lbl, knn.predict(Xte), average="macro")
    logger.info(f"   ✅ KNN val-F1={val_f1s['knn']:.4f}  test-F1={results['KNN_test_f1']:.4f}")

    # ── Reports ────────────────────────────────────────────────────────────
    label_strs = [id2label[i] for i in range(num_cls)]
    logger.info("\n📊 SKLEARN SUMMARY (utterance-level 768-dim):")
    logger.info(f"   LR   val={results['LR_val_acc']:.4f}/{results['LR_val_f1']:.4f}"
                f"  test={results['LR_test_acc']:.4f}/{results['LR_test_f1']:.4f}")
    logger.info(f"   SVM  val={results['SVM_val_acc']:.4f}/{results['SVM_val_f1']:.4f}"
                f"  test={results['SVM_test_acc']:.4f}/{results['SVM_test_f1']:.4f}")
    logger.info(f"   RF   val={results['RF_val_acc']:.4f}/{results['RF_val_f1']:.4f}"
                f"  test={results['RF_test_acc']:.4f}/{results['RF_test_f1']:.4f}")
    logger.info(f"   GBM  val={results['GBM_val_acc']:.4f}/{results['GBM_val_f1']:.4f}"
                f"  test={results['GBM_test_acc']:.4f}/{results['GBM_test_f1']:.4f}")
    logger.info(f"   KNN  val={results['KNN_val_acc']:.4f}/{results['KNN_val_f1']:.4f}"
                f"  test={results['KNN_test_acc']:.4f}/{results['KNN_test_f1']:.4f}")
    logger.info("\n   SVM Test Report:")
    logger.info(classification_report(svm_te_lbl, svm_tv.predict(Xte),
                                      target_names=label_strs))

    return results, scaler, val_f1s, te_probs


# ===========================================================================
# PHASE 6 — AudioMLP with dedicated SCL projection head
# ===========================================================================
class ResBlock(nn.Module):
    def __init__(self, dim, p):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim),
                                 nn.GELU(), nn.Dropout(p))
    def forward(self, x): return x + self.net(x)


class AudioMLP(nn.Module):
    """
    Phase 6: separate 2-layer projection head for SCL (128-dim).
    Classifier operates on 256-dim features.  Both share the backbone.
    SCL gradients → backprop through scl_head AND shared backbone.
    CE  gradients → backprop through classifier  AND shared backbone.
    Neither loss directly conflicts with the other head.
    """
    def __init__(self, input_dim: int, num_labels: int = 7):
        super().__init__()
        self.stem = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 512),
                                  nn.GELU(), nn.Dropout(0.40))
        self.res1       = ResBlock(512, 0.30)
        self.down       = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 256),
                                        nn.GELU(), nn.Dropout(0.35))
        self.res2       = ResBlock(256, 0.25)
        self.classifier = nn.Linear(256, num_labels)
        # Phase 6 — dedicated SCL head (2-layer MLP → 128-dim hypersphere)
        self.scl_head   = nn.Sequential(
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, 128),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        x      = self.res1(self.stem(x))
        feat   = self.res2(self.down(x))        # [B, 256] shared features
        logits = self.classifier(feat)          # CE path
        proj   = self.scl_head(feat)            # SCL path — 128-dim
        return logits, proj


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=1.5, ls=0.1):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else torch.ones(1))
        self.gamma = gamma; self.ls = ls
    def forward(self, lg, y):
        ce = F.cross_entropy(lg, y, weight=self.weight, label_smoothing=self.ls, reduction="none")
        return ((1 - torch.exp(-ce)) ** self.gamma * ce).mean()


def supcon_loss(proj, labels, temp, device):
    """SCL on 128-dim normalised projection vectors."""
    f    = F.normalize(proj, p=2, dim=1)
    sim  = torch.matmul(f, f.T) / temp
    bs   = labels.size(0)
    I    = torch.eye(bs, device=device)
    pos  = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float() * (1 - I)
    s, _ = sim.max(1, keepdim=True)
    sim  = sim - s.detach()
    lp   = sim - torch.log((torch.exp(sim) * (1 - I)).sum(1, keepdim=True) + 1e-8)
    v    = pos.sum(1) > 0
    if not v.any(): return torch.tensor(0.0, device=device)
    return (-(pos[v] * lp[v]).sum(1) / pos[v].sum(1).clamp(1)).mean()


def mixup(x, y, alpha, device):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def mixup_loss(crit, lg, ya, yb, lam):
    return lam * crit(lg, ya) + (1 - lam) * crit(lg, yb)


# ===========================================================================
# MLP training  (+ SWA)
# ===========================================================================
def train_mlp(tr_emb, tr_lbl, va_emb, va_lbl, id2label, device):
    num_cls = len(id2label)
    dim     = tr_emb.shape[1]
    N       = len(tr_lbl)
    dist    = dict(sorted(collections.Counter(tr_lbl.tolist()).items()))
    logger.info(f"\n   MLP training dist ({N} samples after SMOTE): {dist}")
    if N < 1500:
        logger.warning(f"   ⚠️  Only {N} MLP samples. "
                       "SMOTE may not have triggered. Check config.SMOTE_TARGET.")

    Xtr = torch.tensor(tr_emb, dtype=torch.float32)
    ytr = torch.tensor(tr_lbl, dtype=torch.long)
    Xva = torch.tensor(va_emb, dtype=torch.float32)
    yva = torch.tensor(va_lbl, dtype=torch.long)

    tr_loader = DataLoader(TensorDataset(Xtr, ytr),
                           batch_size=config.BATCH_SIZE, shuffle=True, drop_last=False)
    va_loader = DataLoader(TensorDataset(Xva, yva), batch_size=256, shuffle=False)

    model   = AudioMLP(input_dim=dim, num_labels=num_cls).to(device)
    raw_w   = compute_class_weight("balanced", classes=np.arange(num_cls), y=tr_lbl)
    cw      = torch.tensor(raw_w, dtype=torch.float, device=device)
    crit    = FocalLoss(weight=cw, gamma=config.FOCAL_GAMMA, ls=0.1)
    opt     = torch.optim.AdamW(model.parameters(),
                                lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    sched   = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=config.LEARNING_RATE * 10,
        steps_per_epoch=len(tr_loader), epochs=config.EPOCHS,
        pct_start=0.10, anneal_strategy="cos", div_factor=10.0, final_div_factor=1e3)

    # SWA setup — only activated if training runs long enough
    swa_model    = torch.optim.swa_utils.AveragedModel(model) if config.USE_SWA else None
    swa_start    = int(config.EPOCHS * config.SWA_START_FRAC)
    swa_sched    = torch.optim.swa_utils.SWALR(opt, swa_lr=5e-5) if config.USE_SWA else None
    swa_was_used = False   # ← critical: tracks whether SWA actually ran

    best_f1, patience, best_state = 0.0, 0, None

    logger.info(f"\n⚡ AudioMLP Phase-6 | dim={dim} bs={config.BATCH_SIZE}"
                f" lr={config.LEARNING_RATE} epochs={config.EPOCHS}")
    logger.info(f"   SCL proj-head 256→128 | SWA starts ep={swa_start} | "
                f"SMOTE target={config.SMOTE_TARGET}/class")

    for ep in range(config.EPOCHS):
        model.train()
        sc = sn = 0; sce = ssc = 0.0
        for xb, yb in tr_loader:
            xb, yb    = xb.to(device), yb.to(device)
            use_mix   = config.USE_MIXUP and np.random.random() < config.MIXUP_PROB
            if use_mix:
                xb, ya, yb_, lam = mixup(xb, yb, config.MIXUP_ALPHA, device)
                lg, proj = model(xb)
                ce_ = mixup_loss(crit, lg, ya, yb_, lam)
                sc_ = supcon_loss(proj, ya, config.SCL_TEMP, device) if config.USE_SCL else torch.tensor(0.0)
                p   = lg.argmax(1)
                sc  += ((p==ya).float()*lam + (p==yb_).float()*(1-lam)).sum().item()
            else:
                lg, proj = model(xb)
                ce_ = crit(lg, yb)
                sc_ = supcon_loss(proj, yb, config.SCL_TEMP, device) if config.USE_SCL else torch.tensor(0.0)
                sc  += (lg.argmax(1)==yb).sum().item()
            loss = ce_ + config.SCL_WEIGHT * sc_
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            # Only step OneCycleLR before SWA phase
            if not config.USE_SWA or ep < swa_start:
                sched.step()
            sn += yb.size(0); sce += ce_.item(); ssc += sc_.item()

        # SWA accumulation
        if config.USE_SWA and ep >= swa_start:
            swa_model.update_parameters(model)
            swa_sched.step()
            swa_was_used = True

        # Validation with active model
        model.eval(); vp, vt = [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                vp.extend(model(xb.to(device))[0].argmax(1).cpu().numpy())
                vt.extend(yb.numpy())
        va = accuracy_score(vt, vp); vf = f1_score(vt, vp, average="macro")
        imp = vf > best_f1
        if imp:
            best_f1 = vf; best_state = {k: v.clone() for k, v in model.state_dict().items()}; patience = 0
        else:
            patience += 1
        if imp or (ep+1) % config.LOG_EVERY == 0 or ep == 0:
            tag = "⭐" if imp else f"pat {patience}/{config.MAX_PATIENCE}"
            logger.info(f"  Ep {ep:03d} Tr={sc/sn:.4f} Val={va:.4f}/{vf:.4f}"
                        f" CE={sce/len(tr_loader):.3f} SCL={ssc/len(tr_loader):.3f} {tag}")
        if patience >= config.MAX_PATIENCE:
            logger.info(f"  ⏹ Early stop ep={ep}  best val-F1={best_f1:.4f}"); break

    # Restore best checkpoint
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    # Only use SWA if it actually accumulated parameters
    # If training stopped before SWA_START, swa_model has random init weights → useless
    if config.USE_SWA and swa_was_used:
        logger.info(f"   📦 SWA finalised (accumulated from ep {swa_start})")
        return swa_model, model, best_f1
    else:
        if config.USE_SWA:
            logger.info(f"   ⚠️  SWA skipped — training stopped at ep<{swa_start}, using best checkpoint")
        return None, model, best_f1


# ===========================================================================
# Evaluation
# ===========================================================================
@torch.no_grad()
def evaluate(model_or_swa, base_model, emb_np, lbl_np, device, tta=False):
    """Evaluates SWA model if available, otherwise base model."""
    eval_model = model_or_swa if model_or_swa is not None else base_model
    eval_model.eval()
    passes = config.TTA_PASSES if tta else 1
    noise  = config.TTA_NOISE  if tta else 0.0
    acc_p  = None

    for _ in range(passes):
        x   = torch.tensor(emb_np, dtype=torch.float32)
        if noise > 0: x = x + torch.randn_like(x) * noise
        ldr = DataLoader(TensorDataset(x, torch.tensor(lbl_np)), batch_size=256, shuffle=False)
        pr  = []
        for xb, _ in ldr:
            try:
                lg, _ = eval_model(xb.to(device))
            except Exception:
                # SWA AveragedModel wraps forward differently
                lg, _ = eval_model.module(xb.to(device))
            pr.append(F.softmax(lg, -1).cpu().numpy())
        pr = np.concatenate(pr)
        acc_p = pr if acc_p is None else acc_p + pr

    avg = acc_p / passes
    return lbl_np, avg.argmax(1), avg


# ===========================================================================
# Main
# ===========================================================================
def main():
    set_seed(42)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    logger.info(f"\n🚀 ALL 6 PHASES | Device: {device} | {config.MODEL_NAME}")
    logger.info("   SVM/RF/GBM/KNN: utterance 768-dim clean")
    logger.info(f"  MLP: frame 2304-dim + aug×4 + SMOTE")

    # 1. Load splits
    train_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv")
    val_df   = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv")
    test_df  = pd.read_csv(config.SPLIT_CSV_DIR / "test.csv")

    label_names = sorted(train_df["emotion_final"].unique())
    label2id    = {n: i for i, n in enumerate(label_names)}
    id2label    = {i: n for n, i in label2id.items()}
    logger.info(f"🎭 {len(label2id)} classes: {label2id}")
    logger.info(f"   Train {len(train_df)} | Val {len(val_df)} | Test {len(test_df)}")

    # 2. Dual embeddings (cached)
    cache = config.EMBEDDING_CACHE
    if cache.exists():
        logger.info(f"⚡ Cache: {cache}")
        d = np.load(cache)
        svm_tr  = d["svm_tr"];  svm_trl = d["svm_trl"]
        svm_va  = d["svm_va"];  svm_val = d["svm_val"]
        svm_te  = d["svm_te"];  svm_tel = d["svm_tel"]
        mlp_tr  = d["mlp_tr"];  mlp_trl = d["mlp_trl"]
        mlp_va  = d["mlp_va"];  mlp_val = d["mlp_val"]
        mlp_te  = d["mlp_te"];  mlp_tel = d["mlp_tel"]
    else:
        logger.info("🧠 Loading backbone for dual extraction...")
        from funasr import AutoModel
        bb = AutoModel(model=config.MODEL_NAME, hub="hf", trust_remote_code=True)
        bb.model.eval()
        for p in bb.model.parameters(): p.requires_grad = False

        logger.info("\n📼 [1/6] Utterance-level (SVM/RF/GBM/KNN) — clean, fast...")
        svm_tr, svm_trl = extract_embeddings(bb, train_df, label2id, "svm-train",
                                             granularity="utterance", augment=False, first_call=True)
        svm_va, svm_val = extract_embeddings(bb, val_df,   label2id, "svm-val",
                                             granularity="utterance", augment=False)
        svm_te, svm_tel = extract_embeddings(bb, test_df,  label2id, "svm-test",
                                             granularity="utterance", augment=False)

        logger.info("\n📼 [2/6] Frame-level (MLP) — augmented×4, slower...")
        mlp_tr, mlp_trl = extract_embeddings(bb, train_df, label2id, "mlp-train",
                                             granularity="frame", augment=config.AUGMENT_MLP_TRAIN)
        mlp_va, mlp_val = extract_embeddings(bb, val_df,   label2id, "mlp-val",
                                             granularity="frame", augment=False)
        mlp_te, mlp_tel = extract_embeddings(bb, test_df,  label2id, "mlp-test",
                                             granularity="frame", augment=False)

        np.savez_compressed(cache,
            svm_tr=svm_tr, svm_trl=svm_trl, svm_va=svm_va, svm_val=svm_val,
            svm_te=svm_te, svm_tel=svm_tel,
            mlp_tr=mlp_tr, mlp_trl=mlp_trl, mlp_va=mlp_va, mlp_val=mlp_val,
            mlp_te=mlp_te, mlp_tel=mlp_tel)
        logger.info(f"💾 Dual cache: {cache}")
        del bb
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    logger.info(f"   SVM: train={len(svm_trl)} (dim={svm_tr.shape[1]}) val={len(svm_val)} test={len(svm_tel)}")
    logger.info(f"   MLP: train={len(mlp_trl)} (dim={mlp_tr.shape[1]}) val={len(mlp_val)} test={len(mlp_tel)}")
    logger.info(f"   MLP train dist: {dict(sorted(collections.Counter(mlp_trl.tolist()).items()))}")

    # 3. Phase 5 — SMOTE on MLP training embeddings
    #    Target = max class count (balances all classes to the majority)
    if config.USE_SMOTE:
        cls_counts   = collections.Counter(mlp_trl.tolist())
        smote_target = max(cls_counts.values())   # dynamic: match majority class
        logger.info(f"\n🧬 Phase 5 SMOTE: target={smote_target}/class (= majority count)")
        mlp_tr_before = len(mlp_trl)
        mlp_tr, mlp_trl = simple_smote(mlp_tr, mlp_trl,
                                        target_per_class=smote_target,
                                        k=config.SMOTE_K)
        logger.info(f"   SMOTE: {mlp_tr_before} → {len(mlp_trl)} samples")
        logger.info(f"   Post-SMOTE dist: {dict(sorted(collections.Counter(mlp_trl.tolist()).items()))}")

    # 4. Sklearn classifiers
    sk_res, scaler, sk_val_f1s, sk_te_probs = run_sklearn(
        svm_tr, svm_trl, svm_va, svm_val, svm_te, svm_tel, id2label)

    # 5. MLP
    swa_model, mlp_model, best_val_f1 = train_mlp(
        mlp_tr, mlp_trl, mlp_va, mlp_val, id2label, device)

    # 6. MLP evaluation
    true_te, pred_cl,  prob_cl  = evaluate(swa_model, mlp_model, mlp_te, mlp_tel, device, tta=False)
    _,       pred_tta, prob_tta = evaluate(swa_model, mlp_model, mlp_te, mlp_tel, device, tta=True)
    true_va, pred_va,  _        = evaluate(swa_model, mlp_model, mlp_va, mlp_val, device, tta=False)

    mlp_val_acc  = accuracy_score(true_va, pred_va)
    mlp_test_acc = accuracy_score(true_te, pred_cl)
    mlp_test_f1  = f1_score(true_te, pred_cl,  average="macro")
    tta_test_acc = accuracy_score(true_te, pred_tta)
    tta_test_f1  = f1_score(true_te, pred_tta, average="macro")

    # MLP val-F1 for weighted ensemble
    mlp_val_f1  = f1_score(true_va, pred_va, average="macro")
    sk_val_f1s["mlp"] = mlp_val_f1

    # 7. Phase 4 — val-F1 weighted ensemble (all models)
    # Collect sklearn model test probabilities
    all_probs = dict(sk_te_probs)  # svm, rf, gbm, knn
    all_probs["mlp"] = prob_tta    # MLP-TTA probabilities

    # Weight by val-F1 (softmax normalised)
    names  = list(all_probs.keys())
    vf_arr = np.array([sk_val_f1s.get(n, mlp_val_f1) for n in names])
    weights = np.exp(vf_arr) / np.exp(vf_arr).sum()   # softmax weights
    logger.info(f"\n🎯 Ensemble weights (val-F1 softmax):")
    for n, w, v in zip(names, weights, vf_arr):
        logger.info(f"   {n:6s}: val-F1={v:.4f}  weight={w:.3f}")

    ens_prob = sum(w * all_probs[n] for n, w in zip(names, weights))
    ens_pred = ens_prob.argmax(1)
    ens_acc  = accuracy_score(svm_tel, ens_pred)
    ens_f1   = f1_score(svm_tel, ens_pred, average="macro")
    ens_f1w  = f1_score(svm_tel, ens_pred, average="weighted")

    # sklearn-only ensemble (SVM+RF+GBM+KNN) — no MLP (useful comparison)
    sk_names  = [n for n in names if n != "mlp"]
    sk_wts    = np.array([sk_val_f1s.get(n, 0) for n in sk_names])
    sk_wts    = np.exp(sk_wts) / np.exp(sk_wts).sum()
    sk_ens_p  = sum(w * all_probs[n] for n, w in zip(sk_names, sk_wts))
    sk_ens_pred = sk_ens_p.argmax(1)
    sk_ens_acc  = accuracy_score(svm_tel, sk_ens_pred)
    sk_ens_f1   = f1_score(svm_tel, sk_ens_pred, average="macro")

    # Best sklearn alone
    best_sk_acc = max(sk_res[f"{m}_test_acc"] for m in ["LR","SVM","RF","GBM","KNN"])
    best_sk_f1  = max(sk_res[f"{m}_test_f1"]  for m in ["LR","SVM","RF","GBM","KNN"])

    # 8. Per-class reports
    label_strs = [id2label[i] for i in range(len(id2label))]
    logger.info("\n   MLP (clean) Test Report:")
    logger.info(classification_report(true_te, pred_cl, target_names=label_strs))
    logger.info("   MLP (TTA) Test Report:")
    logger.info(classification_report(true_te, pred_tta, target_names=label_strs))
    logger.info("   Ensemble (all models, val-F1 weighted) Test Report:")
    logger.info(classification_report(svm_tel, ens_pred, target_names=label_strs))

    # 9. Final table
    logger.info("\n" + "=" * 72)
    logger.info("🏆  AUDIO EMOTION — FINAL RESULTS (ALL 6 PHASES)")
    logger.info("=" * 72)
    logger.info(f"{'Method':<35} {'Val-F1':>8} {'Test-Acc':>9} {'Test-F1':>9}")
    logger.info("-" * 72)
    for m in ["LR","SVM","RF","GBM","KNN"]:
        logger.info(f"  {m+' (utt,clean)':<33} {sk_res[f'{m}_val_f1']:>8.4f}"
                    f" {sk_res[f'{m}_test_acc']:>9.4f} {sk_res[f'{m}_test_f1']:>9.4f}")
    logger.info(f"  {'MLP (frame,aug+SMOTE)':<33} {best_val_f1:>8.4f}"
                f" {mlp_test_acc:>9.4f} {mlp_test_f1:>9.4f}")
    logger.info(f"  {'MLP + TTA + SWA':<33} {'—':>8}"
                f" {tta_test_acc:>9.4f} {tta_test_f1:>9.4f}")
    logger.info(f"  {'SK Ensemble (no MLP)':<33} {'—':>8}"
                f" {sk_ens_acc:>9.4f} {sk_ens_f1:>9.4f}")
    logger.info(f"  {'Full Ensemble (all models)':<33} {'—':>8}"
                f" {ens_acc:>9.4f} {ens_f1:>9.4f}")
    logger.info("=" * 72)

    best_acc = max(best_sk_acc, mlp_test_acc, tta_test_acc, ens_acc, sk_ens_acc)
    best_f1_ = max(best_sk_f1,  mlp_test_f1,  tta_test_f1,  ens_f1, sk_ens_f1)
    logger.info(f"  🏅 Best Accuracy  : {best_acc:.4f}")
    logger.info(f"  🏅 Best F1-Macro  : {best_f1_:.4f}")
    logger.info(f"  📊 Full Ens F1-Wt : {ens_f1w:.4f}")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
