"""Audio emotion classifier — Dual extraction strategy.

KEY INSIGHT FROM EXPERIMENTS:
  All previous regressions traced to one mistake: giving SVM the wrong data.
  SVM needs utterance-level 768-dim clean embeddings (gives 42% historically).
  MLP needs frame-level 2304-dim augmented embeddings (gets richer temporal info).
  Each model now gets exactly what it's best suited for.

Extraction:
  SVM → granularity="utterance" → 768-dim, NO augmentation (clean 648 train samples)
  MLP → granularity="frame", concat(mean,std,max) → 2304-dim, WITH augmentation (~2592 train)

Classifiers:
  SVM-RBF: proven best at utterance-level; small grid (12 fits), trained on train+val
  AudioMLP: Focal + SCL(t=0.2) + MixUp on frame-level augmented data
  Ensemble: average probabilities from both (they operate in independent spaces)

Run: python -m src.audio_baseline.train

COLAB:
  !pip install funasr librosa -q
  !git pull
  # Delete old caches (new filename audio_embeddings_dual.npz so usually not needed)
  import os; [os.remove(p) for p in ["/content/audio_embeddings_dual.npz",
              "/content/audio_embeddings_frame_aug.npz",
              "/content/audio_embeddings.npz"] if os.path.exists(p)]
  !python -m src.audio_baseline.train
  # Extraction: ~15 min SVM (clean) + ~40 min MLP (aug) — one-time, then cached
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
    level=logging.INFO, format="%(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame aggregation (MLP path only)
# ---------------------------------------------------------------------------
def aggregate_frames(feat: np.ndarray) -> np.ndarray:
    """
    [T, D] → concat(mean, std, max) = [3*D]
    [D]    → [3*D] with std=0 (utterance-level fallback kept consistent)
    """
    if feat.ndim == 1:
        return np.concatenate([feat, np.zeros_like(feat), feat])
    return np.concatenate([feat.mean(0), feat.std(0), feat.max(0)])


# ---------------------------------------------------------------------------
# Audio augmentations (training MLP only)
# ---------------------------------------------------------------------------
def get_augmentations(audio: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Returns [(name, waveform)] — always starts with original."""
    import librosa
    out = [("orig", audio.copy())]
    for rate in config.AUG_SPEED_RATES:
        try:
            out.append(("s" + str(rate),
                        librosa.effects.time_stretch(audio.astype(np.float32), rate=rate)))
        except Exception:
            pass
    noise = config.AUG_NOISE_STD * float(np.std(audio) + 1e-8)
    out.append(("noise", (audio + noise * np.random.randn(len(audio))).astype(np.float32)))
    return out


# ---------------------------------------------------------------------------
# Embedding extraction — supports both utterance-level and frame-level
# ---------------------------------------------------------------------------
def extract_embeddings(backbone, df: pd.DataFrame, label2id: dict,
                       split_name: str,
                       granularity: str = "utterance",
                       augment: bool = False,
                       first_call: bool = False):
    """
    granularity="utterance" → 768-dim single vector   (SVM path)
    granularity="frame"     → 2304-dim mean+std+max   (MLP path)
    augment=True            → 4× versions per file    (MLP train only)
    """
    import librosa
    embeddings: list[np.ndarray] = []
    labels_out: list[int]        = []
    skipped = 0
    detected_dim = None

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"  [{granularity[:3]}{'*' if augment else ' '}] {split_name}"):
        folder   = str(row["folder"]).strip()
        rel_path = str(row["audio_relpath"]).replace("\\", "/").lstrip("/")

        candidate = config.DATA_ROOT / folder / rel_path
        if not candidate.exists():
            for sub in (s for s in config.DATA_ROOT.iterdir() if s.is_dir()):
                alt = sub / folder / rel_path
                if alt.exists():
                    candidate = alt; break

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
                with open(os.devnull, "w") as devnull:
                    old_out, old_err = sys.stdout, sys.stderr
                    sys.stdout = sys.stderr = devnull
                    try:
                        res = backbone.generate(
                            input=wav, granularity=granularity,
                            extract_embedding=True)
                    finally:
                        sys.stdout, sys.stderr = old_out, old_err

                if not res or "feats" not in res[0]:
                    continue

                raw  = np.array(res[0]["feats"], dtype=np.float32)
                feat = aggregate_frames(raw) if granularity == "frame" else raw.flatten()

                if detected_dim is None:
                    detected_dim = feat.shape[0]
                    if first_call:
                        logger.info(f"    🔬 gran={granularity}  raw={raw.shape}"
                                    f"  → dim={detected_dim}")

                embeddings.append(feat)
                labels_out.append(label)
            except Exception:
                continue

    n_files = len(df) - skipped
    logger.info(f"    {split_name}: {n_files} files → {len(embeddings)} embeddings"
                f"  ({skipped} skipped)  dim={detected_dim}")
    return np.array(embeddings, dtype=np.float32), np.array(labels_out, dtype=np.int64)


# ---------------------------------------------------------------------------
# SVM  (utterance-level 768-dim, clean original data, small grid)
# ---------------------------------------------------------------------------
def run_sklearn(svm_train_emb, svm_train_lbl,
                svm_val_emb,   svm_val_lbl,
                svm_test_emb,  svm_test_lbl, id2label):
    """
    StandardScaler + SVM-RBF with a small grid (12 fits → ~2-4 min).
    Final SVM trained on train+val for best test generalisation.
    """
    scaler = StandardScaler()
    Xtr    = scaler.fit_transform(svm_train_emb)
    Xva    = scaler.transform(svm_val_emb)
    Xte    = scaler.transform(svm_test_emb)
    Xtv    = scaler.transform(np.concatenate([svm_train_emb, svm_val_emb]))
    ytv    = np.concatenate([svm_train_lbl, svm_val_lbl])

    results = {}

    # LR reference
    logger.info("   [LR] fitting...")
    lr = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced",
                            solver="lbfgs", random_state=42)
    lr.fit(Xtr, svm_train_lbl)
    for sn, X, y in [("val", Xva, svm_val_lbl), ("test", Xte, svm_test_lbl)]:
        p = lr.predict(X)
        results[f"LR_{sn}_acc"] = accuracy_score(y, p)
        results[f"LR_{sn}_f1"]  = f1_score(y, p, average="macro")

    # SVM grid  — 4 C × 1 gamma × 3 folds = 12 fits
    logger.info("   [SVM] grid-searching C ∈ {1,10,100,500}, gamma=scale (12 fits)...")
    gs = GridSearchCV(
        SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
        {"C": [1, 10, 100, 500], "gamma": ["scale"]},
        cv=3, scoring="f1_macro", n_jobs=-1, verbose=0,
    )
    gs.fit(Xtr, svm_train_lbl)
    best = gs.best_params_
    logger.info(f"   ✅ Best C={best['C']}  cv-F1={gs.best_score_:.4f}")

    svm_tv = SVC(kernel="rbf", probability=True, class_weight="balanced",
                 random_state=42, **best)
    svm_tv.fit(Xtv, ytv)

    for sn, X, y, m in [("val",  Xva, svm_val_lbl,  gs.best_estimator_),
                         ("test", Xte, svm_test_lbl, svm_tv)]:
        p = m.predict(X)
        results[f"SVM_{sn}_acc"] = accuracy_score(y, p)
        results[f"SVM_{sn}_f1"]  = f1_score(y, p, average="macro")

    logger.info(f"\n📊 SKLEARN (utterance-level 768-dim):")
    logger.info(f"   LR  val={results['LR_val_acc']:.4f}/{results['LR_val_f1']:.4f}"
                f"  test={results['LR_test_acc']:.4f}/{results['LR_test_f1']:.4f}")
    logger.info(f"   SVM val={results['SVM_val_acc']:.4f}/{results['SVM_val_f1']:.4f}"
                f"  test={results['SVM_test_acc']:.4f}/{results['SVM_test_f1']:.4f}")

    label_strs = [id2label[i] for i in range(len(id2label))]
    logger.info("\n   SVM Test Report:")
    logger.info(classification_report(svm_test_lbl, svm_tv.predict(Xte),
                                      target_names=label_strs))

    return results, scaler, svm_tv


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim),
                                 nn.GELU(), nn.Dropout(dropout))
    def forward(self, x): return x + self.net(x)


class AudioMLP(nn.Module):
    """
    Input dim auto-detected (2304 for frame-level, 768 for utterance fallback).
    Heavy dropout — N_train < 5000.
    """
    def __init__(self, input_dim: int, num_labels: int = 7):
        super().__init__()
        self.stem = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 512),
                                  nn.GELU(), nn.Dropout(0.40))
        self.res1 = ResBlock(512, 0.30)
        self.down = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 256),
                                  nn.GELU(), nn.Dropout(0.35))
        self.res2 = ResBlock(256, 0.25)
        self.classifier = nn.Linear(256, num_labels)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.res1(self.stem(x))
        p = self.res2(self.down(x))
        return self.classifier(p), p


# ---------------------------------------------------------------------------
# Losses + augmentation
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=1.5, label_smoothing=0.1):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else torch.ones(1))
        self.gamma = gamma; self.ls = label_smoothing
    def forward(self, lg, y):
        ce = F.cross_entropy(lg, y, weight=self.weight,
                             label_smoothing=self.ls, reduction="none")
        return ((1 - torch.exp(-ce)) ** self.gamma * ce).mean()


def supcon_loss(proj, labels, temp, device):
    f   = F.normalize(proj, p=2, dim=1)
    sim = torch.matmul(f, f.T) / temp
    bs  = labels.size(0)
    I   = torch.eye(bs, device=device)
    pos = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float() * (1 - I)
    s,_ = sim.max(1, keepdim=True)
    sim = sim - s.detach()
    lp  = sim - torch.log((torch.exp(sim) * (1 - I)).sum(1, keepdim=True) + 1e-8)
    v   = pos.sum(1) > 0
    if not v.any(): return torch.tensor(0.0, device=device)
    return (-(pos[v] * lp[v]).sum(1) / pos[v].sum(1).clamp(1)).mean()


def mixup(x, y, alpha, device):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def mixup_loss(crit, lg, ya, yb, lam):
    return lam * crit(lg, ya) + (1 - lam) * crit(lg, yb)


# ---------------------------------------------------------------------------
# MLP training  (frame-level 2304-dim augmented data)
# ---------------------------------------------------------------------------
def train_mlp(mlp_train_emb, mlp_train_lbl,
              mlp_val_emb,   mlp_val_lbl, id2label, device):
    num_labels = len(id2label)
    emb_dim    = mlp_train_emb.shape[1]
    N          = len(mlp_train_lbl)
    dist       = dict(sorted(collections.Counter(mlp_train_lbl.tolist()).items()))
    logger.info(f"\n   MLP training dist ({N} samples, incl. augmentation): {dist}")
    if N < 1000:
        logger.warning(f"   ⚠️  Only {N} MLP training samples. "
                       "Delete cache and re-run with AUGMENT_MLP_TRAIN=True.")

    Xtr = torch.tensor(mlp_train_emb, dtype=torch.float32)
    ytr = torch.tensor(mlp_train_lbl, dtype=torch.long)
    Xva = torch.tensor(mlp_val_emb,   dtype=torch.float32)
    yva = torch.tensor(mlp_val_lbl,   dtype=torch.long)

    tr_loader = DataLoader(TensorDataset(Xtr, ytr),
                           batch_size=config.BATCH_SIZE, shuffle=True, drop_last=False)
    va_loader = DataLoader(TensorDataset(Xva, yva), batch_size=256, shuffle=False)

    model  = AudioMLP(input_dim=emb_dim, num_labels=num_labels).to(device)
    raw_w  = compute_class_weight("balanced", classes=np.arange(num_labels), y=mlp_train_lbl)
    cw     = torch.tensor(raw_w, dtype=torch.float, device=device)
    crit   = FocalLoss(weight=cw, gamma=config.FOCAL_GAMMA, label_smoothing=0.1)

    opt   = torch.optim.AdamW(model.parameters(),
                              lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=config.LEARNING_RATE * 10,
        steps_per_epoch=len(tr_loader), epochs=config.EPOCHS,
        pct_start=0.10, anneal_strategy="cos", div_factor=10.0, final_div_factor=1e3)

    best_f1, patience, best_state = 0.0, 0, None
    logger.info(f"\n⚡ AudioMLP + Focal + SCL(t={config.SCL_TEMP}) + MixUp")
    logger.info(f"   dim={emb_dim}  bs={config.BATCH_SIZE}  lr={config.LEARNING_RATE}")
    logger.info(f"   epochs={config.EPOCHS}  patience={config.MAX_PATIENCE}")

    for ep in range(config.EPOCHS):
        model.train()
        sc = sn = 0; sce = ssc = 0.0

        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            use_mix = config.USE_MIXUP and np.random.random() < config.MIXUP_PROB
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
            opt.step(); sched.step()
            sn += yb.size(0); sce += ce_.item(); ssc += sc_.item()

        # Validation
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
            tag = "⭐" if imp else f"patience {patience}/{config.MAX_PATIENCE}"
            logger.info(f"  Ep {ep:03d} | Tr {sc/sn:.4f} | Val {va:.4f}/{vf:.4f} | "
                        f"CE={sce/len(tr_loader):.3f} SCL={ssc/len(tr_loader):.3f}  {tag}")
        if patience >= config.MAX_PATIENCE:
            logger.info(f"  ⏹ Early stop ep={ep}  best val-F1={best_f1:.4f}"); break

    if best_state: model.load_state_dict(best_state)
    model.eval()
    return model, best_f1


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, emb_np, lbl_np, device, tta=False):
    model.eval()
    passes = config.TTA_PASSES if tta else 1
    noise  = config.TTA_NOISE  if tta else 0.0
    acc_p  = None
    for _ in range(passes):
        x = torch.tensor(emb_np, dtype=torch.float32)
        if noise > 0: x = x + torch.randn_like(x) * noise
        ldr = DataLoader(TensorDataset(x, torch.tensor(lbl_np)), batch_size=256, shuffle=False)
        pr  = []
        for xb, _ in ldr:
            lg, _ = model(xb.to(device)); pr.append(F.softmax(lg, -1).cpu().numpy())
        pr = np.concatenate(pr)
        acc_p = pr if acc_p is None else acc_p + pr
    avg = acc_p / passes
    return lbl_np, avg.argmax(1), avg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    set_seed(42)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    logger.info(f"\n🚀 Device: {device}  |  Backbone: {config.MODEL_NAME}")
    logger.info(f"   SVM: uttereance-level 768-dim, CLEAN data only")
    logger.info(f"   MLP: frame-level 2304-dim, {'AUGMENTED' if config.AUGMENT_MLP_TRAIN else 'clean'} data")

    # 1. Load splits
    train_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv")
    val_df   = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv")
    test_df  = pd.read_csv(config.SPLIT_CSV_DIR / "test.csv")

    label_names = sorted(train_df["emotion_final"].unique())
    label2id    = {n: i for i, n in enumerate(label_names)}
    id2label    = {i: n for n, i in label2id.items()}
    logger.info(f"🎭 {len(label2id)} classes: {label2id}")
    logger.info(f"   Train {len(train_df)} | Val {len(val_df)} | Test {len(test_df)}")

    # 2. Embeddings (dual cache)
    cache = config.EMBEDDING_CACHE
    if cache.exists():
        logger.info(f"⚡ Cache: {cache}")
        d = np.load(cache)
        svm_train_emb = d["svm_train_emb"]; svm_train_lbl = d["svm_train_lbl"]
        svm_val_emb   = d["svm_val_emb"];   svm_val_lbl   = d["svm_val_lbl"]
        svm_test_emb  = d["svm_test_emb"];  svm_test_lbl  = d["svm_test_lbl"]
        mlp_train_emb = d["mlp_train_emb"]; mlp_train_lbl = d["mlp_train_lbl"]
        mlp_val_emb   = d["mlp_val_emb"];   mlp_val_lbl   = d["mlp_val_lbl"]
        mlp_test_emb  = d["mlp_test_emb"];  mlp_test_lbl  = d["mlp_test_lbl"]
        logger.info(f"   SVM — Train {len(svm_train_lbl)} (dim {svm_train_emb.shape[1]})"
                    f" | Val {len(svm_val_lbl)} | Test {len(svm_test_lbl)}")
        logger.info(f"   MLP — Train {len(mlp_train_lbl)} (dim {mlp_train_emb.shape[1]})"
                    f" | Val {len(mlp_val_lbl)} | Test {len(mlp_test_lbl)}")
    else:
        logger.info("🧠 Loading backbone for dual extraction...")
        from funasr import AutoModel
        bb = AutoModel(model=config.MODEL_NAME, hub="hf", trust_remote_code=True)
        bb.model.eval()
        for p in bb.model.parameters(): p.requires_grad = False

        logger.info("\n📼 Step 1/6: SVM utterance-level embedding extraction (clean, fast)...")
        svm_train_emb, svm_train_lbl = extract_embeddings(
            bb, train_df, label2id, "svm-train",
            granularity="utterance", augment=False, first_call=True)
        svm_val_emb,   svm_val_lbl   = extract_embeddings(
            bb, val_df,   label2id, "svm-val",   granularity="utterance", augment=False)
        svm_test_emb,  svm_test_lbl  = extract_embeddings(
            bb, test_df,  label2id, "svm-test",  granularity="utterance", augment=False)

        logger.info("\n📼 Step 2/6: MLP frame-level embedding extraction (with augmentation)...")
        mlp_train_emb, mlp_train_lbl = extract_embeddings(
            bb, train_df, label2id, "mlp-train",
            granularity="frame", augment=config.AUGMENT_MLP_TRAIN)
        mlp_val_emb,   mlp_val_lbl   = extract_embeddings(
            bb, val_df,   label2id, "mlp-val",   granularity="frame", augment=False)
        mlp_test_emb,  mlp_test_lbl  = extract_embeddings(
            bb, test_df,  label2id, "mlp-test",  granularity="frame", augment=False)

        np.savez_compressed(cache,
            svm_train_emb=svm_train_emb, svm_train_lbl=svm_train_lbl,
            svm_val_emb=svm_val_emb,     svm_val_lbl=svm_val_lbl,
            svm_test_emb=svm_test_emb,   svm_test_lbl=svm_test_lbl,
            mlp_train_emb=mlp_train_emb, mlp_train_lbl=mlp_train_lbl,
            mlp_val_emb=mlp_val_emb,     mlp_val_lbl=mlp_val_lbl,
            mlp_test_emb=mlp_test_emb,   mlp_test_lbl=mlp_test_lbl,
        )
        logger.info(f"💾 Dual cache saved: {cache}")
        del bb
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # 3. SVM on utterance-level clean data
    sk_res, svm_scaler, svm_model = run_sklearn(
        svm_train_emb, svm_train_lbl,
        svm_val_emb,   svm_val_lbl,
        svm_test_emb,  svm_test_lbl, id2label)

    # 4. MLP on frame-level augmented data
    mlp_model, best_val_f1 = train_mlp(
        mlp_train_emb, mlp_train_lbl,
        mlp_val_emb,   mlp_val_lbl, id2label, device)

    # 5. Evaluate MLP
    true_te, pred_clean, prob_clean = evaluate(mlp_model, mlp_test_emb, mlp_test_lbl, device, tta=False)
    _,       pred_tta,   prob_tta   = evaluate(mlp_model, mlp_test_emb, mlp_test_lbl, device, tta=True)
    true_va, pred_va,    _          = evaluate(mlp_model, mlp_val_emb,  mlp_val_lbl,  device, tta=False)

    mlp_test_acc = accuracy_score(true_te, pred_clean)
    mlp_test_f1  = f1_score(true_te, pred_clean, average="macro")
    tta_test_acc = accuracy_score(true_te, pred_tta)
    tta_test_f1  = f1_score(true_te, pred_tta,   average="macro")
    mlp_val_acc  = accuracy_score(true_va, pred_va)

    # 6. Ensemble  —  SVM probs (768-dim space) + MLP-TTA probs (2304-dim space)
    #    They operate independently and both output [N_test, 7] probabilities.
    svm_prob_te = svm_model.predict_proba(svm_scaler.transform(svm_test_emb))
    ens_prob    = 0.5 * svm_prob_te + 0.5 * prob_tta
    ens_pred    = ens_prob.argmax(1)
    ens_acc     = accuracy_score(svm_test_lbl, ens_pred)    # use SVM labels (should match MLP)
    ens_f1      = f1_score(svm_test_lbl, ens_pred, average="macro")
    ens_f1w     = f1_score(svm_test_lbl, ens_pred, average="weighted")

    # 7. Reports
    label_strs = [id2label[i] for i in range(len(id2label))]
    logger.info("\n   MLP (clean) Test Report:")
    logger.info(classification_report(true_te, pred_clean, target_names=label_strs))
    logger.info("   MLP (TTA) Test Report:")
    logger.info(classification_report(true_te, pred_tta, target_names=label_strs))
    logger.info("   Ensemble (SVM-uttutterance + MLP-TTA-frame) Test Report:")
    logger.info(classification_report(svm_test_lbl, ens_pred, target_names=label_strs))

    logger.info("\n" + "=" * 70)
    logger.info("🏆  AUDIO EMOTION CLASSIFICATION — FINAL RESULTS")
    logger.info("=" * 70)
    logger.info(f"{'Method':<32} {'Val-Acc':>8} {'Val-F1':>8} {'Test-Acc':>9} {'Test-F1':>9}")
    logger.info("-" * 70)
    logger.info(f"{'LR (ref, utterance)':<32} {sk_res['LR_val_acc']:>8.4f} {sk_res['LR_val_f1']:>8.4f} "
                f"{sk_res['LR_test_acc']:>9.4f} {sk_res['LR_test_f1']:>9.4f}")
    logger.info(f"{'SVM-RBF (utt, T+V fit)':<32} {sk_res['SVM_val_acc']:>8.4f} {sk_res['SVM_val_f1']:>8.4f} "
                f"{sk_res['SVM_test_acc']:>9.4f} {sk_res['SVM_test_f1']:>9.4f}")
    logger.info(f"{'MLP (frame, aug)':<32} {mlp_val_acc:>8.4f} {best_val_f1:>8.4f} "
                f"{mlp_test_acc:>9.4f} {mlp_test_f1:>9.4f}")
    logger.info(f"{'MLP + TTA':<32} {'—':>8} {'—':>8} "
                f"{tta_test_acc:>9.4f} {tta_test_f1:>9.4f}")
    logger.info(f"{'Ensemble (SVM-utt + MLP-frame)':<32} {'—':>8} {'—':>8} "
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
