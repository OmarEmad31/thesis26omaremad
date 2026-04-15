"""Train emotion classifier on emotion2vec embeddings.

Pipeline:
  1. Load emotion2vec+ backbone → extract utterance embeddings → cache to disk
  2. SVM-RBF with GridSearchCV (C, gamma)          — strong embedding classifier
  3. Deep Residual MLP + SCL + MixUp               — thesis SCL contribution
  4. Ensemble: SVM + MLP (probability averaging)   — best of both worlds
  5. Final report: accuracy, F1-macro, F1-weighted, per-class F1

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
# Model: Deep Residual MLP
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """Pre-norm residual block: LayerNorm → Linear → GELU → Dropout → Add."""
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class DeepMLP(nn.Module):
    """
    Deep residual MLP classifier on top of frozen emotion2vec embeddings.

    Architecture (input_dim → 512 → 512-res → 256 → 256-res → num_labels):
      [B, input_dim]
        → LayerNorm → Linear → GELU → Dropout(0.3)     proj: [B, 512]
        → ResidualBlock(512, 0.2)
        → LayerNorm → Linear → GELU → Dropout(0.2)     down: [B, 256]
        → ResidualBlock(256, 0.15)                      ← SCL projection
        → Linear(256 → num_labels)
    """
    def __init__(self, input_dim: int = 768, num_labels: int = 7):
        super().__init__()

        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.res1 = ResidualBlock(512, 0.2)

        self.down = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.res2 = ResidualBlock(256, 0.15)

        self.classifier = nn.Linear(256, num_labels)

        # Weight initialisation
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """
        x: [B, input_dim]   pre-computed emotion2vec embeddings
        Returns: logits [B, num_labels],  proj [B, 256] (for SCL)
        """
        x    = self.proj(x)        # [B, 512]
        x    = self.res1(x)        # [B, 512]
        x    = self.down(x)        # [B, 256]
        proj = self.res2(x)        # [B, 256]  ← SCL hook
        logits = self.classifier(proj)         # [B, num_labels]
        return logits, proj


# ---------------------------------------------------------------------------
# Supervised Contrastive Loss  (SupCon; Khosla et al., 2020)
# ---------------------------------------------------------------------------
def scl_loss(embeddings: torch.Tensor, labels: torch.Tensor,
             temp: float, device: torch.device) -> torch.Tensor:
    """
    Compute SupCon loss on L2-normalised projection vectors.
    Uses numerical-stability trick (subtract row max before exp).
    """
    features  = F.normalize(embeddings, p=2, dim=1)
    sim       = torch.matmul(features, features.T) / temp

    bs        = labels.size(0)
    pos_mask  = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
    self_mask = torch.ones(bs, bs, device=device) - torch.eye(bs, device=device)
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
# MixUp augmentation (embedding space)
# ---------------------------------------------------------------------------
def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float, device: torch.device):
    """
    Mix pairs of embedding vectors and their labels.
    Returns mixed_x, y_a, y_b, lam — use mixup_criterion for backprop.
    """
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx   = torch.randperm(x.size(0), device=device)
    mixed = lam * x + (1.0 - lam) * x[idx]
    return mixed, y, y[idx], lam


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
                    logger.info(f"    🔬 Embedding dim={feat.shape}, mean={feat.mean():.4f}, std={feat.std():.4f}")

            embeddings.append(feat.astype(np.float32))
            labels_out.append(label2id[row["emotion_final"]])

        except Exception:
            skipped += 1
            continue

    logger.info(f"    {split_name}: {len(embeddings)} extracted, {skipped} skipped, dim={detected_dim}")
    return np.array(embeddings, dtype=np.float32), np.array(labels_out, dtype=np.int64)


# ---------------------------------------------------------------------------
# Sklearn: SVM with GridSearchCV
# ---------------------------------------------------------------------------
def run_sklearn(train_emb, train_lbl, val_emb, val_lbl, test_emb, test_lbl, id2label):
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train_emb)
    X_val   = scaler.transform(val_emb)
    X_test  = scaler.transform(test_emb)

    # Combine train+val for final SVM fitting (more data = better)
    X_tv    = np.concatenate([X_train, X_val], axis=0)
    y_tv    = np.concatenate([train_lbl, val_lbl], axis=0)

    results = {}

    # 1. Logistic Regression (fast reference)
    logger.info("   Fitting Logistic Regression...")
    lr = LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs",
                            multi_class="multinomial", random_state=42)
    lr.fit(X_train, train_lbl)
    for split_name, X, y in [("val", X_val, val_lbl), ("test", X_test, test_lbl)]:
        preds = lr.predict(X)
        results[f"LR_{split_name}_acc"] = accuracy_score(y, preds)
        results[f"LR_{split_name}_f1"]  = f1_score(y, preds, average="macro")

    # 2. SVM-RBF with Grid Search
    logger.info("   Grid-searching SVM hyperparameters (C, gamma) — may take a few minutes...")
    param_grid = {
        "C":     [1, 5, 10, 50, 100, 200],
        "gamma": ["scale", "auto"],
    }
    grid_svm = GridSearchCV(
        SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced"),
        param_grid,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=0,
    )
    grid_svm.fit(X_train, train_lbl)
    best_C     = grid_svm.best_params_["C"]
    best_gamma = grid_svm.best_params_["gamma"]
    logger.info(f"   ✅ Best SVM params: C={best_C}, gamma={best_gamma}  (CV F1-mac={grid_svm.best_score_:.4f})")

    # Re-fit best SVM on train+val for best test generalisation
    svm_final = SVC(kernel="rbf", C=best_C, gamma=best_gamma,
                    probability=True, random_state=42, class_weight="balanced")
    svm_final.fit(X_tv, y_tv)

    # Evaluate the grid-search winner on val (from train-only model) and test
    svm_val_model = grid_svm.best_estimator_
    for split_name, X, y in [("val", X_val, val_lbl), ("test", X_test, test_lbl)]:
        model = svm_val_model if split_name == "val" else svm_final
        preds = model.predict(X)
        results[f"SVM_{split_name}_acc"] = accuracy_score(y, preds)
        results[f"SVM_{split_name}_f1"]  = f1_score(y, preds, average="macro")

    logger.info("\n📊 SKLEARN RESULTS:")
    logger.info(f"   LR  — Val Acc {results['LR_val_acc']:.4f}  F1-Mac {results['LR_val_f1']:.4f} | "
                f"Test Acc {results['LR_test_acc']:.4f}  F1-Mac {results['LR_test_f1']:.4f}")
    logger.info(f"   SVM — Val Acc {results['SVM_val_acc']:.4f}  F1-Mac {results['SVM_val_f1']:.4f} | "
                f"Test Acc {results['SVM_test_acc']:.4f}  F1-Mac {results['SVM_test_f1']:.4f}")

    # Per-class report for SVM on test
    svm_test_preds = svm_final.predict(X_test)
    label_strs = [id2label[i] for i in range(len(id2label))]
    logger.info("\n   SVM Per-class Report (Test):")
    logger.info(classification_report(test_lbl, svm_test_preds, target_names=label_strs))

    return results, scaler, svm_final, svm_val_model


# ---------------------------------------------------------------------------
# Deep MLP training with SCL + MixUp
# ---------------------------------------------------------------------------
def train_deep_mlp(train_emb, train_lbl, val_emb, val_lbl, test_emb, test_lbl,
                   id2label, device):
    num_labels = len(id2label)
    emb_dim    = train_emb.shape[1]

    # Tensors
    Xtr = torch.tensor(train_emb, dtype=torch.float32)
    ytr = torch.tensor(train_lbl, dtype=torch.long)
    Xv  = torch.tensor(val_emb,   dtype=torch.float32)
    yv  = torch.tensor(val_lbl,   dtype=torch.long)
    Xte = torch.tensor(test_emb,  dtype=torch.float32)
    yte = torch.tensor(test_lbl,  dtype=torch.long)

    train_loader = DataLoader(TensorDataset(Xtr, ytr),
                              batch_size=config.BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader   = DataLoader(TensorDataset(Xv, yv),   batch_size=256, shuffle=False)
    test_loader  = DataLoader(TensorDataset(Xte, yte), batch_size=256, shuffle=False)

    model = DeepMLP(input_dim=emb_dim, num_labels=num_labels).to(device)

    # Class-weighted CE with label smoothing
    weights   = compute_class_weight("balanced",
                                     classes=np.arange(num_labels), y=train_lbl)
    cw        = torch.tensor(weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # OneCycleLR: warm-up then cosine anneal — typically 3-5% better than cosine alone
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE * 5,   # peak LR = 5× base
        steps_per_epoch=steps_per_epoch,
        epochs=config.EPOCHS,
        pct_start=0.1,                     # 10% warmup
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1e4,
    )

    best_f1    = 0.0
    patience   = 0
    MAX_PAT    = 40                         # more patience for deeper model
    best_state = None

    logger.info(f"\n⚡ Training Deep Residual MLP + SCL + MixUp "
                f"({config.EPOCHS} epochs, early stop after {MAX_PAT})...")
    logger.info(f"   dim={emb_dim}  batch={config.BATCH_SIZE}  lr={config.LEARNING_RATE}"
                f"  scl_w={config.SCL_WEIGHT}  scl_t={config.SCL_TEMP}"
                f"  mixup={'on' if config.USE_MIXUP else 'off'}")

    for epoch in range(config.EPOCHS):
        model.train()
        t_correct = t_total = 0
        t_ce = t_scl = 0.0

        for emb_b, lab_b in train_loader:
            emb_b, lab_b = emb_b.to(device), lab_b.to(device)

            if config.USE_MIXUP and np.random.random() < 0.5:
                # MixUp on 50% of batches so model still sees clean data
                emb_b, y_a, y_b, lam = mixup_batch(emb_b, lab_b, config.MIXUP_ALPHA, device)
                logits, proj = model(emb_b)
                ce  = mixup_criterion(criterion, logits, y_a, y_b, lam)
                # SCL always runs on the ORIGINAL labels for proper positives
                sc  = scl_loss(proj, y_a, config.SCL_TEMP, device) if config.USE_SCL else torch.tensor(0.0)
                # Accuracy proxy uses majority label
                preds = logits.argmax(1)
                t_correct += ((preds == y_a).float() * lam + (preds == y_b).float() * (1 - lam)).sum().item()
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
            scheduler.step()

            t_total += lab_b.size(0)
            t_ce    += ce.item()
            t_scl   += sc.item()

        # Validation
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for emb_b, lab_b in val_loader:
                logits, _ = model(emb_b.to(device))
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_true.extend(lab_b.numpy())

        val_acc = accuracy_score(all_true, all_preds)
        val_f1  = f1_score(all_true, all_preds, average="macro")

        if val_f1 > best_f1:
            best_f1    = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience   = 0
            if (epoch + 1) % config.LOG_EVERY == 0 or epoch == 0:
                logger.info(f"  Ep {epoch:03d} | Train Acc {t_correct/t_total:.4f} | "
                            f"Val Acc {val_acc:.4f} | F1-Mac {val_f1:.4f} ⭐  "
                            f"[CE={t_ce/len(train_loader):.3f} SCL={t_scl/len(train_loader):.3f}]")
        else:
            patience += 1
            if (epoch + 1) % config.LOG_EVERY == 0:
                logger.info(f"  Ep {epoch:03d} | Train Acc {t_correct/t_total:.4f} | "
                            f"Val Acc {val_acc:.4f} | F1-Mac {val_f1:.4f} "
                            f"(patience {patience}/{MAX_PAT})")
            if patience >= MAX_PAT:
                logger.info(f"  ⏹️  Early stopping at Epoch {epoch}")
                break

    # Load best checkpoint
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    return model, best_f1


# ---------------------------------------------------------------------------
# Evaluate MLP on a loader  →  (acc, f1_macro, f1_weighted, preds, probs)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_mlp(model, loader, device):
    all_preds, all_true, all_probs = [], [], []
    for emb_b, lab_b in loader:
        logits, _ = model(emb_b.to(device))
        probs      = torch.softmax(logits, dim=-1)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_true.extend(lab_b.numpy())
    return (
        np.array(all_true),
        np.array(all_preds),
        np.array(all_probs),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    set_seed(42)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    logger.info(f"🚀 Device: {device}")
    logger.info(f"🧠 Backbone: {config.MODEL_NAME}")

    # 1. Load splits
    logger.info(f"📁 Loading splits from {config.SPLIT_CSV_DIR.name}...")
    train_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv")
    val_df   = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv")
    test_df  = pd.read_csv(config.SPLIT_CSV_DIR / "test.csv")

    label_names = sorted(train_df["emotion_final"].unique())
    label2id    = {n: i for i, n in enumerate(label_names)}
    id2label    = {i: n for n, i in label2id.items()}
    logger.info(f"🎭 Emotions: {label2id}")
    logger.info(f"   Train {len(train_df)} | Val {len(val_df)} | Test {len(test_df)}")

    # 2. Embedding extraction (cached)
    cache = config.EMBEDDING_CACHE
    if cache.exists():
        logger.info(f"⚡ Loading cached embeddings from {cache}...")
        data      = np.load(cache)
        train_emb = data["train_emb"];  train_lbl = data["train_lbl"]
        val_emb   = data["val_emb"];    val_lbl   = data["val_lbl"]
        test_emb  = data["test_emb"];   test_lbl  = data["test_lbl"]
    else:
        logger.info("🧠 Loading emotion2vec backbone for embedding extraction...")
        from funasr import AutoModel
        backbone = AutoModel(model=config.MODEL_NAME, hub="hf", trust_remote_code=True)
        backbone.model.eval()
        for p in backbone.model.parameters():
            p.requires_grad = False

        logger.info("🔬 Extracting utterance embeddings (runs once, then cached)...")
        train_emb, train_lbl = extract_embeddings(backbone, train_df, label2id, "train", first_call=True)
        val_emb,   val_lbl   = extract_embeddings(backbone, val_df,   label2id, "val")
        test_emb,  test_lbl  = extract_embeddings(backbone, test_df,  label2id, "test")

        np.savez_compressed(cache,
            train_emb=train_emb, train_lbl=train_lbl,
            val_emb=val_emb,     val_lbl=val_lbl,
            test_emb=test_emb,   test_lbl=test_lbl)
        logger.info(f"💾 Embeddings cached to {cache}")
        del backbone
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    logger.info(f"   Embedding dim: {train_emb.shape[1]}")
    logger.info(f"   Train: {len(train_lbl)}  Val: {len(val_lbl)}  Test: {len(test_lbl)}")
    logger.info(f"   Train dist: {dict(collections.Counter(train_lbl.tolist()))}")

    # 3. SVM with Grid Search
    sklearn_results, scaler, svm_tv, svm_val = run_sklearn(
        train_emb, train_lbl, val_emb, val_lbl, test_emb, test_lbl, id2label
    )

    # 4. Deep MLP + SCL + MixUp
    mlp_model, mlp_best_val_f1 = train_deep_mlp(
        train_emb, train_lbl, val_emb, val_lbl, test_emb, test_lbl, id2label, device
    )

    # 5. Evaluate MLP alone
    Xte  = torch.tensor(test_emb,  dtype=torch.float32)
    yte  = torch.tensor(test_lbl,  dtype=torch.long)
    Xv   = torch.tensor(val_emb,   dtype=torch.float32)
    yv   = torch.tensor(val_lbl,   dtype=torch.long)

    test_loader = DataLoader(TensorDataset(Xte, yte), batch_size=256, shuffle=False)
    val_loader  = DataLoader(TensorDataset(Xv,  yv),  batch_size=256, shuffle=False)

    true_te, preds_mlp_te, probs_mlp_te = evaluate_mlp(mlp_model, test_loader, device)
    true_v,  preds_mlp_v,  probs_mlp_v  = evaluate_mlp(mlp_model, val_loader,  device)

    mlp_test_acc = accuracy_score(true_te, preds_mlp_te)
    mlp_test_f1  = f1_score(true_te, preds_mlp_te, average="macro")
    mlp_test_f1w = f1_score(true_te, preds_mlp_te, average="weighted")
    mlp_val_acc  = accuracy_score(true_v,  preds_mlp_v)

    # 6. Ensemble: SVM probs + MLP probs (equal weight)
    scaler_tv = StandardScaler()
    scaler_tv.fit(np.concatenate([train_emb, val_emb], axis=0))
    svm_probs_te = svm_tv.predict_proba(scaler_tv.transform(test_emb))
    ens_probs_te = 0.5 * svm_probs_te + 0.5 * probs_mlp_te
    ens_preds_te = ens_probs_te.argmax(axis=1)
    ens_acc  = accuracy_score(test_lbl, ens_preds_te)
    ens_f1   = f1_score(test_lbl, ens_preds_te, average="macro")
    ens_f1w  = f1_score(test_lbl, ens_preds_te, average="weighted")

    # 7. Final report
    label_strs = [id2label[i] for i in range(len(id2label))]

    logger.info("\n   Deep MLP Per-class Report (Test):")
    logger.info(classification_report(true_te, preds_mlp_te, target_names=label_strs))

    logger.info("\n   Ensemble (SVM + MLP) Per-class Report (Test):")
    logger.info(classification_report(test_lbl, ens_preds_te, target_names=label_strs))

    logger.info("\n" + "=" * 65)
    logger.info("🏆  FINAL REPORT — AUDIO EMOTION CLASSIFICATION")
    logger.info("=" * 65)
    logger.info(f"{'Method':<28} {'Val Acc':>9} {'Val F1-Mac':>11} {'Test Acc':>9} {'Test F1-Mac':>12}")
    logger.info("-" * 65)
    logger.info(f"{'LR (reference)':<28} {sklearn_results['LR_val_acc']:>9.4f} {sklearn_results['LR_val_f1']:>11.4f} "
                f"{sklearn_results['LR_test_acc']:>9.4f} {sklearn_results['LR_test_f1']:>12.4f}")
    logger.info(f"{'SVM-RBF (grid search)':<28} {sklearn_results['SVM_val_acc']:>9.4f} {sklearn_results['SVM_val_f1']:>11.4f} "
                f"{sklearn_results['SVM_test_acc']:>9.4f} {sklearn_results['SVM_test_f1']:>12.4f}")
    logger.info(f"{'Deep MLP + SCL + MixUp':<28} {mlp_val_acc:>9.4f} {mlp_best_val_f1:>11.4f} "
                f"{mlp_test_acc:>9.4f} {mlp_test_f1:>12.4f}")
    logger.info(f"{'Ensemble (SVM + MLP)':<28} {'—':>9} {'—':>11} "
                f"{ens_acc:>9.4f} {ens_f1:>12.4f}")
    logger.info("=" * 65)
    logger.info(f"  Best Test Accuracy  : {max(sklearn_results['SVM_test_acc'], mlp_test_acc, ens_acc):.4f}")
    logger.info(f"  Best Test F1-Macro  : {max(sklearn_results['SVM_test_f1'], mlp_test_f1,  ens_f1):.4f}")
    logger.info(f"  Ensemble F1-Weighted: {ens_f1w:.4f}")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
