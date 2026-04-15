"""Train emotion classifier on emotion2vec embeddings.

Pipeline:
  1. Load emotion2vec+ backbone → extract utterance embeddings → cache to disk
  2. Sklearn baseline: LR + SVM with RBF kernel    (fast, strong for small data)
  3. Linear Probe + SCL                             (thesis SCL contribution)
  4. Final report: accuracy, F1-macro, F1-weighted, per-class F1

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
from sklearn.metrics import accuracy_score, f1_score, classification_report

warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()
logging.getLogger("funasr").setLevel(logging.ERROR)

from src.audio_baseline import config
from src.audio_baseline.data import EmbeddingDataset
from src.text_baseline.metrics_utils import evaluate_predictions

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
# Model: Linear Probe with SCL projection head
# ---------------------------------------------------------------------------
class LinearProbe(nn.Module):
    """
    Lightweight classifier for small datasets.
    Shared encoder → 256-dim projection (used for SCL) → 7 classes.
    """
    def __init__(self, input_dim: int = 768, num_labels: int = 7):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(0.5),     # Heavy dropout for small dataset
        )
        self.classifier = nn.Linear(256, num_labels)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        proj   = self.encoder(x)             # [B, 256] — used for SCL
        logits = self.classifier(proj)       # [B, num_labels]
        return logits, proj


# ---------------------------------------------------------------------------
# Supervised Contrastive Loss
# ---------------------------------------------------------------------------
def scl_loss(embeddings: torch.Tensor, labels: torch.Tensor,
             temp: float, device: torch.device) -> torch.Tensor:
    features = F.normalize(embeddings, p=2, dim=1)
    sim = torch.matmul(features, features.T) / temp

    bs = labels.size(0)
    pos_mask  = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
    self_mask = torch.ones(bs, bs, device=device) - torch.eye(bs, device=device)
    pos_mask  = pos_mask * self_mask

    max_sim, _ = sim.max(dim=1, keepdim=True)
    sim = sim - max_sim.detach()
    exp_sim  = torch.exp(sim) * self_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    valid = pos_mask.sum(1) > 0
    if not valid.any():
        return torch.tensor(0.0, device=device)
    loss = -(pos_mask[valid] * log_prob[valid]).sum(1) / (pos_mask[valid].sum(1) + 1e-8)
    return loss.mean()


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------
def extract_embeddings(backbone, df: pd.DataFrame, label2id: dict,
                       split_name: str, first_call: bool = False):
    import os
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

            if first_call and detected_dim is None:
                detected_dim = feat.shape[0]
                logger.info(f"    🔬 Embedding: shape={feat.shape}, mean={feat.mean():.4f}, std={feat.std():.4f}")

            if detected_dim is None:
                detected_dim = feat.shape[0]

            embeddings.append(feat.astype(np.float32))
            labels_out.append(label2id[row["emotion_final"]])

        except Exception:
            skipped += 1
            continue

    logger.info(f"    {split_name}: {len(embeddings)} extracted, {skipped} skipped, dim={detected_dim}")
    return np.array(embeddings, dtype=np.float32), np.array(labels_out, dtype=np.int64)


# ---------------------------------------------------------------------------
# Sklearn classifiers
# ---------------------------------------------------------------------------
def run_sklearn(train_emb, train_lbl, val_emb, val_lbl, test_emb, test_lbl, id2label):
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train_emb)
    X_val   = scaler.transform(val_emb)
    X_test  = scaler.transform(test_emb)

    results = {}

    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=42)
    lr.fit(X_train, train_lbl)
    for split_name, X, y in [("val", X_val, val_lbl), ("test", X_test, test_lbl)]:
        preds = lr.predict(X)
        results[f"LR_{split_name}_acc"] = accuracy_score(y, preds)
        results[f"LR_{split_name}_f1"]  = f1_score(y, preds, average="macro")

    # 2. SVM (RBF) — best for small high-dim datasets
    svm = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42)
    svm.fit(X_train, train_lbl)
    for split_name, X, y in [("val", X_val, val_lbl), ("test", X_test, test_lbl)]:
        preds = svm.predict(X)
        results[f"SVM_{split_name}_acc"] = accuracy_score(y, preds)
        results[f"SVM_{split_name}_f1"]  = f1_score(y, preds, average="macro")

    # Best model for detailed report
    logger.info("\n📊 SKLEARN RESULTS:")
    logger.info(f"   LR  — Val Acc {results['LR_val_acc']:.4f}  F1-Mac {results['LR_val_f1']:.4f} | "
                f"Test Acc {results['LR_test_acc']:.4f}  F1-Mac {results['LR_test_f1']:.4f}")
    logger.info(f"   SVM — Val Acc {results['SVM_val_acc']:.4f}  F1-Mac {results['SVM_val_f1']:.4f} | "
                f"Test Acc {results['SVM_test_acc']:.4f}  F1-Mac {results['SVM_test_f1']:.4f}")

    # Per-class report for SVM
    svm_test_preds = svm.predict(X_test)
    label_strs = [id2label[i] for i in range(len(id2label))]
    logger.info("\n   SVM Per-class Report (Test):")
    logger.info(classification_report(test_lbl, svm_test_preds, target_names=label_strs))

    return results, scaler, svm


# ---------------------------------------------------------------------------
# Linear Probe training
# ---------------------------------------------------------------------------
def train_linear_probe(train_emb, train_lbl, val_emb, val_lbl, test_emb, test_lbl,
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
                              batch_size=32, shuffle=True, drop_last=False)
    val_loader   = DataLoader(TensorDataset(Xv, yv),  batch_size=64, shuffle=False)
    test_loader  = DataLoader(TensorDataset(Xte, yte), batch_size=64, shuffle=False)

    model = LinearProbe(input_dim=emb_dim, num_labels=num_labels).to(device)

    weights   = compute_class_weight("balanced",
                                     classes=np.arange(num_labels), y=train_lbl)
    cw        = torch.tensor(weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.LEARNING_RATE, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=config.EPOCHS, eta_min=1e-6)

    best_f1   = 0.0
    patience  = 0
    MAX_PATIENCE = 25
    best_state = None

    logger.info(f"\n⚡ Training Linear Probe + SCL ({config.EPOCHS} epochs, early stop after {MAX_PATIENCE})...")

    for epoch in range(config.EPOCHS):
        model.train()
        t_correct = t_total = 0
        t_ce = t_scl = 0.0

        for emb_b, lab_b in train_loader:
            emb_b, lab_b = emb_b.to(device), lab_b.to(device)
            logits, proj = model(emb_b)

            ce  = criterion(logits, lab_b)
            sc  = scl_loss(proj, lab_b, config.SCL_TEMP, device) if config.USE_SCL else torch.tensor(0.0)
            loss = ce + config.SCL_WEIGHT * sc

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            t_correct += (logits.argmax(1) == lab_b).sum().item()
            t_total   += lab_b.size(0)
            t_ce      += ce.item()
            t_scl     += sc.item()

        scheduler.step()

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
                            f"Val Acc {val_acc:.4f} | F1-Mac {val_f1:.4f} ⭐")
        else:
            patience += 1
            if (epoch + 1) % config.LOG_EVERY == 0:
                logger.info(f"  Ep {epoch:03d} | Train Acc {t_correct/t_total:.4f} | "
                            f"Val Acc {val_acc:.4f} | F1-Mac {val_f1:.4f} (patience {patience}/{MAX_PATIENCE})")
            if patience >= MAX_PATIENCE:
                logger.info(f"  ⏹️  Early stopping at Epoch {epoch}")
                break

    # Test evaluation with best model
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for emb_b, lab_b in test_loader:
            logits, _ = model(emb_b.to(device))
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_true.extend(lab_b.numpy())

    test_acc = accuracy_score(all_true, all_preds)
    test_f1  = f1_score(all_true, all_preds, average="macro")
    test_f1w = f1_score(all_true, all_preds, average="weighted")
    label_strs = [id2label[i] for i in range(len(id2label))]
    logger.info("\n   Linear Probe Per-class Report (Test):")
    logger.info(classification_report(all_true, all_preds, target_names=label_strs))

    return test_acc, test_f1, test_f1w, best_f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    set_seed(42)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    logger.info(f"🚀 Device: {device}")

    # 1. Load splits
    logger.info(f"📁 Loading splits from {config.SPLIT_CSV_DIR.name}...")
    train_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv")
    val_df   = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv")
    test_df  = pd.read_csv(config.SPLIT_CSV_DIR / "test.csv")

    label_names = sorted(train_df["emotion_final"].unique())
    label2id    = {n: i for i, n in enumerate(label_names)}
    id2label    = {i: n for n, i in label2id.items()}
    logger.info(f"🎭 Emotions: {label2id}")

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

    logger.info(f"   Train: {len(train_lbl)}  Val: {len(val_lbl)}  Test: {len(test_lbl)}")
    logger.info(f"   Train dist: {dict(collections.Counter(train_lbl.tolist()))}")
    logger.info(f"   Val   dist: {dict(collections.Counter(val_lbl.tolist()))}")

    # 3. Sklearn classifiers (fast, strong for small data)
    sklearn_results, scaler, svm = run_sklearn(
        train_emb, train_lbl, val_emb, val_lbl, test_emb, test_lbl, id2label
    )

    # 4. Linear Probe + SCL (thesis SCL contribution)
    lp_test_acc, lp_test_f1, lp_test_f1w, lp_best_val_f1 = train_linear_probe(
        train_emb, train_lbl, val_emb, val_lbl, test_emb, test_lbl, id2label, device
    )

    # 5. Final Report
    logger.info("\n" + "=" * 60)
    logger.info("🏆  FINAL REPORT — AUDIO EMOTION CLASSIFICATION")
    logger.info("=" * 60)
    logger.info(f"{'Method':<25} {'Val Acc':>9} {'Val F1-Mac':>11} {'Test Acc':>9} {'Test F1-Mac':>12}")
    logger.info("-" * 60)
    logger.info(f"{'LR (baseline)':<25} {sklearn_results['LR_val_acc']:>9.4f} {sklearn_results['LR_val_f1']:>11.4f} "
                f"{sklearn_results['LR_test_acc']:>9.4f} {sklearn_results['LR_test_f1']:>12.4f}")
    logger.info(f"{'SVM-RBF (strong)':<25} {sklearn_results['SVM_val_acc']:>9.4f} {sklearn_results['SVM_val_f1']:>11.4f} "
                f"{sklearn_results['SVM_test_acc']:>9.4f} {sklearn_results['SVM_test_f1']:>12.4f}")
    logger.info(f"{'Linear Probe + SCL':<25} {lp_best_val_f1:>9.4f} {'(best val)':>11} "
                f"{lp_test_acc:>9.4f} {lp_test_f1:>12.4f}")
    logger.info("=" * 60)
    logger.info(f"  Best Test Accuracy  : {max(sklearn_results['SVM_test_acc'], lp_test_acc):.4f}")
    logger.info(f"  Best Test F1-Macro  : {max(sklearn_results['SVM_test_f1'], lp_test_f1):.4f}")
    logger.info(f"  Linear Probe F1-Wt  : {lp_test_f1w:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
