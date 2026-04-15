"""Train emotion classifier on emotion2vec embeddings.

Pipeline:
  1. Load emotion2vec+ backbone (FunASR)
  2. Extract utterance embeddings for train/val/test  → cached to disk
  3. Train EmotionMLP with CE + SCL loss for 100 fast epochs
  4. Report final test Accuracy, F1-Macro, F1-Weighted

Run from project root:
  python -m src.audio_baseline.train
"""

import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from transformers import set_seed
from transformers import logging as transformers_logging
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()
logging.getLogger("funasr").setLevel(logging.ERROR)

from src.audio_baseline import config
from src.audio_baseline.model import EmotionMLP
from src.audio_baseline.data import AudioEmotionDataset, EmbeddingDataset
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
# Supervised Contrastive Loss
# ---------------------------------------------------------------------------
def compute_scl_loss(embeddings: torch.Tensor, labels: torch.Tensor,
                     temp: float, device: torch.device) -> torch.Tensor:
    features = F.normalize(embeddings, p=2, dim=1)
    sim = torch.matmul(features, features.T) / temp

    bs = labels.size(0)
    pos_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
    self_mask = torch.ones(bs, bs, device=device) - torch.eye(bs, device=device)
    pos_mask = pos_mask * self_mask

    # Numerically stable log-softmax
    max_sim, _ = sim.max(dim=1, keepdim=True)
    sim = sim - max_sim.detach()
    exp_sim = torch.exp(sim) * self_mask
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
                       split_name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Run emotion2vec generate() on every audio file in df.
    Returns (embeddings [N, 768], labels [N]).
    """
    import librosa

    embeddings, labels_out = [], []
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  Extracting {split_name}"):
        folder   = str(row["folder"]).strip()
        rel_path = str(row["audio_relpath"]).replace("\\", "/").lstrip("/")

        # Resolve path (with one-level subfolder fallback)
        candidate = config.DATA_ROOT / folder / rel_path
        if not candidate.exists():
            found = False
            try:
                for sub in config.DATA_ROOT.iterdir():
                    if sub.is_dir():
                        alt = sub / folder / rel_path
                        if alt.exists():
                            candidate = alt
                            found = True
                            break
            except Exception:
                pass
            if not found:
                skipped += 1
                continue

        try:
            audio, _ = librosa.load(candidate, sr=config.SAMPLING_RATE, mono=True)
            if len(audio) > config.MAX_AUDIO_SAMPLES:
                audio = audio[: config.MAX_AUDIO_SAMPLES]

            # generate() — uses the FULL emotion2vec pipeline (CNN + Transformer)
            # This gives us the proper emotion-aware utterance embeddings
            res = backbone.generate(
                input=audio,
                granularity="utterance",
                extract_embedding=True,
            )

            if not res or "feats" not in res[0]:
                skipped += 1
                continue

            feat = np.array(res[0]["feats"])
            # Handle both [768] and [T, 768] shapes
            if feat.ndim == 2:
                feat = feat.mean(axis=0)
            feat = feat.flatten()

            if feat.shape[0] != 768:
                # Unexpected dim — pad/trim to 768
                if feat.shape[0] > 768:
                    feat = feat[:768]
                else:
                    feat = np.pad(feat, (0, 768 - feat.shape[0]))

            embeddings.append(feat.astype(np.float32))
            labels_out.append(label2id[row["emotion_final"]])

        except Exception:
            skipped += 1
            continue

    logger.info(f"    {split_name}: {len(embeddings)} extracted, {skipped} skipped")
    return np.array(embeddings, dtype=np.float32), np.array(labels_out, dtype=np.int64)


# ---------------------------------------------------------------------------
# Training one epoch
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, criterion, device, scl_temp, scl_weight):
    model.train()
    total_loss = total_ce = total_scl = 0.0
    correct = total = 0

    for emb, labels in loader:
        emb    = emb.to(device)
        labels = labels.to(device)

        logits, embeddings = model(emb)

        ce_loss = criterion(logits, labels)
        if config.USE_SCL and labels.size(0) > 1:
            scl_loss = compute_scl_loss(embeddings, labels, scl_temp, device)
            loss = ce_loss + scl_weight * scl_loss
            total_scl += scl_loss.item()
        else:
            loss = ce_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_ce   += ce_loss.item()
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    n = len(loader)
    return {
        "loss":     total_loss / n,
        "ce_loss":  total_ce   / n,
        "scl_loss": total_scl  / n,
        "accuracy": correct / total if total > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, loader, criterion, device):
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for emb, labels in loader:
            emb    = emb.to(device)
            labels = labels.to(device)
            logits, _ = model(emb)
            total_loss += criterion(logits, labels).item()
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    if not all_logits:
        return {k: 0 for k in ["loss", "accuracy", "f1", "f1_macro",
                                "f1_weighted", "precision", "recall",
                                "precision_macro", "recall_macro"]}

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    preds  = np.argmax(logits, axis=1)
    metrics = evaluate_predictions(labels, preds)
    metrics["loss"] = total_loss / len(loader)
    return metrics


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
    label2id = {n: i for i, n in enumerate(label_names)}
    id2label = {i: n for n, i in label2id.items()}
    logger.info(f"🎭 Emotions ({len(label2id)}): {label2id}")

    # 2. Embedding extraction (cached)
    cache = config.EMBEDDING_CACHE
    if cache.exists():
        logger.info(f"⚡ Loading cached embeddings from {cache}...")
        data      = np.load(cache)
        train_emb = data["train_emb"];  train_lbl = data["train_lbl"]
        val_emb   = data["val_emb"];    val_lbl   = data["val_lbl"]
        test_emb  = data["test_emb"];   test_lbl  = data["test_lbl"]
        logger.info(f"   Train: {len(train_emb)}  Val: {len(val_emb)}  Test: {len(test_emb)}")
    else:
        logger.info("🧠 Loading emotion2vec backbone for embedding extraction...")
        from funasr import AutoModel
        logging.getLogger("funasr").setLevel(logging.ERROR)
        backbone = AutoModel(model=config.MODEL_NAME, hub="hf", trust_remote_code=True)
        backbone.model.eval()
        for p in backbone.model.parameters():
            p.requires_grad = False

        logger.info("🔬 Extracting utterance embeddings (runs once, then cached)...")
        train_emb, train_lbl = extract_embeddings(backbone, train_df, label2id, "train")
        val_emb,   val_lbl   = extract_embeddings(backbone, val_df,   label2id, "val")
        test_emb,  test_lbl  = extract_embeddings(backbone, test_df,  label2id, "test")

        np.savez_compressed(cache,
            train_emb=train_emb, train_lbl=train_lbl,
            val_emb=val_emb,     val_lbl=val_lbl,
            test_emb=test_emb,   test_lbl=test_lbl)
        logger.info(f"💾 Embeddings cached to {cache}")

        # Free backbone memory
        del backbone
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 3. Datasets
    train_ds = EmbeddingDataset(train_emb, train_lbl)
    val_ds   = EmbeddingDataset(val_emb,   val_lbl)
    test_ds  = EmbeddingDataset(test_emb,  test_lbl)

    # Class-balanced sampler (fixes imbalance without reweighting loss)
    class_counts = np.bincount(train_lbl)
    sample_weights = 1.0 / class_counts[train_lbl]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, sampler=sampler,   num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)

    # 4. Model
    emb_dim = train_emb.shape[1]   # usually 768
    model   = EmotionMLP(input_dim=emb_dim, num_labels=len(label2id)).to(device)
    logger.info(f"🏗️  EmotionMLP  input_dim={emb_dim}  num_labels={len(label2id)}")

    # Class-weighted CE loss (belt + suspenders with the balanced sampler)
    weights     = compute_class_weight("balanced", classes=np.arange(len(label2id)), y=train_lbl)
    cw          = torch.tensor(weights, dtype=torch.float).to(device)
    criterion   = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)

    optimizer   = torch.optim.AdamW(model.parameters(),
                                    lr=config.LEARNING_RATE,
                                    weight_decay=config.WEIGHT_DECAY)

    # Cosine annealing with warm restarts — prevents plateaus
    scheduler   = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=1e-6)

    # 5. Training loop
    best_f1    = 0.0
    best_ckpt  = config.CHECKPOINT_DIR / "best_mlp.pt"
    logger.info(f"\n⚡ TRAINING EmotionMLP for {config.EPOCHS} epochs...\n")

    for epoch in range(config.EPOCHS):
        train_m = train_epoch(model, train_loader, optimizer, criterion, device,
                              config.SCL_TEMP, config.SCL_WEIGHT)
        val_m   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            f"Ep {epoch:03d} | "
            f"Loss {train_m['loss']:.4f} (CE {train_m['ce_loss']:.4f} SCL {train_m['scl_loss']:.4f}) | "
            f"Train {train_m['accuracy']:.4f} | "
            f"Val {val_m['accuracy']:.4f} | "
            f"F1-Mac {val_m['f1_macro']:.4f} | "
            f"F1-Wt {val_m['f1_weighted']:.4f}"
        )

        if val_m["f1_macro"] > best_f1:
            best_f1 = val_m["f1_macro"]
            torch.save(model.state_dict(), best_ckpt)
            logger.info(f"  🏅 New best F1={best_f1:.4f} — saved")

    # 6. Final test evaluation with best model
    logger.info("\n" + "=" * 55)
    logger.info("🏁  FINAL EVALUATION ON TEST SET")
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        logger.info("  ✅ Loaded best checkpoint")

    test_m = evaluate(model, test_loader, criterion, device)

    logger.info("\n".join([
        "=" * 55,
        "🏆  AUDIO EMOTION CLASSIFICATION — FINAL REPORT",
        "=" * 55,
        f"  Test Accuracy   : {test_m['accuracy']:.4f}  ({test_m['accuracy']*100:.1f}%)",
        f"  Test F1 Macro   : {test_m['f1_macro']:.4f}",
        f"  Test F1 Weighted: {test_m['f1_weighted']:.4f}",
        f"  Best Val F1     : {best_f1:.4f}",
        "=" * 55,
    ]))


if __name__ == "__main__":
    main()
