"""Train emotion2vec+ for Audio Emotion Classification.
Run from project root: python -m src.audio_baseline.train
"""

import sys
import gc
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import set_seed, get_linear_schedule_with_warmup
from transformers import logging as transformers_logging
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()
logging.getLogger("funasr").setLevel(logging.ERROR)

# Project imports
from src.audio_baseline import config
from src.audio_baseline.model import Emotion2VecBaseline
from src.audio_baseline.data import AudioEmotionDataset, collate_audio_fn
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
def compute_scl_loss(embeddings, labels, temp=0.07, device="cuda"):
    features = F.normalize(embeddings, p=2, dim=1)
    sim = torch.matmul(features, features.T) / temp

    batch_size = labels.size(0)
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(device)
    logits_mask = (torch.ones_like(mask) - torch.eye(batch_size, device=device))
    mask = mask * logits_mask

    max_sim, _ = torch.max(sim, dim=1, keepdim=True)
    sim = sim - max_sim.detach()

    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    valid = mask.sum(1) > 0
    if not valid.any():
        return torch.tensor(0.0, device=device)

    loss = -(mask[valid] * log_prob[valid]).sum(1) / (mask[valid].sum(1) + 1e-8)
    return loss.mean()


# ---------------------------------------------------------------------------
# Optimizer helper
# ---------------------------------------------------------------------------
def build_optimizer(model, lr):
    """Separate backbone (lower LR) from head (full LR)."""
    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(p)
        else:
            head_params.append(p)

    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": lr, "weight_decay": config.WEIGHT_DECAY})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr * 0.1, "weight_decay": config.WEIGHT_DECAY})
    if not param_groups:
        param_groups = [{"params": list(model.parameters()), "lr": lr}]

    return torch.optim.AdamW(param_groups)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class AudioTrainer:
    def __init__(self, model, train_loader, val_loader, device, class_weights):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights, label_smoothing=0.1
        )
        self.optimizer = build_optimizer(model, config.LEARNING_RATE)
        total_steps = (len(train_loader) // config.GRADIENT_ACCUMULATION_STEPS) * config.EPOCHS
        warmup_steps = int(0.1 * total_steps)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    # ------------------------------------------------------------------ train
    def train_epoch(self, epoch):
        # Unfreeze backbone + rebuild optimizer + rebuild scheduler atomically
        if epoch == config.UNFREEZE_EPOCH:
            self.model.set_backbone_trainable(True)
            self.optimizer = build_optimizer(self.model, config.LEARNING_RATE)
            remaining_epochs = config.EPOCHS - epoch
            remaining_steps = (len(self.train_loader) // config.GRADIENT_ACCUMULATION_STEPS) * remaining_epochs
            warmup_steps = int(0.05 * remaining_steps)
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=remaining_steps,
            )
            logger.info("🔥 [DEEP FINE-TUNING] Backbone UNFROZEN. Optimizer & Scheduler rebuilt.")

        self.model.train()
        total_loss = total_ce = total_scl = 0.0
        correct = total = 0
        processed = skipped = 0

        self.optimizer.zero_grad()
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                    desc=f"  Epoch {epoch} [Train]", leave=False)

        for step, batch in pbar:
            if batch is None:
                skipped += config.BATCH_SIZE
                continue

            inputs = batch["input_values"].to(self.device)
            labels = batch["labels"].to(self.device)
            bs = labels.size(0)
            processed += bs
            skipped += config.BATCH_SIZE - bs

            logits, embeddings = self.model(inputs)

            ce_loss = self.criterion(logits, labels)
            if config.USE_SCL and bs > 1:
                scl_loss = compute_scl_loss(embeddings, labels, config.SCL_TEMP, self.device)
                loss = ce_loss + config.SCL_WEIGHT * scl_loss
                total_scl += scl_loss.item()
            else:
                loss = ce_loss

            # Gradient accumulation
            (loss / config.GRADIENT_ACCUMULATION_STEPS).backward()

            if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0 or (step + 1) == len(self.train_loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item()
            total_ce += ce_loss.item()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += bs

            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{correct/total:.3f}")

        n = len(self.train_loader)
        return {
            "loss": total_loss / n,
            "ce_loss": total_ce / n,
            "scl_loss": total_scl / n,
            "accuracy": correct / total if total > 0 else 0,
            "processed": processed,
            "skipped": skipped,
        }

    # ---------------------------------------------------------------- evaluate
    def evaluate(self, loader=None):
        if loader is None:
            loader = self.val_loader
        self.model.eval()
        all_logits, all_labels = [], []
        total_loss = 0.0

        with torch.no_grad():
            for batch in loader:
                if batch is None:
                    continue
                inputs = batch["input_values"].to(self.device)
                labels = batch["labels"].to(self.device)
                logits, _ = self.model(inputs)
                total_loss += self.criterion(logits, labels).item()
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        if not all_logits:
            logger.warning("⚠️  Evaluation skipped — no valid batches.")
            return {k: 0 for k in
                    ["loss", "accuracy", "f1", "f1_macro", "f1_weighted",
                     "precision", "recall", "precision_macro", "recall_macro"]}

        logits = np.concatenate(all_logits)
        labels = np.concatenate(all_labels)
        preds = np.argmax(logits, axis=1)
        metrics = evaluate_predictions(labels, preds)
        metrics["loss"] = total_loss / len(loader)
        return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    set_seed(42)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    # 1. Load splits
    print(f"📁 Loading splits from {config.SPLIT_CSV_DIR.name}...")
    train_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv")
    val_df   = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv")
    test_df  = pd.read_csv(config.SPLIT_CSV_DIR / "test.csv")

    label_names = sorted(train_df["emotion_final"].unique())
    label2id = {n: i for i, n in enumerate(label_names)}
    print(f"🎭 Emotions: {label2id}")

    # 2. Diagnostic — verify first 3 paths exist
    logger.info(f"🔍 Checking paths in DATA_ROOT: {config.DATA_ROOT}")
    for _, row in train_df.head(3).iterrows():
        p = config.DATA_ROOT / str(row["folder"]).strip() / str(row["audio_relpath"]).replace("\\", "/").lstrip("/")
        logger.info(f"  {'✅' if p.exists() else '❌'} {p.name}")

    # 3. Datasets — NO feature extractor needed
    def make_ds(df):
        ds = AudioEmotionDataset(
            csv_path=None,
            data_root=config.DATA_ROOT,
            label2id=label2id,
            max_samples=config.MAX_AUDIO_SAMPLES,
            sampling_rate=config.SAMPLING_RATE,
        )
        ds.df = df
        return ds

    train_ds = make_ds(train_df)
    val_ds   = make_ds(val_df)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              collate_fn=collate_audio_fn, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                              collate_fn=collate_audio_fn, num_workers=2)

    # 4. Model
    model = Emotion2VecBaseline(config.MODEL_NAME, num_labels=len(label2id))
    model.set_backbone_trainable(False)  # Freeze backbone for warmup epochs

    weights = compute_class_weight("balanced",
                                   classes=np.arange(len(label2id)),
                                   y=train_df["emotion_final"].map(label2id))
    class_weights = torch.tensor(weights, dtype=torch.float)

    trainer = AudioTrainer(model, train_loader, val_loader, device, class_weights)

    # 5. Training loop
    best_val_f1 = 0.0
    print(f"\n⚡ STARTING TRAINING ({config.EPOCHS} Epochs)...\n")

    for epoch in range(config.EPOCHS):
        train_m = trainer.train_epoch(epoch)
        val_m   = trainer.evaluate()

        logger.info(
            f"Epoch {epoch:02d} | "
            f"Loss {train_m['loss']:.4f} (CE {train_m['ce_loss']:.4f} SCL {train_m['scl_loss']:.4f}) | "
            f"Train Acc {train_m['accuracy']:.4f} | "
            f"Val Acc {val_m['accuracy']:.4f} | "
            f"F1-Mac {val_m['f1_macro']:.4f} | "
            f"F1-Wt {val_m['f1_weighted']:.4f} | "
            f"Files {train_m['processed']} ok / {train_m['skipped']} skipped"
        )

        if val_m["f1_macro"] > best_val_f1:
            best_val_f1 = val_m["f1_macro"]
            ckpt = config.CHECKPOINT_DIR / "best_model.pt"
            torch.save(model.state_dict(), ckpt)
            logger.info(f"  🏅 New best F1={best_val_f1:.4f} saved → {ckpt}")

    # 6. Final test evaluation with best model
    logger.info("\n" + "=" * 50)
    logger.info("🏁 FINAL EVALUATION ON TEST SET")

    best_ckpt = config.CHECKPOINT_DIR / "best_model.pt"
    if best_ckpt.exists():
        logger.info("  🔄 Loading best checkpoint...")
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    test_ds = make_ds(test_df)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                             collate_fn=collate_audio_fn)
    test_m = trainer.evaluate(test_loader)

    report = [
        "=" * 50,
        "🏆  POWER MODE FINAL REPORT",
        "=" * 50,
        f"  Test Accuracy  : {test_m['accuracy']:.4f}",
        f"  Test F1 Macro  : {test_m['f1_macro']:.4f}",
        f"  Test F1 Weighted: {test_m['f1_weighted']:.4f}",
        "=" * 50,
    ]
    logger.info("\n".join(report))


if __name__ == "__main__":
    main()
