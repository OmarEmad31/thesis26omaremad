"""
Training script for Video Swin Transformer Emotion Recognition.
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import classification_report, f1_score, accuracy_score

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from video_swin.dataset import (
    VideoEmotionDataset,
    NUM_CLASSES,
    ID_TO_EMOTION,
    EMOTION_LABELS,
)
from video_swin.model import build_model

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Weighted Random Sampler ────────────────────────────────────────────────────

def make_sampler(dataset: VideoEmotionDataset) -> WeightedRandomSampler:
    """Class-balanced over-sampling for the training set."""
    counts = dataset.class_counts
    total = sum(counts)
    class_weights = [total / (c + 1e-6) for c in counts]
    sample_weights = [class_weights[row["_label"]] for row in dataset.samples]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


# ── Loss function ──────────────────────────────────────────────────────────────

def make_criterion(dataset: VideoEmotionDataset, device: torch.device, label_smoothing: float = 0.1):
    """Compute class-frequency inverse weights for CrossEntropyLoss."""
    counts = np.array(dataset.class_counts, dtype=np.float32)
    weights = 1.0 / (counts + 1.0)
    weights = weights / weights.sum() * NUM_CLASSES
    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)
    logger.info("Class weights: %s", {ID_TO_EMOTION[i]: f"{w:.3f}" for i, w in enumerate(weights)})
    return criterion


# ── Evaluation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    for videos, labels in loader:
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast():
            logits = model(videos)
            loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "predictions": all_preds,
        "labels": all_labels,
    }


# ── Training loop ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, scaler, device, grad_clip: float = 1.0):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (videos, labels) in enumerate(loader):
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            logits = model(videos)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 10 == 0:
            logger.info(
                "  [%3d/%3d]  loss=%.4f  acc=%.3f",
                batch_idx + 1, len(loader),
                total_loss / total, correct / total,
            )

    return total_loss / total, correct / total


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train Video Swin for Emotion Recognition")
    parser.add_argument("--backbone", default="swin_base_patch4_window7_224", type=str)
    parser.add_argument("--num_frames", default=16, type=int, help="Frames to sample per clip")
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--lr", default=5e-5, type=float, help="Peak learning rate")
    parser.add_argument("--min_lr", default=1e-7, type=float)
    parser.add_argument("--warmup_epochs", default=5, type=int)
    parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument("--dropout", default=0.4, type=float)
    parser.add_argument("--label_smoothing", default=0.1, type=float)
    parser.add_argument("--freeze_stages", default=2, type=int, help="Freeze first N backbone stages")
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--num_workers", default=0, type=int, help="DataLoader workers (0=main thread)")
    parser.add_argument("--checkpoint_dir", default=str(PROJECT_ROOT / "checkpoints" / "video_swin"), type=str)
    parser.add_argument("--resume", default=None, type=str, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--train_csv",
        default=str(PROJECT_ROOT / "data" / "processed" / "splits" / "video_eligible" / "train.csv"),
    )
    parser.add_argument(
        "--val_csv",
        default=str(PROJECT_ROOT / "data" / "processed" / "splits" / "video_eligible" / "val.csv"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s (%.1f GB)", torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Datasets ───────────────────────────────────────────────────────────────
    logger.info("Loading datasets …")
    train_ds = VideoEmotionDataset(args.train_csv, num_frames=args.num_frames, train=True)
    val_ds = VideoEmotionDataset(args.val_csv, num_frames=args.num_frames, train=False)

    sampler = make_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    logger.info("Train: %d samples | Val: %d samples", len(train_ds), len(val_ds))

    # ── Model ──────────────────────────────────────────────────────────────────
    logger.info("Building model: %s", args.backbone)
    model = build_model(
        backbone=args.backbone,
        pretrained=True,
        dropout=args.dropout,
        freeze_stages=args.freeze_stages,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parameters: %dM total, %dM trainable", total_params // 1_000_000, trainable_params // 1_000_000)

    # ── Loss + Optimiser ───────────────────────────────────────────────────────
    criterion = make_criterion(train_ds, device, label_smoothing=args.label_smoothing)

    # Discriminative learning rates: backbone gets lower LR
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if "head" in n and p.requires_grad]

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr * 0.1, "weight_decay": args.weight_decay},
            {"params": head_params, "lr": args.lr, "weight_decay": args.weight_decay * 0.1},
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Warmup + cosine annealing
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=args.min_lr,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    scaler = GradScaler()

    # ── Resume ─────────────────────────────────────────────────────────────────
    start_epoch = 1
    best_val_acc = 0.0
    best_val_f1 = 0.0
    history = []

    if args.resume and Path(args.resume).exists():
        logger.info("Resuming from %s", args.resume)
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        best_val_f1 = ckpt.get("best_val_f1", 0.0)
        history = ckpt.get("history", [])
        logger.info("Resumed from epoch %d (best val_acc=%.4f)", start_epoch - 1, best_val_acc)

    # ── Training ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Starting training for %d epochs", args.epochs)
    logger.info("=" * 60)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        logger.info("\n── Epoch %d/%d ──", epoch, args.epochs)

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, args.grad_clip
        )
        scheduler.step()

        val_metrics = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        logger.info(
            "Epoch %d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f val_f1=%.4f | %.1fs",
            epoch,
            train_loss, train_acc,
            val_metrics["loss"], val_metrics["accuracy"],
            val_metrics["f1_macro"],
            elapsed,
        )

        # Checkpoint best model
        is_best = val_metrics["accuracy"] > best_val_acc
        if is_best:
            best_val_acc = val_metrics["accuracy"]
            best_val_f1 = val_metrics["f1_macro"]
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
            logger.info("★ New best val_acc=%.4f  f1_macro=%.4f  → saved!", best_val_acc, best_val_f1)

            # Print detailed report
            label_names = [ID_TO_EMOTION[i] for i in range(NUM_CLASSES)]
            print("\n" + classification_report(
                val_metrics["labels"],
                val_metrics["predictions"],
                target_names=label_names,
                zero_division=0,
            ))

        # Save full checkpoint every epoch
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_acc": best_val_acc,
                "best_val_f1": best_val_f1,
                "history": history,
            },
            ckpt_dir / "last_checkpoint.pt",
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_f1_weighted": val_metrics["f1_weighted"],
        })

        # Save history to JSON
        with open(ckpt_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("Training complete!")
    logger.info("Best val accuracy : %.4f (%.1f%%)", best_val_acc, best_val_acc * 100)
    logger.info("Best val F1-macro : %.4f", best_val_f1)
    logger.info("Best checkpoint   : %s", ckpt_dir / "best_model.pt")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
