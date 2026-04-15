"""Train emotion2vec+ for Audio Emotion Classification.
Run from project root: python -m src.audio_baseline.train
"""

import json
import random
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

# 🤫 SUPER-SILENCE MODE (100% CLEAN LOGS)
warnings.filterwarnings("ignore")
logging.getLogger('funasr').setLevel(logging.ERROR)

# Project imports
from src.audio_baseline import config
from src.audio_baseline.model import Emotion2VecBaseline
from src.audio_baseline.data import AudioEmotionDataset, collate_audio_fn
from src.text_baseline.metrics_utils import compute_metrics, evaluate_predictions

# 📄 PERSISTENT LOGGER SETUP
# Now that config is imported, we can safely use it
log_file = config.CHECKPOINT_DIR.parent / "audio_training_log.txt"
if not log_file.parent.exists(): log_file.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

from tqdm import tqdm
from transformers import set_seed, Wav2Vec2FeatureExtractor
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from src.audio_baseline.model import Emotion2VecBaseline
from src.audio_baseline.data import AudioEmotionDataset, collate_audio_fn
from src.text_baseline.metrics_utils import compute_metrics, evaluate_predictions

# ---------------------------------------------------------------------------
# SCL Trainer for Audio
# ---------------------------------------------------------------------------

class AudioSCLTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, device, class_weights, scl_temp=0.1, scl_weight=0.1):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.class_weights = class_weights.to(device)
        self.scl_temp = scl_temp
        self.scl_weight = scl_weight
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=0.15)

    def compute_scl_loss(self, embeddings, labels):
        """Supervised Contrastive Loss implementation."""
        features = F.normalize(embeddings, p=2, dim=1)
        similarity = torch.matmul(features, features.T) / self.scl_temp
        
        # Mask for matching labels (positives)
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        
        # Mask out self-contrast
        batch_size = labels.size(0)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=self.device)
        mask = mask * logits_mask
        
        # Numerical stability
        max_sim, _ = torch.max(similarity, dim=1, keepdim=True)
        sim_stable = similarity - max_sim.detach()
        
        exp_sim = torch.exp(sim_stable) * logits_mask
        log_prob = sim_stable - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        
        valid_anchors = mask.sum(1) > 0
        if valid_anchors.any():
            mean_log_prob_pos = (mask[valid_anchors] * log_prob[valid_anchors]).sum(1) / (mask[valid_anchors].sum(1) + 1e-8)
            return -mean_log_prob_pos.mean()
        return torch.tensor(0.0, device=self.device)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_ce = 0
        total_scl = 0
        correct = 0
        total = 0
        
        # 📈 Progress Tracking
        processed_files = 0
        skipped_files = 0
        
        # Deep Fine-Tuning: Unfreeze at scheduled epoch
        if epoch == config.UNFREEZE_EPOCH:
            self.model.set_backbone_trainable(True)
            # Re-build optimizer to include newly unfrozen backbone params
            params = get_audio_optimizer_params(self.model, config.LEARNING_RATE, 0.95)
            self.optimizer = torch.optim.AdamW(params)
            logger.info("🔥 [DEEP FINE-TUNING] Backbone Unfrozen. Optimizer Reset.")

        pbar = tqdm(self.train_loader, desc=f"  Epoch {epoch} [Training]", leave=False)
        for batch in pbar:
            if batch is None:
                skipped_files += config.BATCH_SIZE # Approximation
                continue
                
            inputs = batch["input_values"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Count success/skips based on batch items
            current_batch_size = labels.size(0)
            processed_files += current_batch_size
            skipped_files += (config.BATCH_SIZE - current_batch_size)
            
            self.optimizer.zero_grad()
            
            # Forward: [batch, num_classes], [batch, 768]
            logits, embeddings = self.model(inputs, attention_mask=mask)
            
            # 1. Cross Entropy Loss
            ce_loss = self.criterion(logits, labels)
            
            # 2. SCL Loss
            if config.USE_SCL:
                scl_loss = self.compute_scl_loss(embeddings, labels)
                loss = ce_loss + (config.SCL_WEIGHT * scl_loss)
                total_scl += scl_loss.item()
            else:
                loss = ce_loss
                
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            total_ce += ce_loss.item()
            
            # Metrics
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                "loss": f"{loss.item():.3f}", 
                "acc": f"{correct/total:.2f}"
            })
            
        return {
            "loss": total_loss / len(self.train_loader) if total > 0 else 0,
            "ce_loss": total_ce / len(self.train_loader) if total > 0 else 0,
            "scl_loss": total_scl / len(self.train_loader) if total > 0 else 0,
            "accuracy": correct / total if total > 0 else 0,
            "processed": processed_files,
            "skipped": skipped_files
        }

    def evaluate(self, loader=None):
        if loader is None: loader = self.val_loader
        self.model.eval()
        all_logits = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in loader:
                if batch is None: continue
                
                inputs = batch["input_values"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                logits, _ = self.model(inputs, attention_mask=mask)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        if not all_logits:
            logger.warning("⚠️ Evaluation skipped: No valid audio files found in validation set.")
            return {"loss": 0, "accuracy": 0, "f1": 0, "precision": 0, "recall": 0}
            
        logits = np.concatenate(all_logits, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        preds = np.argmax(logits, axis=1)
        
        metrics = evaluate_predictions(labels, preds)
        metrics["loss"] = total_loss / len(loader)
        return metrics

# ---------------------------------------------------------------------------
# Optimizer Helpers
# ---------------------------------------------------------------------------

def get_audio_optimizer_params(model, base_lr, layerwise_lr_decay):
    """Build optimizer parameters. Optimized for ModelScope/Pipeline architecture."""
    # Group parameters: Pipeline (Backbone) vs Head
    params = []
    
    # 1. Identify Head vs Backbone parameters
    # We look for anything that isn't the 'feature_extractor' (ModelScope backbone)
    head_params = []
    backbone_params = []
    
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "feature_extractor" in n or "backbone" in n:
            backbone_params.append(p)
        else:
            head_params.append(p)
            
    # 2. Add to optimizer groups
    if head_params:
        params.append({
            "params": head_params,
            "lr": base_lr,
            "weight_decay": 0.01
        })
        
    if backbone_params:
        # We give the backbone a slightly lower learning rate to preserve its pre-trained knowledge
        params.append({
            "params": backbone_params,
            "lr": base_lr * 0.1, 
            "weight_decay": 0.01
        })
        
    # If we couldn't find specific groups, just return everything
    if not params:
        params = [{'params': [p for p in model.parameters() if p.requires_grad], 'lr': base_lr}]
        
    return params

# ---------------------------------------------------------------------------
# Main Execution: POWER MODE
# ---------------------------------------------------------------------------

def main():
    set_seed(42)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"🚀 [POWER MODE] Device: {device}")

    # 1. Load Data Splits (The FULL 5,700 Dataset)
    print(f"📁 Loading Full Dataset Splits from {config.SPLIT_CSV_DIR.name}...")
    train_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv")
    val_df = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv")
    test_df = pd.read_csv(config.SPLIT_CSV_DIR / "test.csv")
    
    # 2. Prepare Label Mapping
    label_names = sorted(train_df["emotion_final"].unique())
    label2id = {name: i for i, name in enumerate(label_names)}
    print(f"🎭 Emotions: {label2id}")

    # 3. Data Loaders
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    
    train_ds = AudioEmotionDataset(None, config.DATA_ROOT, feature_extractor, label2id, config.MAX_AUDIO_SAMPLES)
    train_ds.df = train_df
    
    val_ds = AudioEmotionDataset(None, config.DATA_ROOT, feature_extractor, label2id, config.MAX_AUDIO_SAMPLES)
    val_ds.df = val_df
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_audio_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_audio_fn, num_workers=2)
    
    # 4. Initialize Model & Training Tools
    # --- DIAGNOSTIC: Check physical file existence ---
    logger.info(f"🔍 Diagnostic: Checking first few paths in {config.DATA_ROOT}...")
    for idx, row in train_df.head(3).iterrows():
        folder = str(row["folder"]).strip()
        rel_path = str(row["audio_relpath"]).replace("\\", "/").lstrip("/")
        full_p = config.DATA_ROOT / folder / rel_path
        exists = full_p.exists()
        logger.info(f"   - {'✅' if exists else '❌'} {folder}/{rel_path}")
        if not exists:
            # Check for common zip-folder patterns
            if config.DATA_ROOT.exists():
                for sub in config.DATA_ROOT.iterdir():
                    if sub.is_dir() and (sub / folder / rel_path).exists():
                        logger.info(f"     💡 FOUND IN SUBFOLDER: {sub.name}")
                        break

    model = Emotion2VecBaseline(config.MODEL_NAME, num_labels=len(label2id))
    
    # Start with frozen backbone (Initial Warmup)
    model.set_backbone_trainable(False)
    
    weights = compute_class_weight("balanced", classes=np.arange(len(label2id)), y=train_df["emotion_final"].map(label2id))
    class_weights = torch.tensor(weights, dtype=torch.float)
    
    params = get_audio_optimizer_params(model, config.LEARNING_RATE, 0.95)
    optimizer = torch.optim.AdamW(params)
    
    num_steps = len(train_loader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*num_steps), num_training_steps=num_steps)
    
    trainer = AudioSCLTrainer(model, train_loader, val_loader, optimizer, scheduler, device, class_weights)
    
    # 5. Training Loop
    best_val_f1 = 0
    print(f"\n⚡ STARTING POWER MODE TRAINING ({config.EPOCHS} Epochs)...")
    
    for epoch in range(config.EPOCHS):
        # Train
        train_m = trainer.train_epoch(epoch)
        
        # Log Summary with File Counters
        logger.info(f"✨ Epoch {epoch} Training Summary:")
        logger.info(f"   - Loss: {train_m['loss']:.4f} (CE: {train_m['ce_loss']:.4f}, SCL: {train_m['scl_loss']:.4f})")
        logger.info(f"   - Accuracy: {train_m['accuracy']:.4f}")
        logger.info(f"   - 📁 Files: {train_m['processed']} Processed, {train_m['skipped']} Skipped")
        
        # Valid
        val_m = trainer.evaluate()
        # LOGGING
        msg = (
            f"Epoch {epoch} | "
            f"Loss: {train_m['loss']:.3f} (CE: {train_m['ce_loss']:.3f}, SCL: {train_m['scl_loss']:.3f}) | "
            f"Train Acc: {train_m['accuracy']:.3f} | "
            f"Val Acc: {val_m['accuracy']:.3f} | "
            f"F1 Macro: {val_m['f1_macro']:.4f} | "
            f"F1 Weighted: {val_m['f1_weighted']:.4f}"
        )
        logger.info(msg)
        
        # Save Best
        if val_m["f1_macro"] > best_val_f1:
            best_val_f1 = val_m["f1_macro"]
            save_path = config.CHECKPOINT_DIR / "best_power_model.pt"
            torch.save(model.state_dict(), save_path)
            logger.info(f"🏅 New Record! F1={best_val_f1:.4f} -> Saved to Drive.")

    # 6. FINAL TEST
    logger.info("\n" + "🏁" * 20)
    logger.info("FINAL EVALUATION ON TEST SET")
    test_ds = AudioEmotionDataset(None, config.DATA_ROOT, feature_extractor, label2id, config.MAX_AUDIO_SAMPLES)
    test_ds.df = test_df
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_audio_fn)
    
    test_m = trainer.evaluate(test_loader)
    
    final_report = [
        "\n" + "="*40,
        "🏆 POWER MODE FINAL REPORT",
        "="*40,
        f"Test Accuracy: {test_m['accuracy']:.4f}",
        f"Test F1 Macro: {test_m['f1_macro']:.4f}",
        f"Test F1 Weighted: {test_m['f1_weighted']:.4f}",
        "="*40
    ]
    logger.info("\n".join(final_report))

if __name__ == "__main__":
    main()
