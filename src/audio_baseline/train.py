"""Train emotion2vec+ for Audio Emotion Classification.
Run from project root: python -m src.audio_baseline.train
"""

import json
import random
import sys
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from transformers import (
    Wav2Vec2FeatureExtractor,
    get_linear_schedule_with_warmup,
    set_seed
)

# Project specific imports
from src.audio_baseline import config
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

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in tqdm(self.train_loader, desc="  Training", leave=False):
            inputs = batch["input_values"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward
            # Note: We need a model implementation that returns both logits AND embeddings for SCL
            # I will update model.py to return (logits, pooled_output)
            self.optimizer.zero_grad()
            
            # Since our model.py currently only returns logits, I will adjust it 
            # to return embeddings too if needed, or extract them here if we can.
            # For now, let's assume model() returns logits for the basic Baseline.
            logits = self.model(inputs, attention_mask=mask)
            
            loss = self.criterion(logits, labels)
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch["input_values"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                logits = self.model(inputs, attention_mask=mask)
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        logits = np.concatenate(all_logits, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        preds = np.argmax(logits, axis=1)
        
        return evaluate_predictions(labels, preds)

# ---------------------------------------------------------------------------
# Optimizer Helpers
# ---------------------------------------------------------------------------

def get_audio_optimizer_params(model, base_lr, layerwise_lr_decay):
    """Build LLRD for WavLM/emotion2vec architecture."""
    # Group by layers
    # Backbone is accessable via model.backbone
    # Note: emotion2vec follows Wav2Vec2 structure: backbone.feature_extractor, backbone.encoder.layers
    params = []
    
    # 1. Head + Pooling parameters (Full LR)
    head_params = {
        "params": [p for n, p in model.named_parameters() if "backbone" not in n],
        "lr": base_lr,
        "weight_decay": 0.01
    }
    params.append(head_params)
    
    # 2. Backbone Layers with decay
    if hasattr(model.backbone, "encoder"):
        layers = model.backbone.encoder.layers
        num_layers = len(layers)
        for i, layer in enumerate(layers):
            lr = base_lr * (layerwise_lr_decay ** (num_layers - i))
            params.append({
                "params": layer.parameters(),
                "lr": lr,
                "weight_decay": 0.01
            })
            
    # 3. Feature Extractor (Fixed or very low LR)
    if hasattr(model.backbone, "feature_extractor"):
        lr = base_lr * (layerwise_lr_decay ** (num_layers + 1))
        params.append({
            "params": model.backbone.feature_extractor.parameters(),
            "lr": lr,
            "weight_decay": 0.01
        })
        
    return params

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    set_seed(42)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data Splits
    # We use the text project's HC splits to ensure we are training on the exact same clips
    train_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv")
    val_df = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv")
    test_df = pd.read_csv(config.SPLIT_CSV_DIR / "test.csv")
    
    full_df = pd.concat([train_df, val_df]).reset_index(drop=True)
    
    # 2. Prepare Label Mapping
    label_names = sorted(full_df["emotion_final"].unique())
    label2id = {name: i for i, name in enumerate(label_names)}
    print(f"Clasess: {label2id}")

    # 3. K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=config.NUM_FOLDS, shuffle=True, random_state=42)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    
    all_fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(full_df, full_df["emotion_final"])):
        print(f"\n{'='*20} Fold {fold} {'='*20}")
        
        fold_train_df = full_df.iloc[train_idx]
        fold_val_df = full_df.iloc[val_idx]
        
        train_ds = AudioEmotionDataset(
            csv_path=None, # Not used in current mod, we pass DF
            data_root=config.DATA_ROOT,
            feature_extractor=feature_extractor,
            label2id=label2id,
            max_samples=config.MAX_AUDIO_SAMPLES
        )
        # Override the df for each fold
        train_ds.df = fold_train_df
        
        val_ds = AudioEmotionDataset(
            csv_path=None,
            data_root=config.DATA_ROOT,
            feature_extractor=feature_extractor,
            label2id=label2id,
            max_samples=config.MAX_AUDIO_SAMPLES
        )
        val_ds.df = fold_val_df
        
        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_audio_fn)
        val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_audio_fn)
        
        # Initialize Model
        model = Emotion2VecBaseline(config.MODEL_NAME, num_labels=len(label2id))
        
        # Class Weights
        weights = compute_class_weight("balanced", classes=np.arange(len(label2id)), y=fold_train_df["emotion_final"].map(label2id))
        class_weights = torch.tensor(weights, dtype=torch.float)
        
        # Optimizer & Scheduler
        params = get_audio_optimizer_params(model, config.LEARNING_RATE, 0.95)
        optimizer = torch.optim.AdamW(params)
        num_steps = len(train_loader) * config.EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*num_steps), num_training_steps=num_steps)
        
        # Trainer
        trainer = AudioSCLTrainer(model, train_loader, val_loader, optimizer, scheduler, device, class_weights)
        
        best_f1 = 0
        for epoch in range(config.EPOCHS):
            loss = trainer.train_epoch()
            metrics = trainer.evaluate()
            print(f"  Epoch {epoch}: Loss={loss:.4f}, Val F1={metrics['macro_f1']:.4f}, Val Acc={metrics['accuracy']:.4f}")
            
            if metrics["weighted_f1"] > best_f1:
                best_f1 = metrics["weighted_f1"]
                fold_dir = config.CHECKPOINT_DIR / f"fold_{fold}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), fold_dir / "best_model.pt")
        
        all_fold_metrics.append(best_f1)
        
        # Cleanup
        del model, trainer, optimizer, scheduler
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    print(f"\nTraining Complete. Average Val F1: {np.mean(all_fold_metrics):.4f}")

if __name__ == "__main__":
    main()
