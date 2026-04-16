"""
Pure HuggingFace Trainer Pipeline for Audio Emotion Classification.
Exactly mimics the architecture and flow of the text baseline.
"""

from __future__ import annotations

import json
import random
import sys
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
import librosa
from tqdm import tqdm

from transformers import (
    AutoModelForAudioClassification,
    Wav2Vec2FeatureExtractor,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)
from transformers import logging as hf_log

# Silence HF / Librosa warnings for a clean minimalist terminal
hf_log.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore")

from src.audio_baseline import config


# ==============================================================================
# LOSS / SCL LOGIC (Borrowed exactly from Text Baseline)
# ==============================================================================
class SCLTrainer(Trainer):
    """Trainer with Supervised Contrastive Learning (SCL) and CrossEntropy."""

    def __init__(self, *args, class_weights: torch.Tensor, scl_temp: float = 0.1, scl_weight: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.scl_temp = scl_temp
        self.scl_weight = scl_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        
        # Ensure model doesn't see labels so it doesn't calculate its own standard loss inside.
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        model_inputs["output_hidden_states"] = True
        outputs = model(**model_inputs)
        
        logits = outputs.logits
        
        # 1. Standard Weighted Cross-Entropy Loss
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        ce_loss = loss_fn(logits, labels)
        
        # 2. Supervised Contrastive Loss (SCL)
        # Wav2Vec doesn't use a CLS token; we mean-pool the sequence instead
        hidden_seq = outputs.hidden_states[-1] 
        hidden = hidden_seq.mean(dim=1)
        
        # L2 Normalize embeddings
        features = F.normalize(hidden, p=2, dim=1)
        
        # Compute Cosine Similarity Matrix
        similarity = torch.matmul(features, features.T) / self.scl_temp
        
        # Create mask for matching labels
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        
        # Zero out the diagonal (a sample shouldn't contrast with itself)
        batch_size = labels.size(0)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=mask.device)
        mask = mask * logits_mask
        
        # Numerical stability for logsumexp
        max_sim, _ = torch.max(similarity, dim=1, keepdim=True)
        sim_stable = similarity - max_sim.detach()
        
        # Denominator only looks at other elements
        exp_sim = torch.exp(sim_stable) * logits_mask
        log_prob = sim_stable - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        
        # Compute mean log-likelihood over positive samples
        valid_anchors = mask.sum(1) > 0  
        
        if valid_anchors.any():
            mean_log_prob_pos = (mask[valid_anchors] * log_prob[valid_anchors]).sum(1) / (mask[valid_anchors].sum(1) + 1e-8)
            scl_loss = -mean_log_prob_pos.mean()
        else:
            scl_loss = torch.tensor(0.0, device=ce_loss.device)
            
        # 3. Hybrid Loss
        loss = ce_loss + (self.scl_weight * scl_loss)
        
        clean_outputs = type(outputs)(loss=loss, logits=logits)
        return (loss, clean_outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1}


# ==============================================================================
# DATASET BUILDER
# ==============================================================================
def load_audio_features(df, label2id, feature_extractor):
    print(f"Loading {len(df)} audio files into memory...")
    features, labels = [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        fldr = str(row["folder"]).strip()
        rel = str(row["audio_relpath"]).replace("\\", "/").lstrip("/")
        audio_path = config.DATA_ROOT / fldr / rel
        
        if not audio_path.exists(): continue
        try:
            audio, _ = librosa.load(audio_path, sr=config.SAMPLING_RATE, mono=True)
            audio = audio[:config.MAX_AUDIO_SAMPLES].astype(np.float32)
            lbl = label2id[row["emotion_final"]]
            features.append(audio)
            labels.append(lbl)
        except Exception:
            continue
            
    # Process instantly in memory
    inputs = feature_extractor(features, sampling_rate=config.SAMPLING_RATE, padding=True, 
                               max_length=config.MAX_AUDIO_SAMPLES, truncation=True, return_tensors="pt")
    
    # Store directly into HF dataset
    return Dataset.from_dict({
        "input_values": inputs.input_values,
        "attention_mask": inputs.attention_mask,
        "labels": labels
    })

# ==============================================================================
# MAIN RUNNER
# ==============================================================================
def main() -> None:
    set_seed(config.SEED)
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    # 1. Load CSV Splits
    train_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv")
    val_df   = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv")
    test_df  = pd.read_csv(config.SPLIT_CSV_DIR / "test.csv")

    label2id = {n: i for i, n in enumerate(sorted(train_df["emotion_final"].unique()))}
    id2label = {i: n for n, i in label2id.items()}
    num_labels = len(label2id)

    # 2. Combine exactly like text_baseline for Stratified K-Fold
    all_df = pd.concat([train_df, val_df], ignore_index=True)
    all_labels = np.array([label2id[lbl] for lbl in all_df["emotion_final"]])
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    
    # 3. Load feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config.MODEL_NAME)

    print(f"\n🚀 Minimalist End-to-End Audio Training")
    print(f"Model: {config.MODEL_NAME} | Epochs: {config.NUM_EPOCHS}")
    print(f"======================================================================")

    for fold_idx, (train_index, val_index) in enumerate(skf.split(all_df, all_labels)):
        fold_out_dir = config.CHECKPOINT_DIR / f"fold_{fold_idx}"
        best_dir = fold_out_dir / "best_model"
        
        if (best_dir / "model.safetensors").exists() or (best_dir / "pytorch_model.bin").exists():
            print(f"[RESUME] Skipping Fold {fold_idx} as it is already complete.")
            continue

        print(f"\n{'='*20} TRAINING FOLD {fold_idx} {'='*20}")
        
        fold_train_df = all_df.iloc[train_index]
        fold_val_df   = all_df.iloc[val_index]

        # Calculate balanced weights
        raw_weights = compute_class_weight(
            class_weight="balanced", classes=np.arange(num_labels),
            y=np.array([label2id[lbl] for lbl in fold_train_df["emotion_final"]])
        )
        class_weights_tensor = torch.tensor(raw_weights, dtype=torch.float)

        print(f"Building Dataset for Fold {fold_idx}...")
        train_ds = load_audio_features(fold_train_df, label2id, feature_extractor)
        val_ds   = load_audio_features(fold_val_df, label2id, feature_extractor)

        fold_out_dir.mkdir(parents=True, exist_ok=True)
        best_dir.mkdir(parents=True, exist_ok=True)

        model = AutoModelForAudioClassification.from_pretrained(
            config.MODEL_NAME, num_labels=num_labels, id2label=id2label, label2id=label2id,
            ignore_mismatched_sizes=True 
        )
        
        # 🧊 METHOD B: ARABIC XLSR (FROZEN BACKBONE)
        # Because we swapped out the generic English model for the Arabic XLSR, it natively understands 
        # the phonetics and rhythmic cadence of the Arabic audio. Thus, we can safely and drastically
        # speed up training by fully freezing the 300 Million parameters and only training the new Head.
        model.freeze_feature_encoder()
        for param in model.wav2vec2.parameters():
            param.requires_grad = False
            
        # Ensure only the new PyTorch classification head layers update
        model.projector.weight.requires_grad = True
        model.classifier.weight.requires_grad = True
        model.classifier.bias.requires_grad = True

        training_args = TrainingArguments(
            output_dir=str(fold_out_dir),
            learning_rate=config.LEARNING_RATE,
            per_device_train_batch_size=config.BATCH_SIZE,
            per_device_eval_batch_size=config.BATCH_SIZE,
            gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
            num_train_epochs=config.NUM_EPOCHS,
            weight_decay=config.WEIGHT_DECAY,
            warmup_ratio=config.WARMUP_RATIO,
            eval_strategy="epoch",  
            save_strategy="epoch",  # EarlyStopping requires saving aligned with eval
            load_best_model_at_end=True, # Protect the peak weights!
            metric_for_best_model="f1_macro",
            save_total_limit=1,
            logging_steps=5,     
            seed=config.SEED + fold_idx,
            report_to="none",
            disable_tqdm=False, # Explicitly guarantee visual progress bars
            dataloader_pin_memory=False,
        )

        trainer = SCLTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=feature_extractor,
            compute_metrics=compute_metrics,
            class_weights=class_weights_tensor,
            scl_temp=config.SCL_TEMP,
            scl_weight=config.SCL_WEIGHT,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=config.EARLY_STOP_PATIENCE)]
        )

        print(f"🔥 Kicking off Trainer loop for Fold {fold_idx} (Watch for the progress bar...)")
        trainer.train()
        # Save explicitly at end ensures the `best_model` is what remains in the directory
        trainer.save_model(str(best_dir))
        
        if fold_idx == 0:
            with (config.CHECKPOINT_DIR / "label2id.json").open("w", encoding="utf-8") as f:
                json.dump(label2id, f, ensure_ascii=False, indent=2)

        print(f"[SUCCESS] Fold {fold_idx} complete. Best model saved to {best_dir}")
        
        import gc
        del model, trainer, train_ds, val_ds
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    print(f"\n5-Fold Audio Training Complete! All models saved in {config.CHECKPOINT_DIR}")

if __name__ == "__main__":
    main()
