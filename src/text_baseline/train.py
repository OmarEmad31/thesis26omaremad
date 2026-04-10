"""
Train MARBERT for emotion_final from text_eligible splits.
Run from project root: python -m src.text_baseline.train
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

# Ensure project root is importable when this file is executed directly.
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from sklearn.model_selection import StratifiedKFold

from src.text_baseline import config  # noqa: E402
from src.text_baseline.data import (  # noqa: E402
    build_label2id,
    encode_labels,
    labels_in_order,
    load_split_csv,
)
from src.text_baseline.metrics_utils import (  # noqa: E402
    compute_metrics,
    confusion_matrix_labels,
    evaluate_predictions,
)


def build_hf_dataset(
    texts: list[str],
    label_ids: list[int],
    tokenizer,
    max_length: int,
) -> Dataset:
    ds = Dataset.from_dict({"text": texts, "labels": label_ids})

    def tokenize_batch(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
        )

    return ds.map(tokenize_batch, batched=True, remove_columns=["text"])


from src.text_baseline.model import MARBERTWithMultiSampleDropout


class FGM:
    """Fast Gradient Method for adversarial training on embeddings."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.backup = {}

    def attack(self, epsilon: float = 1.0, emb_name: str = "word_embeddings."):
        # Loop through all params and find embeddings
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name: str = "word_embeddings."):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}



class SCLTrainer(Trainer):
    """Trainer with Supervised Contrastive Learning (SCL) and CrossEntropy."""

    def __init__(self, *args, class_weights: torch.Tensor, scl_temp: float = 0.1, scl_weight: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.scl_temp = scl_temp
        self.scl_weight = scl_weight
        self.fgm = FGM(self.model)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        
        # We need hidden states for SCL; Ensure model doesn't see labels 
        # so it doesn't try to calculate its own standard loss inside.
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        model_inputs["output_hidden_states"] = True
        outputs = model(**model_inputs)
        
        logits = outputs.logits
        
        # 1. Standard Weighted Cross-Entropy Loss
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        ce_loss = loss_fn(logits, labels)
        
        # 2. Supervised Contrastive Loss (SCL)
        # Extract the raw [CLS] token embedding from the last hidden state
        hidden = outputs.hidden_states[-1][:, 0, :]  
        
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
            
        # 3. Hybrid Loss Combines Classification + Embedding Separation
        loss = ce_loss + (self.scl_weight * scl_loss)
        
        # Strip hidden_states back off before returning so the evaluation loop doesn't crash on tuple vstacking
        from transformers.modeling_outputs import SequenceClassifierOutput
        clean_outputs = SequenceClassifierOutput(loss=loss, logits=logits)
        
        return (loss, clean_outputs) if return_outputs else loss


def get_optimizer_grouped_parameters(model, base_lr, layerwise_lr_decay):
    """Build parameter groups with layer-wise learning rate decay (LLRD)."""
    no_decay = ["bias", "LayerNorm.weight"]
    
    # Identify the base model and encoder layers
    encoder_layers = model.bert.encoder.layer
    num_layers = len(encoder_layers)
    params = []
    
    # 1. Classifier head + pooling gets full LR (no decay)
    head_params = {
        "params": [
            p for n, p in model.named_parameters() 
            if "bert" not in n or "pooler" in n
        ],
        "weight_decay": 0.0,
        "lr": base_lr,
    }
    params.append(head_params)
    
    # 2. Embeddings get lowest LR
    emb_lr = base_lr * (layerwise_lr_decay ** (num_layers + 1))
    emb_params = {
        "params": [
            p for n, p in model.named_parameters() 
            if "embeddings" in n
        ],
        "weight_decay": 0.01,
        "lr": emb_lr,
    }
    params.append(emb_params)
    
    # 3. Intermediate encoder layers get gradually decaying LR
    for i in range(num_layers):
        lr = base_lr * (layerwise_lr_decay ** (num_layers - i))
        layer_params = {
            "params": [
                p for n, p in model.named_parameters() 
                if f"encoder.layer.{i}." in n
            ],
            "weight_decay": 0.01,
            "lr": lr,
        }
        params.append(layer_params)
        
    return params

    def training_step(self, model: nn.Module, inputs: dict[str, torch.Tensor], num_items_in_batch: int | None = None) -> torch.Tensor:
        """Override training_step to perform Adversarial Training (FGM)."""
        model.train()
        inputs = self._prepare_inputs(inputs)

        # 1. Standard forward + backward to get original gradients
        loss = self.compute_loss(model, inputs)
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        loss.backward()

        # 2. Adversarial attack
        self.fgm.attack(epsilon=1.0)
        
        # 3. Adversarial forward + backward
        loss_adv = self.compute_loss(model, inputs)
        if self.args.gradient_accumulation_steps > 1:
            loss_adv = loss_adv / self.args.gradient_accumulation_steps
        
        loss_adv.backward()
        
        # 4. Restore original embeddings
        self.fgm.restore()

        return loss.detach()


def print_confusion_matrix(cm: np.ndarray, label_names: list[str], title: str) -> None:
    print(f"\n--- Confusion matrix ({title}) ---")
    header = "true\\pred".ljust(14) + "".join(f"{n[:10]:>12}" for n in label_names)
    print(header)
    for i, row_name in enumerate(label_names):
        line = f"{row_name[:12]:<14}" + "".join(f"{cm[i, j]:>12d}" for j in range(cm.shape[1]))
        print(line)


def main() -> None:
    set_seed(config.SEED)
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    train_texts, train_labels = load_split_csv(
        config.TRAIN_CSV, config.TEXT_COLUMN, config.LABEL_COLUMN
    )
    val_texts, val_labels = load_split_csv(config.VAL_CSV, config.TEXT_COLUMN, config.LABEL_COLUMN)
    test_texts, test_labels = load_split_csv(config.TEST_CSV, config.TEXT_COLUMN, config.LABEL_COLUMN)

    label2id = build_label2id(train_labels)
    id2label = {v: k for k, v in label2id.items()}
    label_names = labels_in_order(label2id)
    num_labels = len(label2id)

    for split_name, labs in [("val", val_labels), ("test", test_labels)]:
        unknown = sorted(set(labs) - set(label2id.keys()))
        if unknown:
            print(f"ERROR: {split_name} has labels not in train: {unknown}", file=sys.stderr)
            sys.exit(1)

    train_ids = encode_labels(train_labels, label2id)
    val_ids = encode_labels(val_labels, label2id)
    test_ids = encode_labels(test_labels, label2id)

    # Combine train and val for K-Fold
    all_texts = np.array(train_texts + val_texts)
    all_labels = np.array(train_ids + val_ids)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    test_ds = build_hf_dataset(test_texts, test_ids, tokenizer, config.MAX_LENGTH)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    for fold_idx, (train_index, val_index) in enumerate(skf.split(all_texts, all_labels)):
        fold_out_dir = config.CHECKPOINT_DIR / f"fold_{fold_idx}"
        best_dir = fold_out_dir / "best_model"
        
        # Resume Logic: Skip if already done
        if (best_dir / "model.safetensors").exists() or (best_dir / "pytorch_model.bin").exists():
            print(f"\n[RESUME] Skipping Fold {fold_idx} as it is already complete.")
            continue
        
        # Special Cleanup for Fold 2 (since it was interrupted)
        if fold_idx == 2 and fold_out_dir.exists():
            print(f"\n[CLEANUP] Removing incomplete Fold 2 directory for a fresh start.")
            import shutil
            shutil.rmtree(fold_out_dir)

        print(f"\n{'='*20} TRAINING FOLD {fold_idx} {'='*20}")
        
        fold_train_texts = all_texts[train_index].tolist()
        fold_train_labels = all_labels[train_index].tolist()
        fold_val_texts = all_texts[val_index].tolist()
        fold_val_labels = all_labels[val_index].tolist()

        # Class weights for this specific fold
        raw_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(num_labels),
            y=np.array(fold_train_labels),
        )
        raw_weights = raw_weights ** 1.5
        raw_weights = raw_weights / raw_weights.min()
        class_weights_tensor = torch.tensor(raw_weights, dtype=torch.float)

        model = MARBERTWithMultiSampleDropout(
            config.MODEL_NAME,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

        train_ds = build_hf_dataset(fold_train_texts, fold_train_labels, tokenizer, config.MAX_LENGTH)
        val_ds = build_hf_dataset(fold_val_texts, fold_val_labels, tokenizer, config.MAX_LENGTH)

        fold_out_dir = config.CHECKPOINT_DIR / f"fold_{fold_idx}"
        fold_out_dir.mkdir(parents=True, exist_ok=True)
        best_dir = fold_out_dir / "best_model"
        best_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(fold_out_dir),
            learning_rate=config.LEARNING_RATE,
            per_device_train_batch_size=config.BATCH_SIZE,
            per_device_eval_batch_size=config.BATCH_SIZE,
            gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
            num_train_epochs=config.NUM_EPOCHS,
            weight_decay=config.WEIGHT_DECAY,
            warmup_ratio=config.WARMUP_RATIO,
            label_smoothing_factor=0.15,
            eval_strategy="epoch",
            save_strategy="no",  # Disable intermediate saves to save memory
            load_best_model_at_end=False, # We will save the final model manually
            save_total_limit=1,
            logging_steps=50,
            seed=config.SEED + fold_idx,
            report_to="none",
            dataloader_pin_memory=False,
        )

        grouped_params = get_optimizer_grouped_parameters(
            model, 
            base_lr=config.LEARNING_RATE, 
            layerwise_lr_decay=0.95
        )
        optimizer = torch.optim.AdamW(grouped_params)
        
        num_train_steps = (len(train_ds) // config.BATCH_SIZE) * config.NUM_EPOCHS
        num_warmup_steps = int(num_train_steps * config.WARMUP_RATIO)
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=num_warmup_steps, 
            num_training_steps=num_train_steps
        )

        trainer = SCLTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            class_weights=class_weights_tensor,
            scl_temp=0.1,
            scl_weight=0.1,
            optimizers=(optimizer, scheduler),
        )

        trainer.train()
        trainer.save_model(str(best_dir))
        
        # Save label2id once in the root checkpoint dir
        if fold_idx == 0:
            with (config.CHECKPOINT_DIR / "label2id.json").open("w", encoding="utf-8") as f:
                json.dump(label2id, f, ensure_ascii=False, indent=2)

        print(f"\n[SUCCESS] Fold {fold_idx} complete. Best model saved to {best_dir}")
        
        # Memory Cleanup (Crucial for CPU training with 5 folds)
        import gc
        del model
        del trainer
        del optimizer
        del scheduler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n5-Fold Training Complete. All models saved in {config.CHECKPOINT_DIR}")
    print(f"To see the final 50%+ ensemble result, run: python -m src.text_baseline.ensemble_predict")


if __name__ == "__main__":
    main()
