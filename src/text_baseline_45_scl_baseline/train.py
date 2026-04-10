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


class SCLTrainer(Trainer):
    """Trainer with Supervised Contrastive Learning (SCL) and CrossEntropy."""

    def __init__(self, *args, class_weights: torch.Tensor, scl_temp: float = 0.1, scl_weight: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.scl_temp = scl_temp
        self.scl_weight = scl_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        
        # We need hidden states for SCL
        inputs["output_hidden_states"] = True
        outputs = model(**inputs)
        
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

    # Class weights (balanced) from training label frequencies
    raw_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_labels),
        y=np.array(train_ids),
    )
    # Amplify minority-class weights moderately to push the model
    # harder toward low-frequency classes (Fear, Surprise, Disgust).
    raw_weights = raw_weights ** 1.5
    raw_weights = raw_weights / raw_weights.min()  # keep min-weight = 1.0
    class_weights_tensor = torch.tensor(raw_weights, dtype=torch.float)
    print(f"  class weights: { {label_names[i]: round(float(raw_weights[i]), 3) for i in range(num_labels)} }")

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    train_ds = build_hf_dataset(train_texts, train_ids, tokenizer, config.MAX_LENGTH)
    val_ds = build_hf_dataset(val_texts, val_ids, tokenizer, config.MAX_LENGTH)
    test_ds = build_hf_dataset(test_texts, test_ids, tokenizer, config.MAX_LENGTH)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    out_dir = config.CHECKPOINT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    best_dir = out_dir / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
        num_train_epochs=config.NUM_EPOCHS,
        weight_decay=config.WEIGHT_DECAY,
        warmup_ratio=config.WARMUP_RATIO,
        label_smoothing_factor=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        seed=config.SEED,
        report_to="none",
        dataloader_pin_memory=False,
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
    )

    print("Training MARBERT text baseline...")
    print(f"  model: {config.MODEL_NAME}")
    print(f"  train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")
    print(f"  labels ({num_labels}): {label_names}")

    trainer.train()

    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    with (best_dir / "label2id.json").open("w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)

    def report_split(ds, y_true_np: np.ndarray, name: str) -> None:
        pred_out = trainer.predict(ds)
        preds = np.argmax(pred_out.predictions, axis=-1)
        m = evaluate_predictions(y_true_np, preds)
        print(f"\n=== {name} ===")
        print(f"  accuracy:     {m['accuracy']:.4f}")
        print(f"  macro F1:     {m['f1_macro']:.4f}")
        print(f"  weighted F1: {m['f1_weighted']:.4f}")
        cm = confusion_matrix_labels(y_true_np, preds, num_labels)
        print_confusion_matrix(cm, label_names, name)

    y_val = np.array(val_ids, dtype=np.int64)
    y_test = np.array(test_ids, dtype=np.int64)

    report_split(val_ds, y_val, "validation")
    report_split(test_ds, y_test, "test")

    print(f"\nBest checkpoint saved to: {best_dir}")


if __name__ == "__main__":
    main()
