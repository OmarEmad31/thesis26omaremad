"""
Egyptian Arabic SER — Sanitized Text Baseline (EMULATED V81)
==========================================================
Directly mirrors the successful 51% ensemble script logic.
Points to the Cleaned Track A data pool.
"""

import os, json, random, sys, torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Fix: Guarantee project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from src.text_baseline.model import MARBERTWithMultiSampleDropout
from src.text_baseline.metrics_utils import compute_metrics

# Re-implementing the successful Trainer & FGM from your train.py
class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}
    def attack(self, epsilon=1.0, emb_name="word_embeddings."):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    param.data.add_(epsilon * param.grad / norm)
    def restore(self, emb_name="word_embeddings."):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                param.data = self.backup[name]
        self.backup = {}

class SCLTrainer(Trainer):
    def __init__(self, *args, class_weights, scl_temp=0.1, scl_weight=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.scl_temp = scl_temp
        self.scl_weight = scl_weight
        self.fgm = FGM(self.model)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        model_inputs["output_hidden_states"] = True
        outputs = model(**model_inputs)
        logits = outputs.logits
        
        ce_loss = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))(logits, labels)
        
        # SCL Logic
        hidden = outputs.hidden_states[-1][:, 0, :]  
        features = F.normalize(hidden, p=2, dim=1)
        similarity = torch.matmul(features, features.T) / self.scl_temp
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        logits_mask = torch.ones_like(mask) - torch.eye(labels.size(0), device=mask.device)
        mask = mask * logits_mask
        exp_sim = torch.exp(similarity - torch.max(similarity, dim=1, keepdim=True)[0].detach()) * logits_mask
        log_prob = (similarity - torch.max(similarity, dim=1, keepdim=True)[0].detach()) - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        valid = mask.sum(1) > 0
        scl_loss = - (mask[valid] * log_prob[valid]).sum(1) / (mask[valid].sum(1) + 1e-8)
        scl_loss = scl_loss.mean() if valid.any() else torch.tensor(0.0).to(ce_loss.device)
        
        loss = ce_loss + (self.scl_weight * scl_loss)
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)
        loss.backward()
        self.fgm.attack(epsilon=1.0)
        loss_adv = self.compute_loss(model, inputs)
        loss_adv.backward()
        self.fgm.restore()
        return loss.detach()

def get_lr_groups(model):
    no_decay = ["bias", "LayerNorm.weight"]
    params = []
    base_lr = 3e-5
    decay = 0.95
    # Head Groups
    params.append({"params": [p for n, p in model.named_parameters() if "bert" not in n], "lr": base_lr, "weight_decay": 0.0})
    # Encoder Layers
    for i in range(12):
        params.append({"params": [p for n, p in model.named_parameters() if f"encoder.layer.{i}." in n], "lr": base_lr * (decay**(12-i)), "weight_decay": 0.01})
    return params

def main():
    set_seed(42)
    root = Path("/content/drive/MyDrive/Thesis Project")
    clean_p = root / "data/processed/splits/trackA_cleaned"
    
    # LOAD THE SANITIZED POOL
    tr_df = pd.read_csv(clean_p / "trackA_train_clean.csv")
    va_df = pd.read_csv(clean_p / "trackA_val_clean.csv")
    all_df = pd.concat([tr_df, va_df])
    
    LID = {e: i for i, e in enumerate(['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise'])}
    all_df['label_id'] = all_df['emotion_final'].map(LID)
    
    texts = all_df['transcript'].values
    labels = all_df['label_id'].values
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")

    for fold, (t_idx, v_idx) in enumerate(skf.split(texts, labels)):
        print(f"\n🚀 FOLD {fold} STARTING...")
        train_ds = Dataset.from_dict({"text": texts[t_idx], "labels": labels[t_idx]})
        val_ds = Dataset.from_dict({"text": texts[v_idx], "labels": labels[v_idx]})
        
        def tokenize_fn(x): return tokenizer(x["text"], truncation=True, padding="max_length", max_length=64)
        train_ds = train_ds.map(tokenize_fn, batched=True); val_ds = val_ds.map(tokenize_fn, batched=True)

        w = compute_class_weight("balanced", classes=np.arange(7), y=labels[t_idx])
        w = torch.tensor(w**1.5, dtype=torch.float)

        model = MARBERTWithMultiSampleDropout("UBC-NLP/MARBERT", num_labels=7)
        
        args = TrainingArguments(
            output_dir=f"checkpoints_sanitized/fold_{fold}",
            learning_rate=3e-5, per_device_train_batch_size=16,
            num_train_epochs=12, weight_decay=0.01, warmup_ratio=0.3,
            eval_strategy="epoch", save_strategy="no", report_to="none"
        )
        
        opt = torch.optim.AdamW(get_lr_groups(model))
        trainer = SCLTrainer(
            model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
            class_weights=w, data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics, optimizers=(opt, None)
        )
        trainer.train()
        trainer.save_model(f"checkpoints_sanitized/fold_{fold}/best")

from datasets import Dataset
if __name__ == "__main__": main()
