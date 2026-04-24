"""
Egyptian Arabic Text SER — The Ultimate "Kitchen Sink" Upgrade (v91)
================================================================
Combines SCL, FGM, LLRD, MSD, and Weighted Sampling for maximum generalization.
"""

import os, json, random, sys, torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Path injection
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, set_seed, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.text_baseline.model import MARBERTWithMultiSampleDropout
from src.text_baseline import config

# ─────────────────────────────────────────────────────────
# ADVANCED TRAINING COMPONENTS
# ─────────────────────────────────────────────────────────

class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}
    def attack(self, epsilon=0.5, emb_name="word_embeddings."):
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

def get_optimizer_params(model, base_lr=2e-5, head_lr=5e-4, decay=0.95):
    params = []
    # Classifier Head
    params.append({'params': [p for n, p in model.named_parameters() if "bert" not in n], 'lr': head_lr, 'weight_decay': 0.0})
    # Transformer Layers
    for i in range(12):
        lr = base_lr * (decay ** (11 - i))
        params.append({'params': [p for n, p in model.named_parameters() if f"encoder.layer.{i}." in n], 'lr': lr, 'weight_decay': 0.01})
    return params

class TextDataset(Dataset):
    def __init__(self, df, tokenizer, lid):
        self.texts = tokenizer(list(df['transcript']), truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        self.labels = [lid[e] for e in df['emotion_final']]
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {
            'input_ids': self.texts['input_ids'][idx],
            'attention_mask': self.texts['attention_mask'][idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ─────────────────────────────────────────────────────────
# THE MASTER TRAINER
# ─────────────────────────────────────────────────────────

class UltimateTextTrainer:
    def __init__(self, model, tr_df, va_df, te_df, tokenizer):
        self.model = model.to("cuda")
        self.tr_df = tr_df
        self.va_df = va_df
        self.te_df = te_df
        self.tokenizer = tokenizer
        self.LID = {'Anger':0, 'Disgust':1, 'Fear':2, 'Happiness':3, 'Neutral':4, 'Sadness':5, 'Surprise':6}
        self.fgm = FGM(self.model)

    def train(self):
        # Data Preparation
        tr_ds = TextDataset(self.tr_df, self.tokenizer, self.LID)
        counts = self.tr_df['emotion_final'].map(self.LID).value_counts().to_dict()
        weights = [1.0 / counts[i] for i in [self.LID[e] for e in self.tr_df['emotion_final']]]
        sampler = WeightedRandomSampler(weights, num_samples=len(tr_ds), replacement=True)
        
        tr_loader = DataLoader(tr_ds, batch_size=16, sampler=sampler)
        va_loader = DataLoader(TextDataset(self.va_df, self.tokenizer, self.LID), batch_size=16)
        te_loader = DataLoader(TextDataset(self.te_df, self.tokenizer, self.LID), batch_size=16)

        opt = torch.optim.AdamW(get_optimizer_params(self.model), weight_decay=0.01)
        sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=100, num_training_steps=len(tr_loader)*20)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.15)

        best_f1 = 0
        for epoch in range(1, 21):
            self.model.train()
            for batch in tqdm(tr_loader, desc=f"Epoch {epoch}"):
                batch = {k: v.to("cuda") for k, v in batch.items()}
                opt.zero_grad()
                
                outputs = self.model(batch['input_ids'], attention_mask=batch['attention_mask'])
                logits = outputs.logits
                loss = criterion(logits, batch['labels'])
                
                # ADD SCL LOSS (Contrastive)
                hidden = outputs.hidden_states[-1][:, 0, :]
                features = F.normalize(hidden, p=2, dim=1)
                sim = torch.matmul(features, features.T) / 0.1
                mask = torch.eq(batch['labels'].unsqueeze(1), batch['labels'].unsqueeze(0)).float()
                diag_mask = 1 - torch.eye(batch['labels'].size(0), device="cuda")
                mask *= diag_mask
                valid = mask.sum(1) > 0
                if valid.any():
                    # Optimized numerically stable log_softmax
                    log_p = (sim - torch.max(sim, 1, True)[0].detach()) - torch.log(torch.exp(sim-torch.max(sim,1,True)[0].detach()).sum(1, True)*diag_mask + 1e-8)
                    scl = - (mask[valid] * log_p[valid]).sum(1) / (mask[valid].sum(1) + 1e-8)
                    loss += 0.1 * scl.mean()

                loss.backward()
                self.fgm.attack() # Adversarial Punch
                adv_out = self.model(batch['input_ids'], batch['attention_mask'])
                criterion(adv_out['logits'], batch['labels']).backward()
                self.fgm.restore()
                
                opt.step()
                sched.step()

            # Eval
            self.model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for b in va_loader:
                    b = {k: v.to("cuda") for k, v in b.items()}
                    preds.extend(torch.argmax(self.model(b['input_ids'], b['attention_mask'])['logits'], 1).cpu().numpy())
                    targets.extend(b['labels'].cpu().numpy())
            
            f1 = f1_score(targets, preds, average='macro')
            print(f"📈 [VAL] Acc: {accuracy_score(targets, preds):.4f} | Macro F1: {f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                torch.save(self.model.state_dict(), "best_text_kitchen_sink.pt")

        # TEST
        print("\n🏁 ULTIMATE TEST REPORT")
        self.model.load_state_dict(torch.load("best_text_kitchen_sink.pt"))
        self.model.eval()
        p, t = [], []
        with torch.no_grad():
            for b in te_loader:
                b = {k: v.to("cuda") for k, v in b.items()}
                p.extend(torch.argmax(self.model(b['input_ids'], b['attention_mask'])['logits'], 1).cpu().numpy())
                t.extend(b['labels'].cpu().numpy())
        print(f"TEST ACCURACY: {accuracy_score(t, p):.4f}")
        print(classification_report(t, p, target_names=list(self.LID.keys())))

def main():
    set_seed(42)
    root = Path("/content/drive/MyDrive/Thesis Project")
    split_dir = root / "data/processed/splits/final_sanitized"
    tr, va, te = pd.read_csv(split_dir / "train.csv"), pd.read_csv(split_dir / "val.csv"), pd.read_csv(split_dir / "test.csv")
    tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")
    
    LID = {'Anger':0, 'Disgust':1, 'Fear':2, 'Happiness':3, 'Neutral':4, 'Sadness':5, 'Surprise':6}
    id2label = {v: k for k, v in LID.items()}
    model = MARBERTWithMultiSampleDropout("UBC-NLP/MARBERT", num_labels=7, id2label=id2label, label2id=LID)
    
    trainer = UltimateTextTrainer(model, tr, va, te, tokenizer)
    trainer.train()

if __name__ == "__main__": main()
