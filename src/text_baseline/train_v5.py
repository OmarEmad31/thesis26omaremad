"""
Egyptian Arabic Text SER — R-Drop Guardian Suite (v93)
====================================================
Goal: Break 50% using R-Drop (Regularized Dropout) and Focal Loss.
Designed for 511 samples.
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
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, set_seed, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.text_baseline.model import MARBERTWithMultiSampleDropout
from src.text_baseline import config

# ─────────────────────────────────────────────────────────
# ADVANCED LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        else: return focal_loss.sum()

def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()
    loss = (p_loss + q_loss) / 2
    return loss

# ─────────────────────────────────────────────────────────
# DATA & TRAINING
# ─────────────────────────────────────────────────────────

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

class RDropTrainer:
    def __init__(self, model, tr_df, va_df, te_df, tokenizer):
        self.model = model.to("cuda")
        self.tr_df = tr_df
        self.va_df = va_df
        self.te_df = te_df
        self.tokenizer = tokenizer
        self.LID = {'Anger':0, 'Disgust':1, 'Fear':2, 'Happiness':3, 'Neutral':4, 'Sadness':5, 'Surprise':6}
        self.alpha = 4.0 # R-Drop weight

    def train(self):
        tr_loader = DataLoader(TextDataset(self.tr_df, self.tokenizer, self.LID), batch_size=16, shuffle=True)
        va_loader = DataLoader(TextDataset(self.va_df, self.tokenizer, self.LID), batch_size=16)
        te_loader = DataLoader(TextDataset(self.te_df, self.tokenizer, self.LID), batch_size=16)

        # Aggressive LLRD (Decay 0.90)
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if "bert" not in n], 'lr': 5e-4},
            {'params': [p for n, p in self.model.named_parameters() if "bert" in n], 'lr': 2e-5, 'weight_decay': 0.01}
        ]
        opt = torch.optim.AdamW(optimizer_grouped_parameters)
        sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=100, num_training_steps=len(tr_loader)*20)
        focal = FocalLoss(gamma=2.0)

        best_acc = 0
        for epoch in range(1, 21):
            self.model.train()
            for batch in tqdm(tr_loader, desc=f"Epoch {epoch}"):
                batch = {k: v.to("cuda") for k, v in batch.items()}
                opt.zero_grad()
                
                # R-DROP: TWO FORWARD PASSES
                out1 = self.model(batch['input_ids'], batch['attention_mask'])
                out2 = self.model(batch['input_ids'], batch['attention_mask'])
                
                # Bi-directional KL Divergence
                loss_kl = compute_kl_loss(out1.logits, out2.logits)
                loss_ce = (focal(out1.logits, batch['labels']) + focal(out2.logits, batch['labels'])) / 2
                
                loss = loss_ce + (self.alpha * loss_kl)
                loss.backward()
                opt.step()
                sched.step()

            # Eval
            self.model.eval()
            p, t = [], []
            with torch.no_grad():
                for b in va_loader:
                    b = {k: v.to("cuda") for k, v in b.items()}
                    p.extend(torch.argmax(self.model(b['input_ids'], b['attention_mask']).logits, 1).cpu().numpy())
                    t.extend(b['labels'].cpu().numpy())
            
            acc = accuracy_score(t, p)
            print(f"📈 [VAL] Acc: {acc:.4f} | F1: {f1_score(t, p, average='macro'):.4f}")
            if acc > best_acc:
                best_acc = acc
                torch.save(self.model.state_dict(), "best_text_rdrop.pt")

        # TEST
        print("\n🏁 R-DROP TEST REPORT")
        self.model.load_state_dict(torch.load("best_text_rdrop.pt"))
        self.model.eval()
        p, t = [], []
        with torch.no_grad():
            for b in te_loader:
                b = {k: v.to("cuda") for k, v in b.items()}
                p.extend(torch.argmax(self.model(b['input_ids'], b['attention_mask']).logits, 1).cpu().numpy())
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
    
    trainer = RDropTrainer(model, tr, va, te, tokenizer)
    trainer.train()

if __name__ == "__main__": main()
