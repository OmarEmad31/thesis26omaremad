"""
Egyptian Arabic Text SER — Master Refinement (v100)
==================================================
Techniques: Full Tuning, R-Drop, SCL Prototypical Loss, Focal Loss.
Designed to break the 43% plateau on the 44-sample sanitized test set.
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
# ELITE LOSS SUITE
# ─────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()

def compute_kl_loss(p, q):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='batchmean')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='batchmean')
    return (p_loss + q_loss) / 2

# ─────────────────────────────────────────────────────────
# MASTER TRAINER
# ─────────────────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, df, tokenizer, lid):
        self.enc = tokenizer(list(df['transcript']), truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        self.labels = [lid[e] for e in df['emotion_final']]
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.enc.items()}, torch.tensor(self.labels[idx], dtype=torch.long)

class MasterTrainer:
    def __init__(self, model, tr_df, va_df, te_df, tokenizer):
        self.model = model.to("cuda")
        self.tr_df = tr_df
        self.va_df = va_df
        self.te_df = te_df
        self.tokenizer = tokenizer
        self.LID = {'Anger':0, 'Disgust':1, 'Fear':2, 'Happiness':3, 'Neutral':4, 'Sadness':5, 'Surprise':6}

    def train(self):
        tr_loader = DataLoader(TextDataset(self.tr_df, self.tokenizer, self.LID), batch_size=16, shuffle=True)
        va_loader = DataLoader(TextDataset(self.va_df, self.tokenizer, self.LID), batch_size=16)
        te_loader = DataLoader(TextDataset(self.te_df, self.tokenizer, self.LID), batch_size=16)

        # Differential Learning Rates
        optimizer = torch.optim.AdamW([
            {'params': [p for n, p in self.model.named_parameters() if "bert" in n], 'lr': 2e-5},
            {'params': [p for n, p in self.model.named_parameters() if "bert" not in n], 'lr': 8e-4}
        ], weight_decay=0.01)
        
        # Balanced Weights
        counts = self.tr_df['emotion_final'].map(self.LID).value_counts().to_dict()
        w = torch.tensor([1.0/counts.get(i, 1) for i in range(7)]).to("cuda")
        w = w / w.sum() * 7
        focal_crit = FocalLoss(weight=w)

        best_acc = 0
        for epoch in range(1, 21):
            self.model.train()
            for batch_data, labels in tqdm(tr_loader, desc=f"Epoch {epoch}"):
                batch = {k: v.to("cuda") for k, v in batch_data.items()}
                labels = labels.to("cuda")
                optimizer.zero_grad()
                
                # R-DROP: Double Pass
                out1 = self.model(**batch)
                out2 = self.model(**batch)
                
                loss_ce = (focal_crit(out1.logits, labels) + focal_crit(out2.logits, labels)) / 2
                loss_kl = compute_kl_loss(out1.logits, out2.logits)
                
                # SCL: Supervised Contrastive
                hidden = out1.hidden_states[-1][:, 0, :]
                features = F.normalize(hidden, p=2, dim=1)
                sim = torch.matmul(features, features.T) / 0.1
                mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
                mask *= (1 - torch.eye(labels.size(0), device="cuda"))
                valid = mask.sum(1) > 0
                loss_scl = 0
                if valid.any():
                    log_p = (sim - torch.max(sim, 1, True)[0].detach()) - torch.log(torch.exp(sim-torch.max(sim,1,True)[0].detach()).sum(1, True) + 1e-8)
                    loss_scl = - (mask[valid] * log_p[valid]).sum(1) / (mask[valid].sum(1) + 1e-8)
                    loss_scl = loss_scl.mean()

                loss = loss_ce + (4.0 * loss_kl) + (0.1 * loss_scl)
                loss.backward()
                optimizer.step()

            # Eval
            self.model.eval()
            p, t = [], []
            with torch.no_grad():
                for b, l in va_loader:
                    b = {k: v.to("cuda") for k, v in b.items()}
                    p.extend(torch.argmax(self.model(**b).logits, 1).cpu().numpy())
                    t.extend(l.numpy())
            
            acc = accuracy_score(t, p)
            print(f"📈 [VAL] Acc: {acc:.4f} | F1: {f1_score(t, p, average='macro'):.4f}")
            if acc > best_acc:
                best_acc = acc
                torch.save(self.model.state_dict(), "best_text_master.pt")

        # TEST
        print("\n🏁 MASTER TEST REPORT (44 samples)")
        self.model.load_state_dict(torch.load("best_text_master.pt"))
        self.model.eval()
        p, t = [], []
        with torch.no_grad():
            for b, l in te_loader:
                b = {k: v.to("cuda") for k, v in b.items()}
                p.extend(torch.argmax(self.model(**b).logits, 1).cpu().numpy())
                t.extend(l.numpy())
        print(f"FINAL TEST ACCURACY: {accuracy_score(t, p):.4f}")
        print(classification_report(t, p, target_names=list(self.LID.keys())))

def main():
    set_seed(42)
    root = Path("/content/drive/MyDrive/Thesis Project")
    split_dir = root / "data/processed/splits/final_sanitized"
    tr = pd.read_csv(split_dir / "train_augmented.csv")
    va = pd.read_csv(split_dir / "val.csv")
    te = pd.read_csv(split_dir / "test.csv")
    
    tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")
    LID = {'Anger':0, 'Disgust':1, 'Fear':2, 'Happiness':3, 'Neutral':4, 'Sadness':5, 'Surprise':6}
    id2label = {v: k for k, v in LID.items()}
    model = MARBERTWithMultiSampleDropout("UBC-NLP/MARBERT", num_labels=7, id2label=id2label, label2id=LID)
    
    trainer = MasterTrainer(model, tr, va, te, tokenizer)
    trainer.train()

if __name__ == "__main__": main()
