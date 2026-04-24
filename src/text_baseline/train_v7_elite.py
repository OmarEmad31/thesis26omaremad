"""
Egyptian Arabic Text SER — The Super Expert (v96)
==============================================
Techniques: R-Drop, EMA, Multi-Sample Dropout, LLRD.
Optimized for the Augmented 1113-sample dataset.
"""

import os, json, random, sys, torch, copy
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
# ELITE COMPONENTS: EMA & R-DROP
# ─────────────────────────────────────────────────────────

class EMA:
    """Exponential Moving Average for weight smoothing."""
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] -= (1 - self.decay) * (self.shadow[name] - param.data)

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}

def kl_loss(p, q):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='batchmean')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='batchmean')
    return (p_loss + q_loss) / 2

# ─────────────────────────────────────────────────────────
# TRAINER
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

class EliteTrainer:
    def __init__(self, model, tr_df, va_df, te_df, tokenizer):
        self.model = model.to("cuda")
        self.tr_df = tr_df
        self.va_df = va_df
        self.te_df = te_df
        self.tokenizer = tokenizer
        self.LID = {'Anger':0, 'Disgust':1, 'Fear':2, 'Happiness':3, 'Neutral':4, 'Sadness':5, 'Surprise':6}
        self.ema = EMA(self.model, 0.999)

    def train(self):
        tr_loader = DataLoader(TextDataset(self.tr_df, self.tokenizer, self.LID), batch_size=16, shuffle=True)
        va_loader = DataLoader(TextDataset(self.va_df, self.tokenizer, self.LID), batch_size=16)
        te_loader = DataLoader(TextDataset(self.te_df, self.tokenizer, self.LID), batch_size=16)

        opt = torch.optim.AdamW(self.model.parameters(), lr=2e-5, weight_decay=0.01)
        sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=100, num_training_steps=len(tr_loader)*25)
        
        # Focal Loss for rare classes
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        best_f1 = 0
        for epoch in range(1, 26):
            self.model.train()
            for batch in tqdm(tr_loader, desc=f"Epoch {epoch}"):
                batch = {k: v.to("cuda") for k, v in batch.items()}
                opt.zero_grad()
                
                # R-DROP: Dual forward pass
                out1 = self.model(batch['input_ids'], batch['attention_mask'])
                out2 = self.model(batch['input_ids'], batch['attention_mask'])
                
                loss_ce = (criterion(out1.logits, batch['labels']) + criterion(out2.logits, batch['labels'])) / 2
                loss_kl = kl_loss(out1.logits, out2.logits)
                
                loss = loss_ce + (4.0 * loss_kl) # Alpha=4.0 for R-Drop
                loss.backward()
                opt.step()
                sched.step()
                self.ema.update()

            # Eval with EMA
            self.model.eval()
            self.ema.apply_shadow() # Test with the smoothed weights
            p, t = [], []
            with torch.no_grad():
                for b in va_loader:
                    b = {k: v.to("cuda") for k, v in b.items()}
                    p.extend(torch.argmax(self.model(b['input_ids'], b['attention_mask']).logits, 1).cpu().numpy())
                    t.extend(b['labels'].cpu().numpy())
            self.ema.restore()
            
            f1 = f1_score(t, p, average='macro')
            print(f"📈 [VAL] Acc: {accuracy_score(t, p):.4f} | Macro F1: {f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                self.ema.apply_shadow()
                torch.save(self.model.state_dict(), "best_text_elite.pt")
                self.ema.restore()

        # TEST
        print("\n🏁 FINAL ELITE TEST REPORT")
        self.model.load_state_dict(torch.load("best_text_elite.pt"))
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
    
    # LOAD THE AUGMENTED DATA
    tr = pd.read_csv(split_dir / "train_augmented.csv")
    va = pd.read_csv(split_dir / "val.csv")
    te = pd.read_csv(split_dir / "test.csv")
    
    tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")
    
    LID = {'Anger':0, 'Disgust':1, 'Fear':2, 'Happiness':3, 'Neutral':4, 'Sadness':5, 'Surprise':6}
    id2label = {v: k for k, v in LID.items()}
    model = MARBERTWithMultiSampleDropout("UBC-NLP/MARBERT", num_labels=7, id2label=id2label, label2id=LID)
    
    trainer = EliteTrainer(model, tr, va, te, tokenizer)
    trainer.train()

if __name__ == "__main__": main()
