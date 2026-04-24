"""
Egyptian Arabic Text SER — Freeze & Focus (v99)
==============================================
Technique: Freeze Backbone (Layers 0-9), Weighted Loss, and Augmented Data.
Goal: Efficiently learn 1113 augmented samples without overfitting.
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

class FreezeTrainer:
    def __init__(self, model, tr_df, va_df, te_df, tokenizer):
        self.model = model.to("cuda")
        self.tr_df = tr_df
        self.va_df = va_df
        self.te_df = te_df
        self.tokenizer = tokenizer
        self.LID = {'Anger':0, 'Disgust':1, 'Fear':2, 'Happiness':3, 'Neutral':4, 'Sadness':5, 'Surprise':6}

    def train(self):
        # 1. FREEZE 0-9
        for name, param in self.model.named_parameters():
            if "encoder.layer" in name:
                layer_idx = int(name.split("encoder.layer.")[1].split(".")[0])
                if layer_idx < 10:
                    param.requires_grad = False
            elif "embeddings" in name:
                param.requires_grad = False

        tr_loader = DataLoader(TextDataset(self.tr_df, self.tokenizer, self.LID), batch_size=16, shuffle=True)
        va_loader = DataLoader(TextDataset(self.va_df, self.tokenizer, self.LID), batch_size=16)
        te_loader = DataLoader(TextDataset(self.te_df, self.tokenizer, self.LID), batch_size=16)

        # Higher LR for Focal layers (10, 11 and Head)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-4)
        
        # 2. HEAVY WEIGHTS for 0% classes
        counts = self.tr_df['emotion_final'].map(self.LID).value_counts().to_dict()
        w = torch.tensor([1.0/counts.get(i, 1) for i in range(7)]).to("cuda")
        w = w / w.sum() * 7
        w[2] *= 4.0 # Boost Fear
        w[6] *= 4.0 # Boost Surprise
        
        criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)

        best_acc = 0
        for epoch in range(1, 16):
            self.model.train()
            for batch in tqdm(tr_loader, desc=f"Epoch {epoch}"):
                batch = {k: v.to("cuda") for k, v in batch.items()}
                optimizer.zero_grad()
                out = self.model(batch['input_ids'], batch['attention_mask'])
                loss = criterion(out.logits, batch['labels'])
                loss.backward()
                optimizer.step()

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
                torch.save(self.model.state_dict(), "best_text_freeze.pt")

        # TEST
        print("\n🏁 FREEZE & FOCUS TEST REPORT")
        self.model.load_state_dict(torch.load("best_text_freeze.pt"))
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
    tr = pd.read_csv(split_dir / "train_augmented.csv") # Using the 1113 samples
    va = pd.read_csv(split_dir / "val.csv")
    te = pd.read_csv(split_dir / "test.csv")
    
    tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")
    LID = {'Anger':0, 'Disgust':1, 'Fear':2, 'Happiness':3, 'Neutral':4, 'Sadness':5, 'Surprise':6}
    id2label = {v: k for k, v in LID.items()}
    model = MARBERTWithMultiSampleDropout("UBC-NLP/MARBERT", num_labels=7, id2label=id2label, label2id=LID)
    
    trainer = FreezeTrainer(model, tr, va, te, tokenizer)
    trainer.train()

if __name__ == "__main__": main()
