"""
Egyptian Arabic Text SER — 50% Hunter Mode (v87)
==============================================
Goal: Push the sanitized accuracy from 43% to 50%+.
Techniques: Differential LRs, Increased Epochs, and Weighted Focal Loss.
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
from transformers import AutoTokenizer, set_seed
from sklearn.metrics import accuracy_score, f1_score, classification_report
from src.text_baseline.model import MARBERTWithMultiSampleDropout
from src.text_baseline import config

class TrainerV3:
    def __init__(self, model, train_df, val_df, test_df, tokenizer):
        self.model = model
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if hasattr(config, 'EMOTIONS'):
            self.LID = {e: i for i, e in enumerate(config.EMOTIONS)}
        else:
            self.LID = {'Anger':0, 'Disgust':1, 'Fear':2, 'Happiness':3, 'Neutral':4, 'Sadness':5, 'Surprise':6}

    def tokenize(self, texts):
        return self.tokenizer(list(texts), truncation=True, padding=True, max_length=64, return_tensors="pt").to(self.device)

    def train(self):
        # 1. Differential Learning Rates
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if "bert" in n], 'lr': 2e-5},
            {'params': [p for n, p in self.model.named_parameters() if "bert" not in n], 'lr': 5e-4}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        best_f1 = 0
        for epoch in range(1, 16):
            self.model.train()
            # Shuffle training data
            train_shuffled = self.train_df.sample(frac=1).reset_index(drop=True)
            for i in tqdm(range(0, len(train_shuffled), 16), desc=f"Epoch {epoch}"):
                batch = train_shuffled.iloc[i:i+16]
                inputs = self.tokenize(batch['transcript'].values)
                labels = torch.tensor([self.LID[e] for e in batch['emotion_final']]).to(self.device)
                
                optimizer.zero_grad()
                out = self.model(inputs['input_ids'], inputs['attention_mask'])
                loss = criterion(out.logits, labels)
                loss.backward()
                optimizer.step()

            # Eval
            self.model.eval()
            v_preds, v_targets = [], []
            with torch.no_grad():
                for i in range(0, len(self.val_df), 16):
                    batch = self.val_df.iloc[i:i+16]
                    inputs = self.tokenize(batch['transcript'].values)
                    out = self.model(inputs['input_ids'], inputs['attention_mask'])
                    v_preds.extend(torch.argmax(out.logits, 1).cpu().numpy())
                    v_targets.extend([self.LID[e] for e in batch['emotion_final']])
            
            acc = accuracy_score(v_targets, v_preds)
            f1 = f1_score(v_targets, v_preds, average='macro')
            print(f"📈 [VAL] Acc: {acc:.4f} | Macro F1: {f1:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                torch.save(self.model.state_dict(), "best_text_v3.pt")

        # Final Test
        print("\n🏁 FINAL TEST REPORT (Unseen 44 samples)")
        self.model.load_state_dict(torch.load("best_text_v3.pt"))
        self.model.eval()
        t_preds, t_targets = [], []
        with torch.no_grad():
            for i in range(0, len(self.test_df), 16):
                batch = self.test_df.iloc[i:i+16]
                inputs = self.tokenize(batch['transcript'].values)
                out = self.model(inputs['input_ids'], inputs['attention_mask'])
                t_preds.extend(torch.argmax(out.logits, 1).cpu().numpy())
                t_targets.extend([self.LID[e] for e in batch['emotion_final']])
        
        print(f"TEST ACCURACY: {accuracy_score(t_targets, t_preds):.4f}")
        print(classification_report(t_targets, t_preds, target_names=list(self.LID.keys())))

def main():
    set_seed(42)
    root = Path("/content/drive/MyDrive/Thesis Project")
    split_dir = root / "data/processed/splits/final_sanitized"
    tr = pd.read_csv(split_dir / "train.csv")
    va = pd.read_csv(split_dir / "val.csv")
    te = pd.read_csv(split_dir / "test.csv")
    
    tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")
    model = MARBERTWithMultiSampleDropout("UBC-NLP/MARBERT", num_labels=7).to("cuda")
    
    trainer = TrainerV3(model, tr, va, te, tokenizer)
    trainer.train()

if __name__ == "__main__": main()
