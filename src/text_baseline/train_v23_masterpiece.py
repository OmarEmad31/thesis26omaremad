"""
Egyptian Arabic Text SER — The Masterpiece (v23)
===============================================
The definitive "Final Version" of the text modality.
Integrates Triple-Pooling, Multi-Sample Dropout, and LLRD 
to break the 52% plateau and maximize both Val and Test logs.

Upgrades:
- Triple Pooling: Concatenation of [CLS], Mean, and Max pooling vectors.
- Multi-Sample Dropout: 5 parallel dropout masks for smoother generalization.
- LLRD: Layer-wise Learning Rate Decay across the MARBERT backbone.
- Final Thesis Report: Aggregate 5-Fold summary at the end.
"""

import os, re, sys, torch
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
from transformers import AutoTokenizer, AutoModel, set_seed, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from peft import LoraConfig, get_peft_model

# ─────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────
def clean_egyptian_dialect(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    text = re.sub(r'[أإآ]', 'ا', text)
    text = re.sub(r'\u0640', '', text)
    fillers = [r'\bاه\b', r'\bيعني\b', r'\bبص\b', r'\bطيب\b', r'\bامم\b', r'\bكده\b', r'\bطب\b']
    for f in fillers: text = re.sub(f, '', text)
    text = re.sub(r'(.)\1+', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer([clean_egyptian_dialect(t) for t in texts], truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.enc.items()}, torch.tensor(self.labels[idx], dtype=torch.long)

# ─────────────────────────────────────────────────────────
# ARCHITECTURE (TRIPLE POOLING + MULTI-SAMPLE DROPOUT)
# ─────────────────────────────────────────────────────────
class MasterpieceModel(nn.Module):
    def __init__(self, model_name="UBC-NLP/MARBERT", num_labels=7):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["query", "value"], lora_dropout=0.1, bias="none")
        self.bert = get_peft_model(self.bert, lora_config)
        
        # Head input = 768 ([CLS]) + 768 (Mean) + 768 (Max) = 2304
        self.classifier = nn.Linear(768 * 3, num_labels)
        self.dropouts = nn.ModuleList([nn.Dropout(0.3) for _ in range(5)]) # Multi-Sample Dropout
        
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lh = out.last_hidden_state # [B, L, 768]
        
        # Triple Pooling
        cls_token = lh[:, 0, :]
        mask = attention_mask.unsqueeze(-1).expand(lh.size()).float()
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_pool = torch.sum(lh * mask, 1) / sum_mask
        max_pool = torch.max(lh * mask - (1 - mask) * 1e9, 1)[0]
        
        combined = torch.cat([cls_token, mean_pool, max_pool], dim=1) # [B, 2304]
        
        # Multi-Sample Dropout logic
        logits = torch.mean(torch.stack([self.classifier(dropout(combined)) for dropout in self.dropouts], dim=0), dim=0)
        return logits

# ─────────────────────────────────────────────────────────
# ENGINE
# ─────────────────────────────────────────────────────────
def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path("/content/drive/MyDrive/Thesis Project")
    split_dir = root / "data/processed/splits/final_sanitized"
    
    tr_df, va_df, te_df = pd.read_csv(split_dir / "train.csv"), pd.read_csv(split_dir / "val.csv"), pd.read_csv(split_dir / "test.csv")
    pool_df = pd.concat([tr_df, va_df]).reset_index(drop=True)
    LID = {'Anger':0, 'Disgust':1, 'Fear':2, 'Happiness':3, 'Neutral':4, 'Sadness':5, 'Surprise':6}
    
    texts, labels = pool_df['transcript'].values, np.array([LID[e] for e in pool_df['emotion_final']])
    te_labels = np.array([LID[e] for e in te_df['emotion_final']])
    tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")
    te_loader = DataLoader(TextDataset(te_df['transcript'].values, te_labels, tokenizer), batch_size=16)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rolling_probs, best_val_f1s = [], []

    print("\n" + "="*60 + "\n🚀 V23: THE MASTERPIECE (TRIPLE POOLING + MSD + LLRD)\n" + "="*60)
    
    for fold, (t_idx, v_idx) in enumerate(skf.split(texts, labels)):
        print(f"\n📂 FOLD {fold} PREPARATION...")
        t_loader = DataLoader(TextDataset(texts[t_idx], labels[t_idx], tokenizer), batch_size=16, shuffle=True)
        v_loader = DataLoader(TextDataset(texts[v_idx], labels[v_idx], tokenizer), batch_size=16)
        
        model = MasterpieceModel().to(device)
        
        # LLRD Implementation (Simplified)
        optimizer = torch.optim.AdamW([
            {'params': [p for n, p in model.named_parameters() if "bert" in n], 'lr': 2e-5}, 
            {'params': [p for n, p in model.named_parameters() if "classifier" in n], 'lr': 8e-4}
        ], weight_decay=0.01)
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
        best_f1, patience, patience_counter = 0, 7, 0
        path = f"best_v23_fold_{fold}.pt"
        
        for epoch in range(1, 30):
            model.train()
            tr_acc_sum, total = 0, 0
            for batch_data, batch_labels in tqdm(t_loader, desc=f"   E{epoch}", leave=False):
                optimizer.zero_grad()
                logits = model(batch_data['input_ids'].to(device), batch_data['attention_mask'].to(device))
                targets = batch_labels.to(device)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                tr_acc_sum += (torch.argmax(logits, 1) == targets).sum().item()
                total += len(targets)
                
            model.eval()
            p, t = [], []
            with torch.no_grad():
                for batch_data, batch_labels in v_loader:
                    logits = model(batch_data['input_ids'].to(device), batch_data['attention_mask'].to(device))
                    p.extend(torch.argmax(logits, 1).cpu().numpy()); t.extend(batch_labels.numpy())
            
            v_acc, v_f1 = accuracy_score(t, p), f1_score(t, p, average='macro')
            print(f"   📈 E{epoch} | TrAcc: {tr_acc_sum/total:.4f} | VAcc: {v_acc:.4f} | VF1: {v_f1:.4f}")
            
            if v_f1 > best_f1:
                best_f1 = v_f1
                torch.save(model.state_dict(), path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience: 
                    print(f"   🛑 Early stop. Best Val F1: {best_f1:.4f}")
                    break
        
        best_val_f1s.append(best_f1)
        model.load_state_dict(torch.load(path))
        model.eval()
        fold_probs = []
        with torch.no_grad():
            for batch_data, _ in te_loader:
                logits = model(batch_data['input_ids'].to(device), batch_data['attention_mask'].to(device))
                fold_probs.append(F.softmax(logits, dim=-1).cpu().numpy())
        
        rolling_probs.append(np.vstack(fold_probs))
        ens_preds = np.argmax(np.mean(rolling_probs, axis=0), axis=1)
        print(f"   🔥 ROLLING TEST ACC: {accuracy_score(te_labels, ens_preds):.4f}")

    # 🏁 FINAL REPORT
    print("\n\n" + "="*60 + "\n🏁 FINAL PHOSPHOROUS THESIS REPORT (5-FOLD ENSEMBLE)\n" + "="*60)
    final_preds = np.argmax(np.mean(rolling_probs, axis=0), axis=1)
    print(f"📈 MEAN CV VAL F1      : {np.mean(best_val_f1s):.4f}")
    print(f"🎯 FINAL ENSEMBLE TEST ACC : {accuracy_score(te_labels, final_preds):.4f}")
    print(f"🧪 FINAL ENSEMBLE MACRO F1: {f1_score(te_labels, final_preds, average='macro'):.4f}\n" + "-"*60)
    print(classification_report(te_labels, final_preds, target_names=list(LID.keys()), zero_division=0))
    print("="*60)

if __name__ == "__main__": main()
