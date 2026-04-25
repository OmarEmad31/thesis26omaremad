"""
Egyptian Arabic Text SER — The Omega Point (v29)
==============================================
The absolute frontier of NIR-level text classification. 
Integrates R-Drop, Attention Pooling, and OneCycleLR.

Upgrades:
- R-Drop: Forces consistency across dropout passes (massive stability).
- Attention Pooling: Learnable weights to find the "emotional" words.
- OneCycleLR: Professional scheduler for deep minima convergence.
- Quad-Pooling + PolyLoss: Retained for feature richness.
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
from transformers import AutoTokenizer, AutoModel, set_seed, get_linear_schedule_with_warmup
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
    def __getitem__(self, idx): return {k: v[idx] for k, v in self.enc.items()}, torch.tensor(self.labels[idx], dtype=torch.long)

# ─────────────────────────────────────────────────────────
# LOSS: POLYLOSS + R-DROP KL
# ─────────────────────────────────────────────────────────
def compute_kl_loss(p, q):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    return (p_loss.sum() + q_loss.sum()) / 2

class OmegaLoss(nn.Module):
    def __init__(self, epsilon=1.0, alpha=5.0):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = alpha
    def forward(self, logits1, logits2, targets):
        # PolyLoss 1
        ce1 = F.cross_entropy(logits1, targets, reduction='none', label_smoothing=0.05)
        pt1 = torch.exp(-ce1)
        poly1 = (ce1 + self.epsilon * (1 - pt1)).mean()
        # PolyLoss 2
        ce2 = F.cross_entropy(logits2, targets, reduction='none', label_smoothing=0.05)
        pt2 = torch.exp(-ce2)
        poly2 = (ce2 + self.epsilon * (1 - pt2)).mean()
        # KL Consistency (R-Drop)
        kl = compute_kl_loss(logits1, logits2)
        return (poly1 + poly2) / 2 + self.alpha * kl

# ─────────────────────────────────────────────────────────
# ARCHITECTURE (ATTENTION POOLING + QUAD + 10-MSD)
# ─────────────────────────────────────────────────────────
class OmegaModel(nn.Module):
    def __init__(self, model_name="UBC-NLP/MARBERT"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["query", "value"], lora_dropout=0.1, bias="none")
        self.bert = get_peft_model(self.bert, lora_config)
        
        # Attention Pooling
        self.att_query = nn.Linear(768, 1)
        self.classifier = nn.Linear(768 * 4, 7) # Quad Pooling 
        self.dropouts = nn.ModuleList([nn.Dropout(0.3) for _ in range(5)]) # Reduced for R-Drop stability
        
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lh = out.last_hidden_state # [B, L, 768]
        
        # 1. Attention Pooling
        att_weights = torch.softmax(self.att_query(lh).squeeze(-1), dim=-1) # [B, L]
        att_p = torch.sum(lh * att_weights.unsqueeze(-1), 1) # [B, 768]
        
        # 2. Quad Components
        cls_t = lh[:, 0, :]
        mask = attention_mask.unsqueeze(-1).expand(lh.size()).float()
        mean_p = torch.sum(lh * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        std_p = torch.sqrt(torch.sum((lh - mean_p.unsqueeze(1))**2 * mask, 1) / torch.clamp(mask.sum(1), min=1e-9) + 1e-9)
        
        combined = torch.cat([att_p, mean_p, cls_t, std_p], dim=1)
        logits = torch.mean(torch.stack([self.classifier(d(combined)) for d in self.dropouts]), dim=0)
        return logits

def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    split_dir = Path("/content/drive/MyDrive/Thesis Project/data/processed/splits/final_sanitized")
    tr_df, va_df, te_df = pd.read_csv(split_dir / "train.csv"), pd.read_csv(split_dir / "val.csv"), pd.read_csv(split_dir / "test.csv")
    pool_df = pd.concat([tr_df, va_df]).reset_index(drop=True)
    LID = {'Anger':0, 'Disgust':1, 'Fear':2, 'Happiness':3, 'Neutral':4, 'Sadness':5, 'Surprise':6}
    
    texts, labels = pool_df['transcript'].values, np.array([LID[e] for e in pool_df['emotion_final']])
    te_labels = np.array([LID[e] for e in te_df['emotion_final']])
    tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")
    te_loader = DataLoader(TextDataset(te_df['transcript'].values, te_labels, tokenizer), batch_size=16)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rolling_probs, best_val_f1s, best_val_accs = [], [], []

    print("\n" + "="*60 + "\n🚀 V29: THE OMEGA POINT (R-DROP + ATTENTION)\n" + "="*60)
    
    for fold, (t_idx, v_idx) in enumerate(skf.split(texts, labels)):
        print(f"\n📂 FOLD {fold} PREPARATION...")
        t_loader, v_loader = DataLoader(TextDataset(texts[t_idx], labels[t_idx], tokenizer), batch_size=16, shuffle=True), DataLoader(TextDataset(texts[v_idx], labels[v_idx], tokenizer), batch_size=16)
        
        model = OmegaModel().to(device)
        optimizer = torch.optim.AdamW([
            {'params': [p for n, p in model.named_parameters() if "classifier" in n or "att" in n], 'lr': 1e-3},
            {'params': [p for n, p in model.named_parameters() if "bert" in n], 'lr': 2e-5}
        ], weight_decay=0.01)
        
        num_epochs = 35
        total_steps = len(t_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
        
        criterion = OmegaLoss()
        best_f1, best_acc, patience, patience_counter = 0, 0, 8, 0
        path = f"best_v29_fold_{fold}.pt"
        
        for epoch in range(1, num_epochs + 1):
            model.train()
            tr_acc_sum, total = 0, 0
            for batch_data, batch_labels in tqdm(t_loader, desc=f"   E{epoch}", leave=False):
                optimizer.zero_grad()
                ids, mask, targets = batch_data['input_ids'].to(device), batch_data['attention_mask'].to(device), batch_labels.to(device)
                
                # R-Drop: Forward Twice
                logits1 = model(ids, mask)
                logits2 = model(ids, mask)
                loss = criterion(logits1, logits2, targets)
                
                loss.backward(); optimizer.step(); scheduler.step()
                tr_acc_sum += (torch.argmax(logits1, 1) == targets).sum().item(); total += len(targets)
                
            model.eval()
            p, t = [], []
            with torch.no_grad():
                for batch_data, batch_labels in v_loader:
                    logits = model(batch_data['input_ids'].to(device), batch_data['attention_mask'].to(device))
                    p.extend(torch.argmax(logits, 1).cpu().numpy()); t.extend(batch_labels.numpy())
            
            v_acc, v_f1 = accuracy_score(t, p), f1_score(t, p, average='macro')
            print(f"   📈 E{epoch} | TrAcc: {tr_acc_sum/total:.4f} | VAcc: {v_acc:.4f} | VF1: {v_f1:.4f}")
            
            if v_f1 > best_f1:
                best_f1, best_acc = v_f1, v_acc; torch.save(model.state_dict(), path); patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience: break
        
        best_val_f1s.append(best_f1); best_val_accs.append(best_acc)
        model.load_state_dict(torch.load(path))
        model.eval()
        fold_probs = []
        with torch.no_grad():
            for batch_data, _ in te_loader:
                logits = model(batch_data['input_ids'].to(device), batch_data['attention_mask'].to(device))
                fold_probs.append(F.softmax(logits, dim=-1).cpu().numpy())
        rolling_probs.append(np.vstack(fold_probs))
        print(f"   🔥 ROLLING TEST ACC: {accuracy_score(te_labels, np.argmax(np.mean(rolling_probs, axis=0), axis=1)):.4f}")

    print("\n\n" + "="*60 + "\n🏁 FINAL PHOSPHOROUS THESIS REPORT (V29 OMEGA)\n" + "="*60)
    final_preds = np.argmax(np.mean(rolling_probs, axis=0), axis=1)
    print(f"📈 MEAN CV VAL ACC      : {np.mean(best_val_accs):.4f}")
    print(f"📈 MEAN CV VAL F1       : {np.mean(best_val_f1s):.4f}")
    print(f"🎯 FINAL ENSEMBLE TEST ACC : {accuracy_score(te_labels, final_preds):.4f}")
    print(f"🧪 FINAL ENSEMBLE MACRO F1: {f1_score(te_labels, final_preds, average='macro'):.4f}\n" + "-"*60)
    print(classification_report(te_labels, final_preds, target_names=list(LID.keys()), zero_division=0))
    print("="*60)

if __name__ == "__main__": main()
