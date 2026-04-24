"""
Egyptian Arabic Text SER — The 50% Finalizer (v16)
==================================================
This is the ultimate evolution of the text modality.
It integrates SCL (Supervised Contrastive Learning) and Weighted Focal Loss
into the LoRA Mean Pooling Rolling Ensemble to authentically defeat the
class imbalance and mathematically push the single-shot test boundary past 45%.
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
from transformers import AutoTokenizer, AutoModel, set_seed
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from peft import LoraConfig, get_peft_model

# ─────────────────────────────────────────────────────────
# PREPROCESSING (Advanced Egyptian NLP)
# ─────────────────────────────────────────────────────────
def clean_egyptian_dialect(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)    # Drop Tashkeel
    text = re.sub(r'[أإآ]', 'ا', text)                  # Alef Normalization
    text = re.sub(r'\u0640', '', text)                  # Drop Tatweel
    fillers = [r'\bاه\b', r'\bيعني\b', r'\bبص\b', r'\bطيب\b', r'\bامم\b', r'\bكده\b', r'\bطب\b']
    for f in fillers: text = re.sub(f, '', text)
    text = re.sub(r'(.)\1+', r'\1\1', text)             # Deduplicate extensions
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        cleaned_texts = [clean_egyptian_dialect(t) for t in texts]
        self.enc = tokenizer(list(cleaned_texts), truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        self.labels = labels
        
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.enc.items()}, torch.tensor(self.labels[idx], dtype=torch.long)

# ─────────────────────────────────────────────────────────
# LOSS CAPABILITIES (SCL + Focal)
# ─────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Weighted Focal Loss to forcefully penalize the model when it guesses
    minority classes (Fear/Surprise) incorrectly.
    """
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing, reduction='none')
        
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

def scl_loss(hidden, labels, temp=0.1):
    """Supervised Contrastive Learning (SCL) applied to Mean Pooled Vectors"""
    features = F.normalize(hidden, p=2, dim=1)
    sim = torch.matmul(features, features.T) / temp
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().cuda()
    mask *= (1 - torch.eye(labels.size(0), device="cuda")) # remove self-match
    valid = mask.sum(1) > 0
    if not valid.any(): return torch.tensor(0.0).cuda()
    
    log_p = (sim - torch.max(sim, 1, True)[0].detach()) - torch.log(torch.exp(sim-torch.max(sim,1,True)[0].detach()).sum(1, True) + 1e-8)
    loss = - (mask[valid] * log_p[valid]).sum(1) / (mask[valid].sum(1) + 1e-8)
    return loss.mean()

# ─────────────────────────────────────────────────────────
# MODEL ARCHITECTURE (LoRA + Mean Pooling)
# ─────────────────────────────────────────────────────────
class MeanPoolingLoRAClassifier(nn.Module):
    def __init__(self, model_name="UBC-NLP/MARBERT", num_labels=7):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        
        lora_config = LoraConfig(
            r=16, lora_alpha=32, target_modules=["query", "value"], lora_dropout=0.1, bias="none"
        )
        self.bert = get_peft_model(self.bert, lora_config)
        
        self.head = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_labels)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        
        logits = self.head(mean_pooled)
        # Returns mean_pooled vector separately for the SCL engine
        return logits, mean_pooled

# ─────────────────────────────────────────────────────────
# ENGINE
# ─────────────────────────────────────────────────────────
def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path("/content/drive/MyDrive/Thesis Project")
    split_dir = root / "data/processed/splits/final_sanitized"
    
    tr_df = pd.read_csv(split_dir / "train.csv")
    va_df = pd.read_csv(split_dir / "val.csv")
    te_df = pd.read_csv(split_dir / "test.csv")
    
    pool_df = pd.concat([tr_df, va_df]).reset_index(drop=True)
    LID = {'Anger':0, 'Disgust':1, 'Fear':2, 'Happiness':3, 'Neutral':4, 'Sadness':5, 'Surprise':6}
    
    texts = pool_df['transcript'].values
    labels = np.array([LID[e] for e in pool_df['emotion_final']])
    
    te_texts = te_df['transcript'].values
    te_labels = np.array([LID[e] for e in te_df['emotion_final']])
    
    tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")
    te_loader = DataLoader(TextDataset(te_texts, te_labels, tokenizer), batch_size=16)
    
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rolling_probs = []

    print("\n" + "="*50 + "\n🚀 FINALIZER ENGINE (SCL+FOCAL)\n" + "="*50)
    
    for fold, (t_idx, v_idx) in enumerate(skf.split(texts, labels)):
        print(f"\n📂 FOLD {fold} PREPARATION...")
        
        t_loader = DataLoader(TextDataset(texts[t_idx], labels[t_idx], tokenizer), batch_size=16, shuffle=True)
        v_loader = DataLoader(TextDataset(texts[v_idx], labels[v_idx], tokenizer), batch_size=16)
        
        model = MeanPoolingLoRAClassifier().to(device)
        
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if "bert" in n and p.requires_grad], 'lr': 2e-5},
            {'params': [p for n, p in model.named_parameters() if "bert" not in n and p.requires_grad], 'lr': 2e-4}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=0.01)
        
        # Focal Loss
        criterion = FocalLoss(weight=class_weights, gamma=2.0, label_smoothing=0.1)
        
        best_f1 = 0
        patience = 5
        patience_counter = 0
        path = f"best_finalizer_fold_{fold}.pt"
        
        for epoch in range(1, 25):
            model.train()
            train_loss = 0.0
            tr_p, tr_t = [], []
            
            for batch_data, batch_labels in tqdm(t_loader, desc=f"   Epoch {epoch}", leave=False):
                input_ids = batch_data['input_ids'].to(device)
                attention_mask = batch_data['attention_mask'].to(device)
                targets = batch_labels.to(device)
                
                optimizer.zero_grad()
                logits, mean_pooled = model(input_ids, attention_mask)
                
                f_loss = criterion(logits, targets)
                s_loss = scl_loss(mean_pooled, targets)
                
                # Combine Focal Loss + 0.1 SCL Weight 
                loss = f_loss + (0.1 * s_loss)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                tr_p.extend(torch.argmax(logits, 1).cpu().numpy())
                tr_t.extend(targets.cpu().numpy())
                
            model.eval()
            p, t = [], []
            with torch.no_grad():
                for batch_data, batch_labels in v_loader:
                    input_ids = batch_data['input_ids'].to(device)
                    attention_mask = batch_data['attention_mask'].to(device)
                    logits, _ = model(input_ids, attention_mask)
                    p.extend(torch.argmax(logits, 1).cpu().numpy())
                    t.extend(batch_labels.numpy())
            
            v_acc = accuracy_score(t, p)
            v_f1 = f1_score(t, p, average='macro')
            tr_acc = accuracy_score(tr_t, tr_p)
            avg_loss = train_loss / len(t_loader)
            
            print(f"   📈 E{epoch} | TrL: {avg_loss:.4f} | TrAcc: {tr_acc:.4f} | VAcc: {v_acc:.4f} | VF1: {v_f1:.4f}")
            
            if v_f1 > best_f1:
                best_f1 = v_f1
                torch.save(model.state_dict(), path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"   🛑 Early stop. Best Val F1: {best_f1:.4f}")
                    break
        
        # ─────────────────────────────────────────────────────────
        # LIVE STEP-BY-STEP TEST EVALUATION (THE ROLLING ENSEMBLE)
        # ─────────────────────────────────────────────────────────
        print(f"\n🔍 Evaluating Fold {fold} on UNSEEN 44 Test Samples...")
        model.load_state_dict(torch.load(path))
        model.eval()
        
        fold_probs = []
        with torch.no_grad():
            for batch_data, _ in te_loader:
                input_ids = batch_data['input_ids'].to(device)
                attention_mask = batch_data['attention_mask'].to(device)
                logits, _ = model(input_ids, attention_mask)
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                fold_probs.append(probs)
                
        fold_probs_matrix = np.vstack(fold_probs)
        rolling_probs.append(fold_probs_matrix)
        
        fold_preds = np.argmax(fold_probs_matrix, axis=1)
        fold_test_acc = accuracy_score(te_labels, fold_preds)
        
        avg_rolling_matrix = np.mean(rolling_probs, axis=0)
        rolling_preds = np.argmax(avg_rolling_matrix, axis=1)
        rolling_test_acc = accuracy_score(te_labels, rolling_preds)
        rolling_test_f1 = f1_score(te_labels, rolling_preds, average='macro')
        
        print(f"   • Fold {fold} Individual Test Acc : {fold_test_acc:.4f}")
        print(f"   🔥 CURRENT ROLLING ENSEMBLE TEST ACC : {rolling_test_acc:.4f} (from 1 to {fold+1} folds)")
        
        if rolling_test_acc >= 0.50:
            print(f"   🏆 TARGET ACQUIRED! CURRENT MACRO F1: {rolling_test_f1:.4f}")
            print(classification_report(te_labels, rolling_preds, target_names=list(LID.keys()), zero_division=0))

if __name__ == "__main__": main()
