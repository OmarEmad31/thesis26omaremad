"""
Egyptian Arabic Text SER — LoRA Anti-Overfitting Suite (v13)
===========================================================
Integrates Parameter-Efficient Fine-Tuning (PEFT/LoRA) to constrain
the 160M MARBERT parameters into a 2M adapter, mathematically
preventing 100% training memorization on small datasets.
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
from transformers import AutoTokenizer, set_seed
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold

from peft import LoraConfig, get_peft_model
from src.text_baseline.model import MARBERTWithMultiSampleDropout

# ─────────────────────────────────────────────────────────
# PREPROCESSING & DATA
# ─────────────────────────────────────────────────────────
def clean_arabic_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    text = re.sub(r'[أإآ]', 'ا', text)
    text = re.sub(r'\u0640', '', text)
    text = re.sub(r'(.)\1+', r'\1\1', text)
    return text.strip()

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        cleaned_texts = [clean_arabic_text(t) for t in texts]
        self.enc = tokenizer(list(cleaned_texts), truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.enc.items()}, torch.tensor(self.labels[idx], dtype=torch.long)

def scl_loss(hidden, labels, temp=0.1):
    features = F.normalize(hidden, p=2, dim=1)
    sim = torch.matmul(features, features.T) / temp
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().cuda()
    mask *= (1 - torch.eye(labels.size(0), device="cuda"))
    valid = mask.sum(1) > 0
    if not valid.any(): return torch.tensor(0.0).cuda()
    
    log_p = (sim - torch.max(sim, 1, True)[0].detach()) - torch.log(torch.exp(sim-torch.max(sim,1,True)[0].detach()).sum(1, True) + 1e-8)
    loss = - (mask[valid] * log_p[valid]).sum(1) / (mask[valid].sum(1) + 1e-8)
    return loss.mean()

# ─────────────────────────────────────────────────────────
# ENGINE
# ─────────────────────────────────────────────────────────

def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path("/content/drive/MyDrive/Thesis Project")
    split_dir = root / "data/processed/splits/final_sanitized"
    
    # 1. USE AUGMENTED DATA
    tr_df = pd.read_csv(split_dir / "train_augmented.csv") 
    va_df = pd.read_csv(split_dir / "val.csv")   
    te_df = pd.read_csv(split_dir / "test.csv")  
    
    pool_df = pd.concat([tr_df, va_df]).reset_index(drop=True)
    LID = {'Anger':0, 'Disgust':1, 'Fear':2, 'Happiness':3, 'Neutral':4, 'Sadness':5, 'Surprise':6}
    
    texts = pool_df['transcript'].values
    labels = np.array([LID[e] for e in pool_df['emotion_final']])
    
    tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    model_paths = []
    
    for fold, (t_idx, v_idx) in enumerate(skf.split(texts, labels)):
        print(f"\n{'='*40}\n🚀 TRAINING FOLD {fold} (LoRA PEFT)\n{'='*40}")
        
        t_loader = DataLoader(TextDataset(texts[t_idx], labels[t_idx], tokenizer), batch_size=16, shuffle=True)
        v_loader = DataLoader(TextDataset(texts[v_idx], labels[v_idx], tokenizer), batch_size=16)

        id2label = {v: k for k, v in LID.items()}
        
        # Load Native Custom Model
        model = MARBERTWithMultiSampleDropout(
            "UBC-NLP/MARBERT", 
            num_labels=7, 
            id2label=id2label, 
            label2id=LID,
            dropout_rate=0.3
        ).to(device)
        
        # 2. INJECT LoRA (r=16) directly into the BERT backbone
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none"
        )
        model.bert = get_peft_model(model.bert, lora_config)
        
        # Print Trainable Parameters to verify 2M memory
        trainable_params, all_param = model.bert.get_nb_trainable_parameters()
        print(f"✅ LoRA Active: {trainable_params:,} parameters training out of {all_param:,} ({100 * trainable_params / all_param:.2f}%)")
        
        # Ensure our custom classifier head is also trainable
        for name, param in model.classifier.named_parameters():
            param.requires_grad = True
            
        # 3. LOWER LEARNING RATE (5e-5)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5, weight_decay=0.01)
        
        # 4. LABEL SMOOTHING 0.1
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        best_f1 = 0
        path = f"best_lora_fold_{fold}.pt"
        
        # 5. EARLY STOPPING
        patience = 5
        patience_counter = 0
        
        for epoch in range(1, 20): # Increased max to 20 since Early Stopping will handle the exit
            model.train()
            train_loss = 0.0
            tr_p, tr_t = [], []
            
            for batch_data, batch_labels in tqdm(t_loader, desc=f"Epoch {epoch}", leave=False):
                batch = {k: v.to(device) for k, v in batch_data.items()}
                targets = batch_labels.to(device)
                optimizer.zero_grad()
                
                out = model(**batch)
                
                ce_loss = criterion(out.logits, targets)
                hidden = out.hidden_states[-1][:, 0, :] 
                con_loss = scl_loss(hidden, targets)
                
                loss = ce_loss + (0.1 * con_loss) 
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                tr_p.extend(torch.argmax(out.logits, 1).cpu().numpy())
                tr_t.extend(targets.cpu().numpy())
                
            model.eval()
            p, t = [], []
            with torch.no_grad():
                for batch_data, batch_labels in v_loader:
                    batch = {k: v.to(device) for k, v in batch_data.items()}
                    p.extend(torch.argmax(model(**batch).logits, 1).cpu().numpy())
                    t.extend(batch_labels.numpy())
            
            acc = accuracy_score(t, p)
            f1 = f1_score(t, p, average='macro')
            tr_acc = accuracy_score(tr_t, tr_p)
            avg_loss = train_loss / len(t_loader)
            
            print(f"📈 Epoch {epoch} | Train Loss: {avg_loss:.4f} | Train Acc: {tr_acc:.4f} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"🛑 Early stopping triggered at Epoch {epoch}. Best Val F1 was {best_f1:.4f}")
                    break
                
        print(f"✅ Fold {fold} Final Best Val F1: {best_f1:.4f}")
        model_paths.append(path)

    # ─────────────────────────────────────────────────────────
    # ENSEMBLE REPORT ON THE SECRET 44 TEST SAMPLES
    # ─────────────────────────────────────────────────────────
    print("\n" + "="*40 + "\n🏁 FINAL PHASE 4 ENSEMBLE REPORT (44 TEST SAMPLES)\n" + "="*40)
    
    te_labels = np.array([LID[e] for e in te_df['emotion_final']])
    te_loader = DataLoader(TextDataset(te_df['transcript'].values, te_labels, tokenizer), batch_size=16)
    
    ensemble_probs = []
    
    for path in model_paths:
        model = MARBERTWithMultiSampleDropout("UBC-NLP/MARBERT", num_labels=7, id2label=id2label, label2id=LID).to(device)
        model.bert = get_peft_model(model.bert, lora_config)
        model.load_state_dict(torch.load(path))
        model.eval()
        
        probs = []
        with torch.no_grad():
            for batch_data, _ in te_loader:
                batch = {k: v.to(device) for k, v in batch_data.items()}
                logits = model(**batch).logits
                probs.append(F.softmax(logits, dim=-1).cpu().numpy())
        ensemble_probs.append(np.vstack(probs))
        
    avg_probs = np.mean(ensemble_probs, axis=0)
    final_preds = np.argmax(avg_probs, axis=1)
    
    print(f"\nFINAL TEST ACCURACY: {accuracy_score(te_labels, final_preds):.4f}")
    print(f"FINAL MACRO F1:      {f1_score(te_labels, final_preds, average='macro'):.4f}\n")
    print(classification_report(te_labels, final_preds, target_names=list(LID.keys())))

if __name__ == "__main__": main()
