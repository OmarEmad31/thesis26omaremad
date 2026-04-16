import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
import random

import sys
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.audio_baseline import config
from src.text_baseline.train import SupervisedContrastiveLoss # Reuse the exact same SCL Loss from text

# ==============================================================================
# MLP ARCHITECTURE
# ==============================================================================
class WhisperMLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_labels=7):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(0.3)
        
        self.classifier = nn.Linear(hidden_dim // 2, num_labels)

    def forward(self, x):
        features = F.gelu(self.bn1(self.fc1(x)))
        features = self.dropout1(features)
        
        # We extract this embedded features vector precisely for the SCL logic to cluster!
        pooled_output = F.gelu(self.bn2(self.fc2(features)))
        pooled_output = self.dropout2(pooled_output)
        
        logits = self.classifier(pooled_output)
        return logits, pooled_output

# ==============================================================================
# MAIN RUNNER
# ==============================================================================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    set_seed(config.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("🚀 Initiating Minimalist PyTorch Fast-Track Pipeline...")
    
    # 1. Load the Offline Cached Database instantly
    data_path = config.OFFLINE_FEATURES_DIR / "whisper_dataset.pkl"
    if not data_path.exists():
        print(f"[FATAL ERROR] Offline features not found at {data_path}")
        print("You must run 'python -m src.audio_baseline.extract_whisper_features' first!")
        return
        
    df = pd.read_pickle(data_path)
    print(f"Loaded {len(df)} pre-computed Whisper arrays natively.")

    # Sort identically to text baseline for stratified 5-fold parity
    label2id = {lbl: i for i, lbl in enumerate(sorted(df["emotion_final"].unique()))}
    id2label = {i: lbl for lbl, i in label2id.items()}
    num_labels = len(label2id)

    X = np.stack(df["whisper_features"].values)
    y = np.array([label2id[lbl] for lbl in df["emotion_final"]])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    
    # Track cross-validation scores
    all_fold_f1 = []

    for fold_idx, (train_index, val_index) in enumerate(skf.split(X, y)):
        print(f"\n========================================")
        print(f"🔥 FOLD {fold_idx + 1} / 5")
        print(f"========================================")
        
        X_train, X_val = torch.tensor(X[train_index], dtype=torch.float32), torch.tensor(X[val_index], dtype=torch.float32)
        y_train, y_val = torch.tensor(y[train_index], dtype=torch.long), torch.tensor(y[val_index], dtype=torch.long)
        
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=config.BATCH_SIZE, shuffle=False)

        # Handle class imbalance explicitly
        class_w = compute_class_weight("balanced", classes=np.unique(y_train.numpy()), y=y_train.numpy())
        ce_loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_w, dtype=torch.float32).to(device))
        scl_loss_fn = SupervisedContrastiveLoss(temperature=config.SCL_TEMP)

        model = WhisperMLP(input_dim=X.shape[1], num_labels=num_labels).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

        best_f1 = 0
        patience_counter = 0

        # Sub-10 Second Training Loop!
        for epoch in range(1, config.NUM_EPOCHS + 1):
            model.train()
            epoch_loss = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                
                logits, pooled_out = model(batch_x)
                
                # Hybrid Loss (Identical to Text Baseline)
                loss_ce = ce_loss_fn(logits, batch_y)
                loss_scl = scl_loss_fn(pooled_out, batch_y)
                loss = (1 - config.SCL_WEIGHT) * loss_ce + config.SCL_WEIGHT * loss_scl
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            model.eval()
            val_preds, val_truth = [], []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    logits, _ = model(batch_x)
                    val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                    val_truth.extend(batch_y.numpy())
                    
            val_acc = accuracy_score(val_truth, val_preds)
            val_f1 = f1_score(val_truth, val_preds, average="macro")
            
            # Since epochs run in 0.1 seconds, we print every 10 epochs to keep the terminal perfectly clean
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:2d}/{config.NUM_EPOCHS} | Train Loss: {epoch_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= config.EARLY_STOP_PATIENCE:
                print(f"🛑 Early stopping triggered at Epoch {epoch}! Best F1: {best_f1:.4f}")
                break
                
        all_fold_f1.append(best_f1)
        print(f"✅ Fold {fold_idx + 1} Best Marco F1: {best_f1 * 100:.2f}%")

    mean_f1 = np.mean(all_fold_f1)
    print(f"\n========================================")
    print(f"🏆 ALL FOLDS COMPLETE. Mean F1 Macro: {mean_f1 * 100:.2f}%")
    print(f"========================================")

if __name__ == "__main__":
    main()
