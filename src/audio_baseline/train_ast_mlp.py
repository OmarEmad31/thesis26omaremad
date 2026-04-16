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
import warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# SUPERVISED CONTRASTIVE LOSS
# ==============================================================================
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        similarity = torch.matmul(features, features.T) / self.temperature
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        batch_size = labels.size(0)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=mask.device)
        mask = mask * logits_mask
        
        max_sim, _ = torch.max(similarity, dim=1, keepdim=True)
        sim_stable = similarity - max_sim.detach()
        
        exp_sim = torch.exp(sim_stable) * logits_mask
        log_prob = sim_stable - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        
        valid_anchors = mask.sum(1) > 0  
        if valid_anchors.any():
            mean_log_prob_pos = (mask[valid_anchors] * log_prob[valid_anchors]).sum(1) / (mask[valid_anchors].sum(1) + 1e-8)
            return -mean_log_prob_pos.mean()
        return torch.tensor(0.0, device=features.device, requires_grad=True)

# ==============================================================================
# MULTI-SAMPLE DROPOUT (Upgraded Neural Stabilization)
# ==============================================================================
class MultiSampleDropout(nn.Module):
    def __init__(self, in_features, out_features, num_samples=5, p=0.4):
        super().__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(p) for _ in range(num_samples)])
        self.classifier = nn.Linear(in_features, out_features)

    def forward(self, x):
        # Passes the identical features through 5 parallel mutated dropouts and averages them
        # mathematically guaranteeing regularization.
        out = None
        for dropout in self.dropouts:
            if out is None:
                out = self.classifier(dropout(x))
            else:
                out += self.classifier(dropout(x))
        return out / len(self.dropouts)

class AST_MLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_labels=7):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        
        # Improvement 1: Multi-Sample Dropout Injection
        self.msd = MultiSampleDropout(hidden_dim // 2, num_labels, num_samples=5, p=0.4)

    def forward(self, x):
        features = F.gelu(self.bn1(self.fc1(x)))
        features = self.dropout1(features)
        
        pooled_output = F.gelu(self.bn2(self.fc2(features)))
        logits = self.msd(pooled_output)
        
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

def evaluate_metrics(model, dataloader, device):
    all_preds, all_truth = [], []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            logits, _ = model(batch_x)
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_truth.extend(batch_y.numpy())
    acc = accuracy_score(all_truth, all_preds)
    f1 = f1_score(all_truth, all_preds, average="macro")
    return acc, f1

def main():
    set_seed(config.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("🚀 Initiating Methodologically-Upgraded AST MLP (With Multi-Sample Dropout & LR Scaling)...")
    
    data_path = config.OFFLINE_FEATURES_DIR / "ast_dataset.pkl"
    if not data_path.exists():
        print(f"[FATAL ERROR] AST Offline Vision features not found at {data_path}")
        return
        
    df = pd.read_pickle(data_path)
    
    # ISOLATE TEST SET
    test_df = df[df['split'] == 'test']
    train_val_df = df[df['split'] != 'test'].reset_index(drop=True)
    
    print(f"Loaded {len(train_val_df)} K-Fold Training arrays | Loaded {len(test_df)} Blind Test arrays.")

    label2id = {lbl: i for i, lbl in enumerate(sorted(df["emotion_final"].unique()))}
    num_labels = len(label2id)

    X_trainval = np.stack(train_val_df["ast_features"].values)
    y_trainval = np.array([label2id[lbl] for lbl in train_val_df["emotion_final"]])
    
    X_test = np.stack(test_df["ast_features"].values)
    y_test = np.array([label2id[lbl] for lbl in test_df["emotion_final"]])
    
    test_loader = DataLoader(TensorDataset(
        torch.tensor(X_test, dtype=torch.float32), 
        torch.tensor(y_test, dtype=torch.long)
    ), batch_size=config.BATCH_SIZE, shuffle=False)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    
    metrics_log = []

    for fold_idx, (train_index, val_index) in enumerate(skf.split(X_trainval, y_trainval)):
        print(f"\n========================================")
        print(f"🔥 AST FOLD {fold_idx + 1} / 5")
        print(f"========================================")
        
        X_tr, X_va = torch.tensor(X_trainval[train_index], dtype=torch.float32), torch.tensor(X_trainval[val_index], dtype=torch.float32)
        y_tr, y_va = torch.tensor(y_trainval[train_index], dtype=torch.long), torch.tensor(y_trainval[val_index], dtype=torch.long)
        
        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=config.BATCH_SIZE, shuffle=False)

        class_w = compute_class_weight("balanced", classes=np.unique(y_trainval[train_index]), y=y_trainval[train_index])
        ce_loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_w, dtype=torch.float32).to(device))
        scl_loss_fn = SupervisedContrastiveLoss(temperature=config.SCL_TEMP)

        model = AST_MLP(input_dim=X_trainval.shape[1], num_labels=num_labels).to(device)
        
        # Improvement 2: Aggressive LR Kinetics with Cosine Annealing to pierce Local Minima
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        best_f1 = 0
        best_model_weights = None
        patience = 0

        for epoch in range(1, config.NUM_EPOCHS + 1):
            model.train()
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                
                logits, pooled_out = model(batch_x)
                
                loss_ce = ce_loss_fn(logits, batch_y)
                loss_scl = scl_loss_fn(pooled_out, batch_y)
                loss = (1 - config.SCL_WEIGHT) * loss_ce + config.SCL_WEIGHT * loss_scl
                
                loss.backward()
                optimizer.step()
                
            scheduler.step()
            
            # Improvement 3: Explicit Metric Matrix evaluation per requested constraint
            train_acc, train_f1 = evaluate_metrics(model, train_loader, device)
            val_acc, val_f1 = evaluate_metrics(model, val_loader, device)
            
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:2d}/{config.NUM_EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_model_weights = model.state_dict().copy()
                patience = 0
            else:
                patience += 1
                
            if patience >= config.EARLY_STOP_PATIENCE:
                print(f"🛑 Plateau Avoidance (Early Stopping). Best Val F1 fixed at: {best_f1:.4f}")
                break
                
        # Load best weights to test blindly against the locked test set
        model.load_state_dict(best_model_weights)
        test_acc, test_f1 = evaluate_metrics(model, test_loader, device)
        
        print(f"✅ FOLD {fold_idx + 1} EXPLICIT TRUTH: Val Acc: {val_acc*100:.1f}% | Test Acc: {test_acc*100:.1f}% | Test F1 Macro: {test_f1*100:.1f}%")
        
        metrics_log.append({
            "val_acc": val_acc, "val_f1": best_f1,
            "test_acc": test_acc, "test_f1": test_f1
        })

    # Massive Summary
    mean_val_f1 = np.mean([m['val_f1'] for m in metrics_log])
    mean_test_acc = np.mean([m['test_acc'] for m in metrics_log])
    mean_test_f1 = np.mean([m['test_f1'] for m in metrics_log])
    
    print(f"\n========================================")
    print(f"🏆 AST ARCHITECTURE VALIDATED ON TEST DATASET 🏆")
    print(f"Mean Validation F1: {mean_val_f1 * 100:.2f}%")
    print(f"Mean Test Accuracy: {mean_test_acc * 100:.2f}%")
    print(f"Mean Test F1 Macro: {mean_test_f1 * 100:.2f}%")
    print(f"========================================")

if __name__ == "__main__":
    main()
