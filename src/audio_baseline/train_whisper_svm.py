import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import random

import sys
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.audio_baseline import config
import warnings
warnings.filterwarnings("ignore")

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def main():
    set_seed(config.SEED)
    
    print("🚀 Initiating RBF Support Vector Machine (Arabic Whisper Engine)...")
    
    data_path = config.OFFLINE_FEATURES_DIR / "whisper_dataset.pkl"
    if not data_path.exists():
        print(f"[FATAL ERROR] Arabic Whisper features not found at {data_path}")
        return
        
    df = pd.read_pickle(data_path)
    
    # ISOLATE TEST SET
    test_df = df[df['split'] == 'test']
    train_val_df = df[df['split'] != 'test'].reset_index(drop=True)
    
    print(f"Loaded {len(train_val_df)} K-Fold Training arrays | Loaded {len(test_df)} Blind Test arrays.")

    label2id = {lbl: i for i, lbl in enumerate(sorted(df["emotion_final"].unique()))}

    # Extract dense [512] embeddings matrices explicitly natively mapped by Whisper
    X_trainval = np.stack(train_val_df["whisper_features"].values)
    y_trainval = np.array([label2id[lbl] for lbl in train_val_df["emotion_final"]])
    
    X_test_raw = np.stack(test_df["whisper_features"].values)
    y_test = np.array([label2id[lbl] for lbl in test_df["emotion_final"]])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    metrics_log = []

    for fold_idx, (train_index, val_index) in enumerate(skf.split(X_trainval, y_trainval)):
        print(f"\n========================================")
        print(f"🔥 WHISPER-SVM FOLD {fold_idx + 1} / 5")
        print(f"========================================")
        
        # Split Data
        X_tr, X_va = X_trainval[train_index], X_trainval[val_index]
        y_tr, y_va = y_trainval[train_index], y_trainval[val_index]
        
        # 1. Feature Scaling (Acoustic SVM hyperplanes shatter if scales are uneven)
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)
        X_test_scaled = scaler.transform(X_test_raw)

        # 2. Support Vector Machine Setup
        # probability=True maintains Multimodal Late-Fusion Output Logging limits
        svm = SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True, random_state=config.SEED)
        
        # 3. Fit Model (Instant calculation of Arabic structural geometries)
        svm.fit(X_tr_scaled, y_tr)
        
        # 4. Metrics Logging
        tr_preds = svm.predict(X_tr_scaled)
        va_preds = svm.predict(X_va_scaled)
        test_preds = svm.predict(X_test_scaled)
        
        train_acc = accuracy_score(y_tr, tr_preds)
        train_f1 = f1_score(y_tr, tr_preds, average="macro")
        
        val_acc = accuracy_score(y_va, va_preds)
        val_f1 = f1_score(y_va, va_preds, average="macro")
        
        test_acc = accuracy_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds, average="macro")

        print(f"Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
        print(f"  Val Acc: {val_acc:.4f} |   Val F1: {val_f1:.4f}")
        print(f" Test Acc: {test_acc:.4f} |  Test F1: {test_f1:.4f}")
        
        metrics_log.append({
            "val_acc": val_acc, "val_f1": val_f1,
            "test_acc": test_acc, "test_f1": test_f1
        })

    # Massive Summary
    mean_val_f1 = np.mean([m['val_f1'] for m in metrics_log])
    mean_test_acc = np.mean([m['test_acc'] for m in metrics_log])
    mean_test_f1 = np.mean([m['test_f1'] for m in metrics_log])
    
    print(f"\n========================================")
    print(f"🏆 ARABIC WHISPER-SVM VALIDATED ON TEST DATASET 🏆")
    print(f"Mean Validation F1: {mean_val_f1 * 100:.2f}%")
    print(f"Mean Test Accuracy: {mean_test_acc * 100:.2f}%")
    print(f"Mean Test F1 Macro: {mean_test_f1 * 100:.2f}%")
    print(f"========================================")

if __name__ == "__main__":
    main()
