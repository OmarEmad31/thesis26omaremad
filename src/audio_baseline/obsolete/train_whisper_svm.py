import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
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
        
        # 1. PCA: Compress noisy 512-D space into dense 128-D manifold
        # This dramatically improves SVM separability on small datasets by removing low-variance noise dimensions
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)
        X_test_scaled = scaler.transform(X_test_raw)

        n_components = min(128, X_tr_scaled.shape[0] - 1, X_tr_scaled.shape[1])
        pca = PCA(n_components=n_components, random_state=config.SEED)
        X_tr_pca = pca.fit_transform(X_tr_scaled)
        X_va_pca = pca.transform(X_va_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        print(f"  PCA: {X_tr_scaled.shape[1]}D → {n_components}D ({pca.explained_variance_ratio_.sum()*100:.1f}% variance retained)")

        # 2. SVM GridSearch: Tune C and gamma on inner cross-validation
        # C=1 is far too conservative on 512-dimensional embeddings with 600 samples
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto'],
        }
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.SEED)
        svm_base = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=config.SEED)
        grid_search = GridSearchCV(svm_base, param_grid, cv=inner_cv, scoring='f1_macro', n_jobs=-1)
        grid_search.fit(X_tr_pca, y_tr)
        
        svm = grid_search.best_estimator_
        print(f"  Best SVM params: C={grid_search.best_params_['C']}, gamma={grid_search.best_params_['gamma']}")
        
        # 4. Metrics Logging
        tr_preds = svm.predict(X_tr_pca)
        va_preds = svm.predict(X_va_pca)
        test_preds = svm.predict(X_test_pca)
        
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
