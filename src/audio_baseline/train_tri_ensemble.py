"""
Tri-Expert Ensemble Trainer.
Uses PCA + XGBoost with GridSearchCV to target 58%+ on individual Audio.

Run: python -m src.audio_baseline.train_tri_ensemble
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
from pathlib import Path
import random, warnings
warnings.filterwarnings("ignore")

import sys
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.audio_baseline import config

def main():
    random.seed(config.SEED)
    np.random.seed(config.SEED)

    print(f"\n{'='*60}")
    print("🏆  TRI-EXPERT ENSEMBLE TRAINER  (Target: 58-60%)")
    print("    Arabic-Native + SER-Expert + Acoustic Physics")
    print(f"{'='*60}")

    data_path = config.SER_FEATURES_DIR / "tri_expert_dataset.pkl"
    if not data_path.exists():
        print(f"[FATAL] Features not found at {data_path}")
        print("Run: python -m src.audio_baseline.extract_tri_expert  first!")
        return

    data = pd.read_pickle(data_path)
    df = data["metadata"]
    
    # Concatenate all features
    X_all = np.concatenate([
        data["feat_arabic"], 
        data["feat_ser"], 
        data["feat_physics"]
    ], axis=1)
    
    label2id = {l: i for i, l in enumerate(sorted(df["emotion_final"].unique()))}
    id2label = {i: l for l, i in label2id.items()}
    y = np.array([label2id[l] for l in df["emotion_final"]])
    
    train_mask = (df["split"] != "test").values
    test_mask  = (df["split"] == "test").values
    
    X_tv, y_tv = X_all[train_mask], y[train_mask]
    X_te, y_te = X_all[test_mask],  y[test_mask]
    
    print(f"   Train/Val: {len(X_tv)}  |  Test: {len(X_te)}")
    print(f"   Original Dimensionality: {X_all.shape[1]}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    fold_results = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_tv, y_tv)):
        print(f"\n--- FOLD {fold_idx+1} / 5 ---")
        
        X_tr, y_tr = X_tv[tr_idx], y_tv[tr_idx]
        X_va, y_va = X_tv[va_idx], y_tv[va_idx]

        # 1. Scaling
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)
        X_te_s = scaler.transform(X_te)

        # 2. PCA (Crucial to prevent overfitting on 2300+ features)
        # We find components to keep 99% variance
        pca = PCA(n_components=0.99, random_state=config.SEED)
        X_tr_p = pca.fit_transform(X_tr_s)
        X_va_p = pca.transform(X_va_s)
        X_te_p = pca.transform(X_te_s)
        print(f"   PCA: {X_tr_s.shape[1]}-D -> {X_tr_p.shape[1]}-D (99% variance)")

        # 3. XGBoost Hyperparameter Search
        # Light grid to keep it fast but effective
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8]
        }
        
        xgb = XGBClassifier(
            use_label_encoder=False, 
            eval_metric='mlogloss',
            random_state=config.SEED,
            tree_method='hist' # Efficient
        )
        
        grid = GridSearchCV(xgb, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
        grid.fit(X_tr_p, y_tr)
        
        best_clf = grid.best_estimator_
        print(f"   Best Params: {grid.best_params_}")

        va_acc = accuracy_score(y_va, best_clf.predict(X_va_p))
        va_f1  = f1_score(y_va,  best_clf.predict(X_va_p),  average="macro")
        te_acc = accuracy_score(y_te, best_clf.predict(X_te_p))
        te_f1  = f1_score(y_te,  best_clf.predict(X_te_p),  average="macro")

        print(f"   Val Acc: {va_acc:.4f}  Val F1: {va_f1:.4f}")
        print(f"   Test Acc: {te_acc:.4f}  Test F1: {te_f1:.4f}")
        
        fold_results.append({"val_f1": va_f1, "test_acc": te_acc, "test_f1": te_f1})

    print(f"\n{'='*60}")
    print(f"🏆  TRI-EXPERT ENSEMBLE — FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  Mean Val  F1 : {np.mean([r['val_f1']  for r in fold_results])*100:.2f}%")
    print(f"  Mean Test Acc: {np.mean([r['test_acc'] for r in fold_results])*100:.2f}%")
    print(f"  Mean Test F1 : {np.mean([r['test_f1']  for r in fold_results])*100:.2f}%")
    print(f"{'='*60}")

    print("\n📊  Per-Class Analysis (Last Fold):")
    print(classification_report(y_te, best_clf.predict(X_te_p), target_names=[id2label[i] for i in range(len(label2id))]))

if __name__ == "__main__":
    main()
