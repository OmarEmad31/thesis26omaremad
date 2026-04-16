"""
SVM classifier on top of pre-extracted SER-expert embeddings.
Run AFTER extract_ser_features.py.

Run: python -m src.audio_baseline.train_ser_svm
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
from pathlib import Path
import random

import sys
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.audio_baseline import config
import warnings
warnings.filterwarnings("ignore")

def main():
    random.seed(config.SEED)
    np.random.seed(config.SEED)

    print("🚀  SER-Expert SVM  (superb/wav2vec2-large-superb-er + RBF-SVM)")

    data_path = config.SER_FEATURES_DIR / "ser_dataset.pkl"
    if not data_path.exists():
        print(f"[FATAL] Features not found at {data_path}")
        print("Run: python -m src.audio_baseline.extract_ser_features  first!")
        return

    df = pd.read_pickle(data_path)

    test_df      = df[df["split"] == "test"]
    train_val_df = df[df["split"] != "test"].reset_index(drop=True)
    print(f"   Train/Val: {len(train_val_df)}  |  Test: {len(test_df)}")

    label2id   = {lbl: i for i, lbl in enumerate(sorted(df["emotion_final"].unique()))}
    id2label   = {i: lbl for lbl, i in label2id.items()}
    num_labels = len(label2id)
    print(f"   Labels ({num_labels}): {list(label2id.keys())}\n")

    X_tv   = np.stack(train_val_df["ser_features"].values)
    y_tv   = np.array([label2id[l] for l in train_val_df["emotion_final"]])
    X_test = np.stack(test_df["ser_features"].values)
    y_test = np.array([label2id[l] for l in test_df["emotion_final"]])

    skf          = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    fold_results = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_tv, y_tv)):
        print(f"{'='*55}")
        print(f"  FOLD {fold_idx+1} / 5")
        print(f"{'='*55}")

        X_tr, X_va = X_tv[tr_idx], X_tv[va_idx]
        y_tr, y_va = y_tv[tr_idx], y_tv[va_idx]

        # Scale
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)
        X_te_s = scaler.transform(X_test)

        # PCA: compress 1024-D → 256-D (keeps ≥95% variance, removes noise)
        n_comp = min(256, X_tr_s.shape[0] - 1, X_tr_s.shape[1])
        pca    = PCA(n_components=n_comp, random_state=config.SEED)
        X_tr_p = pca.fit_transform(X_tr_s)
        X_va_p = pca.transform(X_va_s)
        X_te_p = pca.transform(X_te_s)
        print(f"  PCA: 1024-D → {n_comp}-D  ({pca.explained_variance_ratio_.sum()*100:.1f}% variance)")

        # GridSearchCV to find optimal C and gamma
        param_grid  = {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto"]}
        inner_cv    = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.SEED)
        grid_search = GridSearchCV(
            SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=config.SEED),
            param_grid, cv=inner_cv, scoring="f1_macro", n_jobs=-1, verbose=0
        )
        grid_search.fit(X_tr_p, y_tr)
        svm = grid_search.best_estimator_
        print(f"  Best SVM: C={grid_search.best_params_['C']}  gamma={grid_search.best_params_['gamma']}")

        tr_acc  = accuracy_score(y_tr, svm.predict(X_tr_p))
        tr_f1   = f1_score(y_tr,  svm.predict(X_tr_p),  average="macro")
        va_acc  = accuracy_score(y_va, svm.predict(X_va_p))
        va_f1   = f1_score(y_va,  svm.predict(X_va_p),  average="macro")
        te_acc  = accuracy_score(y_test, svm.predict(X_te_p))
        te_f1   = f1_score(y_test, svm.predict(X_te_p), average="macro")

        print(f"\n  Train Acc: {tr_acc:.4f}  Train F1: {tr_f1:.4f}")
        print(f"    Val Acc: {va_acc:.4f}    Val F1: {va_f1:.4f}")
        print(f"   Test Acc: {te_acc:.4f}   Test F1: {te_f1:.4f}")

        fold_results.append({"val_f1": va_f1, "test_acc": te_acc, "test_f1": te_f1})

    print(f"\n{'='*55}")
    print(f"🏆  SER-Expert SVM  —  FINAL RESULTS")
    print(f"{'='*55}")
    mv  = np.mean([r['val_f1']  for r in fold_results])
    mta = np.mean([r['test_acc'] for r in fold_results])
    mtf = np.mean([r['test_f1'] for r in fold_results])
    print(f"  Mean Val  F1 : {mv  * 100:.2f}%")
    print(f"  Mean Test Acc: {mta * 100:.2f}%")
    print(f"  Mean Test F1 : {mtf * 100:.2f}%")
    print(f"{'='*55}")

    # Per-class breakdown on full test set (use last fold's SVM as representative)
    print("\n📊  Per-Class Test Report (last fold):")
    print(classification_report(
        y_test, svm.predict(X_te_p),
        target_names=[id2label[i] for i in range(num_labels)]
    ))

if __name__ == "__main__":
    main()
