"""
SVM on classical acoustic features for Egyptian Arabic SER.
Run: python -m src.audio_baseline.train_acoustic_svm
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
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

    print("🚀  Classical Acoustic SVM (MFCC + Pitch + Energy + Spectral)")

    data_path = config.SER_FEATURES_DIR / "acoustic_dataset.pkl"
    if not data_path.exists():
        print(f"[FATAL] Not found: {data_path}")
        print("Run: python -m src.audio_baseline.extract_acoustic_features  first!")
        return

    df = pd.read_pickle(data_path)
    test_df      = df[df["split"] == "test"]
    train_val_df = df[df["split"] != "test"].reset_index(drop=True)
    print(f"   Train/Val: {len(train_val_df)}  |  Test: {len(test_df)}")

    label2id   = {lbl: i for i, lbl in enumerate(sorted(df["emotion_final"].unique()))}
    id2label   = {i: lbl for lbl, i in label2id.items()}
    num_labels = len(label2id)
    print(f"   Labels ({num_labels}): {list(label2id.keys())}\n")

    X_tv   = np.stack(train_val_df["acoustic_features"].values)
    y_tv   = np.array([label2id[l] for l in train_val_df["emotion_final"]])
    X_test = np.stack(test_df["acoustic_features"].values)
    y_test = np.array([label2id[l] for l in test_df["emotion_final"]])

    print(f"   Feature dimensionality: {X_tv.shape[1]}")

    skf          = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    fold_results = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_tv, y_tv)):
        print(f"\n{'='*55}")
        print(f"  FOLD {fold_idx+1} / 5")
        print(f"{'='*55}")

        X_tr, X_va = X_tv[tr_idx], X_tv[va_idx]
        y_tr, y_va = y_tv[tr_idx], y_tv[va_idx]

        # Scale (critical for SVM on mixed-scale acoustic features)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)
        X_te_s = scaler.transform(X_test)

        # GridSearch over C and gamma
        param_grid  = {"C": [0.1, 1, 10, 100, 1000], "gamma": ["scale", "auto", 0.001, 0.0001]}
        inner_cv    = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.SEED)
        grid_search = GridSearchCV(
            SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=config.SEED),
            param_grid, cv=inner_cv, scoring="f1_macro", n_jobs=-1, verbose=0
        )
        grid_search.fit(X_tr_s, y_tr)
        svm = grid_search.best_estimator_
        print(f"  Best SVM: C={grid_search.best_params_['C']}  gamma={grid_search.best_params_['gamma']}")

        tr_acc = accuracy_score(y_tr,   svm.predict(X_tr_s))
        tr_f1  = f1_score(y_tr,         svm.predict(X_tr_s), average="macro")
        va_acc = accuracy_score(y_va,   svm.predict(X_va_s))
        va_f1  = f1_score(y_va,         svm.predict(X_va_s), average="macro")
        te_acc = accuracy_score(y_test, svm.predict(X_te_s))
        te_f1  = f1_score(y_test,       svm.predict(X_te_s), average="macro")

        print(f"\n  Train Acc: {tr_acc:.4f}  Train F1: {tr_f1:.4f}")
        print(f"    Val Acc: {va_acc:.4f}    Val F1: {va_f1:.4f}")
        print(f"   Test Acc: {te_acc:.4f}   Test F1: {te_f1:.4f}")
        fold_results.append({"val_f1": va_f1, "test_acc": te_acc, "test_f1": te_f1})

    print(f"\n{'='*55}")
    print(f"🏆  Classical Acoustic SVM  —  FINAL RESULTS")
    print(f"{'='*55}")
    mv  = np.mean([r["val_f1"]   for r in fold_results])
    mta = np.mean([r["test_acc"] for r in fold_results])
    mtf = np.mean([r["test_f1"]  for r in fold_results])
    print(f"  Mean Val  F1 : {mv  * 100:.2f}%")
    print(f"  Mean Test Acc: {mta * 100:.2f}%")
    print(f"  Mean Test F1 : {mtf * 100:.2f}%")
    print(f"{'='*55}")

    # Per-class breakdown
    print("\n📊  Per-Class Test Report (last fold's SVM):")
    print(classification_report(
        y_test, svm.predict(X_te_s),
        target_names=[id2label[i] for i in range(num_labels)]
    ))


if __name__ == "__main__":
    main()
