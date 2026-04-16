"""
Precision Tri-Expert Trainer (v2).
Implements: Feature Selection, SMOTE, and Voting Ensemble.
Goal: 58-60% individual audio performance.

Run: python -m src.audio_baseline.train_tri_ensemble
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
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

    print(f"\n{'='*70}")
    print("🏆  PRECISION TRI-EXPERT ENSEMBLE (v2)  —  TARGET: 58%+")
    print("    Mask-Aware 2k Features + SMOTE + Voting Ensemble")
    print(f"{'='*70}")

    data_path = config.SER_FEATURES_DIR / "tri_expert_dataset_v2.pkl"
    if not data_path.exists():
        print(f"[FATAL] Features v2 not found at {data_path}")
        print("Run: python -m src.audio_baseline.extract_tri_expert  first!")
        return

    data = pd.read_pickle(data_path)
    df = data["metadata"]
    X_all = np.concatenate([data["feat_arabic"], data["feat_ser"], data["feat_physics"]], axis=1)
    
    label2id = {l: i for i, l in enumerate(sorted(df["emotion_final"].unique()))}
    id2label = {i: l for l, i in label2id.items()}
    y = np.array([label2id[l] for l in df["emotion_final"]])
    
    train_mask = (df["split"] != "test").values
    test_mask  = (df["split"] == "test").values
    X_tv, y_tv = X_all[train_mask], y[train_mask]
    X_te, y_te = X_all[test_mask],  y[test_mask]

    print(f"   Original Features: {X_all.shape[1]}")
    print(f"   Train samples: {len(X_tv)} | Test samples: {len(X_te)}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    fold_results = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_tv, y_tv)):
        print(f"\n🌀 FOLD {fold_idx+1} / 5")
        
        X_tr, y_tr = X_tv[tr_idx], y_tv[tr_idx]
        X_va, y_va = X_tv[va_idx], y_tv[va_idx]

        # 1. Scale
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)
        X_te_s = scaler.transform(X_te)

        # 2. Feature Selection: Select top 512 most "emotional" features (Reduce noise by 75%)
        # K=512 is a sweet spot for 600 samples
        selector = SelectKBest(f_classif, k=512)
        X_tr_k = selector.fit_transform(X_tr_s, y_tr)
        X_va_k = selector.transform(X_va_s)
        X_te_k = selector.transform(X_te_s)
        print(f"   Selected top {X_tr_k.shape[1]} features via f_classif.")

        # 3. SMOTE Oversampling: Equalize classes in training set
        smote = SMOTE(random_state=config.SEED)
        X_tr_sm, y_tr_sm = smote.fit_resample(X_tr_k, y_tr)
        print(f"   SMOTE: Balanced train set to {len(X_tr_sm)} samples.")

        # 4. Voting Ensemble (Expert Committee)
        clf_svm = SVC(kernel="rbf", C=10, gamma="auto", probability=True, random_state=config.SEED)
        clf_rf  = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=config.SEED)
        clf_xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, tree_method='hist', random_state=config.SEED)
        
        voter = VotingClassifier(
            estimators=[('svm', clf_svm), ('rf', clf_rf), ('xgb', clf_xgb)],
            voting='soft'
        )
        
        voter.fit(X_tr_sm, y_tr_sm)
        
        va_acc = accuracy_score(y_va, voter.predict(X_va_k))
        va_f1  = f1_score(y_va,  voter.predict(X_va_k),  average="macro")
        te_acc = accuracy_score(y_te, voter.predict(X_te_k))
        te_f1  = f1_score(y_te,  voter.predict(X_te_k),  average="macro")

        print(f"   Val Acc: {va_acc:.4f} | Val F1: {va_f1:.4f}")
        print(f"   Test Acc: {te_acc:.4f} | Test F1: {te_f1:.4f}")
        
        fold_results.append({"val_f1": va_f1, "test_acc": te_acc, "test_f1": te_f1})

    print(f"\n{'='*70}")
    print(f"🏆  FINAL PERFORMANCE: PRECISION ENSEMBLE")
    print(f"{'='*70}")
    print(f"  Mean Val  F1 : {np.mean([r['val_f1']  for r in fold_results])*100:.2f}%")
    print(f"  Mean Test Acc: {np.mean([r['test_acc'] for r in fold_results])*100:.2f}%")
    print(f"  Mean Test F1 : {np.mean([r['test_f1']  for r in fold_results])*100:.2f}%")
    print(f"{'='*70}")

    print("\n📊  Per-Class (Last Fold):")
    print(classification_report(y_te, voter.predict(X_te_k), target_names=[id2label[i] for i in range(len(label2id))]))

if __name__ == "__main__":
    main()
