import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, classification_report, 
                             confusion_matrix, balanced_accuracy_score)
import joblib

def load_data(feat_manifest, audit_report, model_name):
    audit_df = pd.read_csv(audit_report)
    feat_df = pd.read_csv(feat_manifest)
    
    df = audit_df.merge(feat_df, on="sample_id")
    
    LID = {'Anger':0, 'Disgust':1, 'Fear':2, 'Happiness':3, 'Neutral':4, 'Sadness':5, 'Surprise':6}
    df['label'] = df['emotion_final'].map(LID)
    
    col = f"{model_name.lower()}_path"
    if col not in df.columns:
        return None, None, None, None, None, None
    
    def load_feat(path):
        if pd.isna(path): return None
        return np.load(path)
    
    tr_df = df[df['split'] == 'train']
    va_df = df[df['split'] == 'val']
    te_df = df[df['split'] == 'test']
    
    def get_xy(target_df):
        X, y = [], []
        for _, row in target_df.iterrows():
            f = load_feat(row[col])
            if f is not None:
                X.append(f)
                y.append(row['label'])
        return np.array(X), np.array(y)

    X_train, y_train = get_xy(tr_df)
    X_val, y_val = get_xy(va_df)
    X_test, y_test = get_xy(te_df)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, name):
    model.fit(X_train, y_train)
    
    EMOTIONS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
    
    print(f"\n" + "="*40)
    print(f"RESULTS: {name}")
    print("="*40)
    
    for split_name, X, y in [("VAL", X_val, y_val), ("TEST", X_test, y_test)]:
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        uar = balanced_accuracy_score(y, preds)
        f1_macro = f1_score(y, preds, average='macro')
        f1_weighted = f1_score(y, preds, average='weighted')
        
        print(f"\n[{split_name}]")
        print(f"  Accuracy: {acc:.4f} | UAR: {uar:.4f}")
        print(f"  F1 Macro: {f1_macro:.4f} | F1 Weighted: {f1_weighted:.4f}")
        
        print("\n  Per-Class F1:")
        report = classification_report(y, preds, target_names=EMOTIONS, output_dict=True, zero_division=0)
        for emo in EMOTIONS:
            print(f"    {emo:<10}: {report[emo]['f1-score']:.4f}")
            
        print("\n  Prediction Counts:")
        unique, counts = np.unique(preds, return_counts=True)
        pred_map = dict(zip(unique, counts))
        for i, emo in enumerate(EMOTIONS):
            print(f"    {emo:<10}: {pred_map.get(i, 0)}")

        print("\n  Confusion Matrix:")
        cm = confusion_matrix(y, preds)
        print(cm)
    
    return {
        "model": name,
        "test_acc": accuracy_score(y_test, model.predict(X_test)),
        "test_uar": balanced_accuracy_score(y_test, model.predict(X_test)),
        "test_f1_macro": f1_score(y_test, model.predict(X_test), average='macro')
    }

def run_eval():
    root = Path(r"d:\Thesis Project")
    audit_report = root / "video_manifest_trackA.csv" # Manifest acts as audit
    feat_manifest = root / "video_feature_manifest_v2.csv"
    
    if not feat_manifest.exists():
        print("ERROR: Run video_stage3_extract_features.py (v2) first.")
        # Fallback to v1 if exists
        feat_manifest = root / "video_feature_manifest.csv"
        if not feat_manifest.exists(): return

    models = ["CLIP", "DINOv2", "ResNet50"]
    all_results = []

    for m in models:
        X_train, y_train, X_val, y_val, X_test, y_test = load_data(feat_manifest, audit_report, m)
        if X_train is None or len(X_train) == 0: continue
            
        # Linear SVM
        all_results.append(evaluate_model(
            SVC(kernel='linear', C=1.0, class_weight='balanced'), 
            X_train, y_train, X_val, y_val, X_test, y_test, f"{m}_LinearSVM"
        ))
        
        # RBF SVM
        all_results.append(evaluate_model(
            SVC(kernel='rbf', C=1.0, class_weight='balanced'), 
            X_train, y_train, X_val, y_val, X_test, y_test, f"{m}_RBFSVM"
        ))

    pd.DataFrame(all_results).to_csv(root / "video_evaluation_summary_v2.csv", index=False)

if __name__ == "__main__":
    run_eval()
