"""
Egyptian Arabic SER — Baseline Verification
===========================================
STRICTLY reproduces the 56.9% baseline on the clean manifest.
Used to verify data integrity before adding WavLM/SSL.
"""

import os, sys, pandas as pd, numpy as np, librosa, zipfile
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def extract_features(path):
    try:
        y, sr = librosa.load(path, sr=16000)
        yt, _ = librosa.effects.trim(y, top_db=40)
        if len(yt) < 100: return None
        mfcc = librosa.feature.mfcc(y=yt, sr=sr, n_mfcc=13)
        zcr = librosa.feature.zero_crossing_rate(y=yt)
        rms = librosa.feature.rms(y=yt)
        f0 = librosa.yin(yt, fmin=65, fmax=1000)
        voiced = f0[~np.isnan(f0)]
        feat = [len(yt)/sr, np.mean(rms), np.std(rms), np.mean(zcr), np.std(zcr), np.mean(voiced) if len(voiced)>0 else 0]
        feat.extend(np.mean(mfcc, axis=1)); feat.extend(np.std(mfcc, axis=1))
        return np.nan_to_num(feat)
    except: return None

def main():
    root = Path("/content/drive/MyDrive/Thesis Project")
    if not root.exists(): root = Path("D:/Thesis Project") # Fallback to local
    
    csv_r = root / "data/processed/splits/text_hc"
    man_p = root / "audio_manifest.csv"
    
    # 1. Load Original Split
    print("⏳ Loading Original Split Identities...")
    df = pd.read_csv(man_p)
    orig_val_ids = set(pd.read_csv(csv_r/"val.csv")['sample_id'])
    df['split'] = df['sample_id'].apply(lambda x: 'val' if x in orig_val_ids else 'train')
    
    # 2. Extract Features
    print(f"📦 Extracting Handcrafted for {len(df)} samples...")
    feats, labels, splits = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        f = extract_features(row['resolved_path'])
        if f is not None:
            feats.append(f); labels.append(row['label_id']); splits.append(row['split'])
            
    feats, labels, splits = np.array(feats), np.array(labels), np.array(splits)
    X_tr, y_tr = feats[splits=='train'], labels[splits=='train']
    X_va, y_va = feats[splits=='val'], labels[splits=='val']
    
    # 3. Train Simple Logistic Regression
    print("\n🎻 Training Verification Baseline...")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    clf = LogisticRegression(class_weight='balanced', max_iter=2000)
    clf.fit(X_tr_s, y_tr)
    
    preds = clf.predict(scaler.transform(X_va))
    print(f"\n🏆 VERIFICATION RESULT (Original Split):")
    print(f"Accuracy: {accuracy_score(y_va, preds):.4f}")
    print(classification_report(y_va, preds, target_names=['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']))

if __name__ == "__main__": main()
