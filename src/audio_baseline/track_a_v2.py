"""
Egyptian Arabic SER — Feature Fusion Rebuild (Track A)
=====================================================
Goal: Reach >60% Accuracy on the Original Split.
Features: Handcrafted, WavLM-Base-Plus.
Classifiers: LogReg, SVM, Random Forest.
"""

import os, sys, time, random, warnings, zipfile, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from collections import Counter
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, 
    classification_report
)
from transformers import WavLMModel

class TrackAConfig:
    SR = 16000
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    EMOTIONS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
    LID = {e: i for i, e in enumerate(EMOTIONS)}
    CACHE_DIR = Path("features_cache")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def extract_handcrafted(path):
    try:
        y, sr = librosa.load(path, sr=TrackAConfig.SR)
        yt, _ = librosa.effects.trim(y, top_db=40)
        if len(yt) < 100: return np.zeros(35)
        mfcc = librosa.feature.mfcc(y=yt, sr=sr, n_mfcc=13)
        zcr = librosa.feature.zero_crossing_rate(y=yt)
        rms = librosa.feature.rms(y=yt)
        f0 = librosa.yin(yt, fmin=65, fmax=1000)
        voiced = f0[~np.isnan(f0)]
        feat = [len(yt)/sr, np.mean(rms), np.std(rms), np.mean(zcr), np.std(zcr), np.mean(voiced) if len(voiced)>0 else 0]
        feat.extend(np.mean(mfcc, axis=1)); feat.extend(np.std(mfcc, axis=1))
        return np.nan_to_num(feat)
    except: return np.zeros(35)

@torch.no_grad()
def extract_wavlm_embeddings(paths, model):
    model.eval()
    embeddings = []
    for p in tqdm(paths, desc="Extracting WavLM"):
        try:
            if not p or not os.path.exists(p): raise FileNotFoundError
            y, _ = librosa.load(p, sr=TrackAConfig.SR)
            if len(y) > 80000:
                start = (len(y)-80000)//2
                y = y[start:start+80000]
            inputs = torch.from_numpy(y).float().unsqueeze(0).to(TrackAConfig.DEVICE)
            out = model(inputs).last_hidden_state
            mean = torch.mean(out, dim=1)
            std = torch.std(out, dim=1)
            emb = torch.cat([mean, std], dim=-1).cpu().numpy().squeeze()
            embeddings.append(emb)
        except:
            embeddings.append(np.zeros(1536))
    return np.array(embeddings)

def run_suite(X_tr, y_tr, X_va, y_va, name="Model"):
    print(f"\n--- {name} ---")
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_va_s = sc.transform(X_va)
    models = {
        "LogReg": LogisticRegression(class_weight='balanced', max_iter=2000),
        "SVM": SVC(kernel='rbf', probability=True, class_weight='balanced'),
        "RF": RandomForestClassifier(n_estimators=200, class_weight='balanced')
    }
    best_obj, best_acc = None, 0
    for n, m in models.items():
        m.fit(X_tr_s, y_tr)
        acc = accuracy_score(y_va, m.predict(X_va_s))
        print(f"  {n:<6}: {acc:.4f}")
        if acc > best_acc: best_acc = acc; best_obj = m
    return best_obj

def main():
    seed_everything(TrackAConfig.SEED)
    TrackAConfig.CACHE_DIR.mkdir(exist_ok=True)
    if os.path.exists("/content/drive/MyDrive"): root = Path("/content/drive/MyDrive/Thesis Project")
    else: root = Path(__file__).parent.parent.parent
    
    man_p = root / "audio_manifest.csv"
    df = pd.read_csv(man_p)
    orig_val = set(pd.read_csv(root / "data/processed/splits/text_hc/val.csv")['sample_id'])
    df['split'] = df['sample_id'].apply(lambda x: 'val' if x in orig_val else 'train')
    tr_df, va_df = df[df['split'] == 'train'], df[df['split'] == 'val']
    y_tr, y_va = tr_df['label_id'], va_df['label_id']

    # Extraction
    hc_tr_p, hc_va_p = TrackAConfig.CACHE_DIR / "hc_tr.npy", TrackAConfig.CACHE_DIR / "hc_va.npy"
    if not hc_tr_p.exists():
        print("Extracting Handcrafted...")
        np.save(hc_tr_p, [extract_handcrafted(p) for p in tqdm(tr_df['resolved_path'])])
        np.save(hc_va_p, [extract_handcrafted(p) for p in tqdm(va_df['resolved_path'])])
    
    wv_tr_p, wv_va_p = TrackAConfig.CACHE_DIR / "wv_tr.npy", TrackAConfig.CACHE_DIR / "wv_va.npy"
    if not wv_tr_p.exists():
        print("Extracting WavLM...")
        wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(TrackAConfig.DEVICE)
        np.save(wv_tr_p, extract_wavlm_embeddings(tr_df['resolved_path'], wavlm))
        np.save(wv_va_p, extract_wavlm_embeddings(va_df['resolved_path'], wavlm))

    hc_tr, hc_va = np.load(hc_tr_p), np.load(hc_va_p)
    wv_tr, wv_va = np.load(wv_tr_p), np.load(wv_va_p)

    print(f"\n🕵️ SANITY: HC Std {hc_tr.std():.4f} | WV Std {wv_tr.std():.4f}")
    if hc_tr.std() < 1e-6 or wv_tr.std() < 1e-6:
        print("Dead features. Clearing cache."); shutil.rmtree(TrackAConfig.CACHE_DIR); return

    m1 = run_suite(hc_tr, y_tr, hc_va, y_va, "Handcrafted")
    m2 = run_suite(wv_tr, y_tr, wv_va, y_va, "WavLM")
    
    f_tr, f_va = np.concatenate([hc_tr, wv_tr], 1), np.concatenate([hc_va, wv_va], 1)
    m3 = run_suite(f_tr, y_tr, f_va, y_va, "Fusion")

    # Ensemble
    print("\n🚀 FINAL ENSEMBLE (WavLM-SVM + Fusion-SVM)")
    s1, s2 = StandardScaler().fit(wv_tr), StandardScaler().fit(f_tr)
    p = (m2.predict_proba(s1.transform(wv_va)) + m3.predict_proba(s2.transform(f_va))) / 2
    preds = np.argmax(p, 1)
    print(f"Final Acc: {accuracy_score(y_va, preds):.4f}")
    print(classification_report(y_va, preds, target_names=TrackAConfig.EMOTIONS, zero_division=0))

if __name__ == "__main__": main()
