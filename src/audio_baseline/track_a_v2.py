"""
Egyptian Arabic SER — Feature Fusion Rebuild (Track A)
=====================================================
Goal: Reach >60% Accuracy on the Original Split.
Features: Handcrafted, WavLM-Base-Plus, emotion2vec.
Classifiers: LogReg, SVM, Random Forest, XGBoost.
Fusion: Late Probability Ensemble & Early Feature Concatenation.
"""

import os, sys, time, random, warnings, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, 
    classification_report, balanced_accuracy_score
)
from transformers import WavLMModel
# Optional: from modelscope.pipelines import pipeline

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

# ─────────────────────────────────────────────────────────
# STAGE 2: FEATURE EXTRACTION SUITE
# ─────────────────────────────────────────────────────────

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
            y, _ = librosa.load(p, sr=TrackAConfig.SR)
            # 5s crop center
            if len(y) > 80000:
                start = (len(y)-80000)//2
                y = y[start:start+80000]
            inputs = torch.from_numpy(y).float().unsqueeze(0).to(TrackAConfig.DEVICE)
            out = model(inputs).last_hidden_state # [1, T, 768]
            # Mean+Std Pool
            mean = torch.mean(out, dim=1)
            std = torch.std(out, dim=1)
            emb = torch.cat([mean, std], dim=-1).cpu().numpy().squeeze()
            embeddings.append(emb)
        except: embeddings.append(np.zeros(768*2))
    return np.array(embeddings)

# ─────────────────────────────────────────────────────────
# STAGE 3: CLASSIFIER TUNING
# ─────────────────────────────────────────────────────────

def run_classifier_suite(X_tr, y_tr, X_va, y_va, name="Model"):
    print(f"\n--- Tuning Suite: {name} ---")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    
    models = {
        "LogReg": LogisticRegression(class_weight='balanced', max_iter=2000),
        "SVM_RBF": SVC(kernel='rbf', probability=True, class_weight='balanced'),
        "RF": RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    }
    
    best_acc = 0
    best_m_name = ""
    best_m_obj = None
    
    for m_name, m_obj in models.items():
        m_obj.fit(X_tr_s, y_tr)
        preds = m_obj.predict(X_va_s)
        acc = accuracy_score(y_va, preds)
        macro = f1_score(y_va, preds, average='macro')
        print(f"  {m_name:<10} | Acc: {acc:.4f} | MacroF1: {macro:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_m_name = m_name
            best_m_obj = m_obj
            
    return best_m_obj, best_acc, best_m_name

def main():
    seed_everything(TrackAConfig.SEED)
    TrackAConfig.CACHE_DIR.mkdir(exist_ok=True)
    
    if os.path.exists("/content/drive/MyDrive"): root = Path("/content/drive/MyDrive/Thesis Project")
    else: root = Path(__file__).parent.parent.parent
    
    manifest_p = root / "audio_manifest.csv"
    if not manifest_p.exists():
         print("❌ Manifest not found. Run Stage 1 logic first.")
         return
    
    # We use original split (Track A) for target performance
    # We re-import the original split from the source CSVs to be sure
    csv_r = root / "data/processed/splits/text_hc"
    df = pd.read_csv(manifest_p)
    orig_val_ids = set(pd.read_csv(csv_r/"val.csv")['sample_id'])
    
    df['split'] = df['sample_id'].apply(lambda x: 'val' if x in orig_val_ids else 'train')
    
    tr_df = df[df['split'] == 'train']
    va_df = df[df['split'] == 'val']
    
    print(f"TRACK A (Original Split): Train {len(tr_df)} | Val {len(va_df)}")
    y_tr, y_va = tr_df['label_id'], va_df['label_id']

    # 1. Handcrafted Features
    hc_tr_p = TrackAConfig.CACHE_DIR / "hc_tr.npy"
    hc_va_p = TrackAConfig.CACHE_DIR / "hc_va.npy"
    
    if not hc_tr_p.exists():
        print("Extracting Handcrafted...")
        hc_tr = np.array([extract_handcrafted(p) for p in tqdm(tr_df['resolved_path'])])
        hc_va = np.array([extract_handcrafted(p) for p in tqdm(va_df['resolved_path'])])
        np.save(hc_tr_p, hc_tr); np.save(hc_va_p, hc_va)
    hc_tr, hc_va = np.load(hc_tr_p), np.load(hc_va_p)

    # 2. WavLM Features
    wv_tr_p = TrackAConfig.CACHE_DIR / "wv_tr.npy"
    wv_va_p = TrackAConfig.CACHE_DIR / "wv_va.npy"
    
    if not wv_tr_p.exists():
        print("Extracting WavLM (Frozen)...")
        wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(TrackAConfig.DEVICE)
        wv_tr = extract_wavlm_embeddings(tr_df['resolved_path'], wavlm)
        wv_va = extract_wavlm_embeddings(va_df['resolved_path'], wavlm)
        np.save(wv_tr_p, wv_tr); np.save(wv_va_p, wv_va)
    wv_tr, wv_va = np.load(wv_tr_p), np.load(wv_va_p)

    # --- EXPERIMENTS ---
    
    # Track 1: Handcrafted Only
    run_classifier_suite(hc_tr, y_tr, hc_va, y_va, name="Handcrafted Only")
    
    # Track 2: WavLM Only
    wav_best_m, wav_best_acc, wav_best_name = run_classifier_suite(wv_tr, y_tr, wv_va, y_va, name="WavLM Only")
    
    # Track 3: Early Fusion (Concatenation)
    fused_tr = np.concatenate([hc_tr, wv_tr], axis=1)
    fused_va = np.concatenate([hc_va, wv_va], axis=1)
    fus_best_m, fus_best_acc, fus_best_name = run_classifier_suite(fused_tr, y_tr, fused_va, y_va, name="Handcrafted + WavLM Fusion")

    # Track 4: Late Fusion (Ensemble of WavLM and Fusion)
    # We'll use the two best models based on Accuracy
    print("\n🚀 Probability Ensemble (WavLM-SVM + Fused-SVM)...")
    scaler_wv = StandardScaler().fit(wv_tr)
    scaler_fus = StandardScaler().fit(fused_tr)
    
    # Get probs
    p1 = wav_best_m.predict_proba(scaler_wv.transform(wv_va))
    p2 = fus_best_m.predict_proba(scaler_fus.transform(fused_va))
    
    p_final = (p1 + p2) / 2
    ens_preds = np.argmax(p_final, axis=1)
    
    print("\n🏆 FINAL TRACK A RESULTS (Ensemble):")
    print(f"Accuracy: {accuracy_score(y_va, ens_preds):.4f}")
    print(f"Macro F1: {f1_score(y_va, ens_preds, average='macro'):.4f}")
    print(classification_report(y_va, ens_preds, target_names=TrackAConfig.EMOTIONS, zero_division=0))
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_va, ens_preds))

if __name__ == "__main__":
    main()
