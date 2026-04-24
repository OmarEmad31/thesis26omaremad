"""
Egyptian Arabic SER — TRACK A BEST VERSION (v65)
===============================================
FIX: Returning to the exact 768-dim feature space that got 59.6%.
"""

import os, sys, random, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import torch
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
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

def extract_handcrafted(p):
    try:
        y, sr = librosa.load(p, sr=16000)
        yt, _ = librosa.effects.trim(y, top_db=40)
        if len(yt)<100: return np.zeros(35)
        mfcc = librosa.feature.mfcc(y=yt, sr=sr, n_mfcc=13)
        feat = [len(yt)/sr, np.mean(librosa.feature.rms(y=yt)), np.std(librosa.feature.rms(y=yt)), 
                np.mean(librosa.feature.zero_crossing_rate(y=yt)), np.std(librosa.feature.zero_crossing_rate(y=yt)), 0]
        feat.extend(np.mean(mfcc, axis=1)); feat.extend(np.std(mfcc, axis=1))
        return np.nan_to_num(feat)
    except: return np.zeros(35)

@torch.no_grad()
def extract_lean_wavlm(paths, model):
    model.eval()
    embs = []
    for p in tqdm(paths, desc="Extracting Lean WavLM"):
        try:
            y, _ = librosa.load(p, sr=16000)
            if len(y)>80000: y = y[(len(y)-80000)//2 : (len(y)-80000)//2+80000]
            inp = torch.from_numpy(y).float().unsqueeze(0).to(TrackAConfig.DEVICE)
            out = model(inp, output_hidden_states=True).hidden_states
            # Exact Best Strategy: Average the LAST 4 layers into a single 768-dim vector
            stack = torch.stack([out[i].squeeze(0) for i in [9,10,11,12]], dim=0) # [4, T, 768]
            layer_mean = torch.mean(stack, dim=0) # [T, 768]
            final_vec = torch.mean(layer_mean, dim=0) # [768]
            embs.append(final_vec.cpu().numpy())
        except: embs.append(np.zeros(768))
    return np.array(embs)

def main():
    seed_everything(TrackAConfig.SEED)
    TrackAConfig.CACHE_DIR.mkdir(exist_ok=True)
    if os.path.exists("/content/drive/MyDrive"): root = Path("/content/drive/MyDrive/Thesis Project")
    else: root = Path(__file__).parent.parent.parent
    
    csv_r = root / "data/processed/splits/text_hc"
    df = pd.read_csv(root / "audio_manifest.csv")
    orig_val_ids = set(pd.read_csv(csv_r/"val.csv")['sample_id'])
    df['split'] = df['sample_id'].apply(lambda x: 'val' if x in orig_val_ids else 'train')
    tr_df, va_df = df[df['split'] == 'train'], df[df['split'] == 'val']
    y_tr, y_va = tr_df['label_id'].values, va_df['label_id'].values

    hc_tr_p, hc_va_p = TrackAConfig.CACHE_DIR/"hc_tr.npy", TrackAConfig.CACHE_DIR/"hc_va.npy"
    wv_tr_p, wv_va_p = TrackAConfig.CACHE_DIR/"mwv_tr.npy", TrackAConfig.CACHE_DIR/"mwv_va.npy"

    if not hc_tr_p.exists():
        np.save(hc_tr_p, [extract_handcrafted(p) for p in tqdm(tr_df['resolved_path'])])
        np.save(hc_va_p, [extract_handcrafted(p) for p in tqdm(va_df['resolved_path'])])
    if not wv_tr_p.exists():
        wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(TrackAConfig.DEVICE)
        np.save(wv_tr_p, extract_lean_wavlm(tr_df['resolved_path'], wavlm))
        np.save(wv_va_p, extract_lean_wavlm(va_df['resolved_path'], wavlm))

    X_hc_tr, X_wv_tr = np.load(hc_tr_p), np.load(wv_tr_p)
    X_hc_va, X_wv_va = np.load(hc_va_p), np.load(wv_va_p)

    # Sanity Check
    print(f"📊 Feature Dim: HC {X_hc_tr.shape[1]} + WavLM {X_wv_tr.shape[1]}")

    X_tr = np.concatenate([X_hc_tr, X_wv_tr], axis=1)
    X_va = np.concatenate([X_hc_va, X_wv_va], axis=1)
    
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_va_s = sc.transform(X_va)
    
    clf = SVC(kernel='rbf', probability=True, class_weight='balanced', C=1.5)
    clf.fit(X_tr_s, y_tr)
    
    preds = clf.predict(X_va_s)
    print(f"\nFinal Accuracy (Track A): {accuracy_score(y_va, preds):.4f}")
    print(f"Macro F1: {f1_score(y_va, preds, average='macro'):.4f}")
    print(classification_report(y_va, preds, target_names=TrackAConfig.EMOTIONS, zero_division=0))

if __name__ == "__main__": main()
