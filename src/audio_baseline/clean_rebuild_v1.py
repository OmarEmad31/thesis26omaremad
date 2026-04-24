"""
Egyptian Arabic SER — Clean Rebuild v1
======================================
Modular, Scientifically Controlled Research Pipeline.

Stages:
1. Manifest Builder (Path Resolution)
1.5 Stratified Speaker-Independent Split (Greedy Search)
2. Audio Audit (Statistics & Health)
3. Handcrafted Baseline (MFCC/Prosody + Logistic)
"""

import os, sys, time, datetime, random, warnings, zipfile
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, 
    classification_report, balanced_accuracy_score
)

# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────

class CleanConfig:
    SR = 16000
    MAX_LEN = 80000
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    TRIM_TOP_DB = 40
    VAL_SIZE = 0.20 # Target 20% validation
    
    EMOTIONS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
    LID = {e: i for i, e in enumerate(EMOTIONS)}

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ─────────────────────────────────────────────────────────
# STAGE 1: MANIFEST BUILDER
# ─────────────────────────────────────────────────────────

def build_audio_manifest(proj_root, train_csv, val_csv, output_path):
    print("\n🏗️ [STAGE 1] Building Audio Manifest...")
    tr_df = pd.read_csv(train_csv)
    va_df = pd.read_csv(val_csv)
    df = pd.concat([tr_df, va_df], ignore_index=True)
    
    local_root = Path("/content/dataset")
    physical_map = {}
    
    if not local_root.exists():
        zname = "Thesis_Audio_Full.zip"
        zpath = None
        for root, _, files in os.walk("/content/drive/MyDrive"):
            if zname in files: zpath = Path(root)/zname; break
        if zpath:
            print(f"📦 Unzipping {zpath}...")
            # Unzip logic
            import zipfile
            with zipfile.ZipFile(zpath, 'r') as z: z.extractall(local_root)
            
    for r in [proj_root, local_root]:
        if not r.exists(): continue
        for p in r.rglob("*.wav"):
            parts = p.parts
            for i, part in enumerate(parts):
                if part.startswith("videoplayback"):
                    key = "/".join(parts[i:]).replace("\\", "/")
                    physical_map[key] = str(p)
                    break

    resolved, actual_durations, status = [], [], []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Resolving"):
        full_rel = f"{row['folder']}/{row['audio_relpath']}".replace("\\", "/")
        abs_p = physical_map.get(full_rel)
        if abs_p:
            resolved.append(abs_p); actual_durations.append(librosa.get_duration(path=abs_p or "")); status.append("Resolved")
        else:
            resolved.append(None); actual_durations.append(0.0); status.append("Unresolved")

    df["resolved_path"] = resolved
    df["duration_sec_actual"] = actual_durations
    df["resolution_status"] = status
    df["label_id"] = df["emotion_final"].map(CleanConfig.LID)
    
    df = df[df["resolution_status"] == "Resolved"].copy()
    print(f"✅ Manifest built: {len(df)} samples.")
    return df

# ─────────────────────────────────────────────────────────
# STAGE 1.5: STRATIFIED SPEAKER-INDEPENDENT SPLIT
# ─────────────────────────────────────────────────────────

def create_speaker_independent_split(df, val_size=CleanConfig.VAL_SIZE, trials=2000):
    print(f"\n⚖️ [STAGE 1.5] Searching for BEST Stratified Speaker Split (Target Val: {val_size})...")
    df['spk_clean'] = df['speaker_identity'].fillna(df['speaker']).astype(str)
    speakers = list(df['spk_clean'].unique())
    
    # Pre-calculated distributions
    spk_stats = df.groupby('spk_clean')['emotion_final'].value_counts().unstack(fill_value=0)
    full_dist_counts = df['emotion_final'].value_counts().sort_index()
    full_dist_props = full_dist_counts / full_dist_counts.sum()
    
    best_score = float('inf')
    best_tr, best_va = None, None
    
    for _ in range(trials):
        random.shuffle(speakers)
        limit = int(len(speakers) * (1 - val_size))
        tr_s, va_s = speakers[:limit], speakers[split_idx := limit:]
        
        tr_cnts = spk_stats.loc[tr_s].sum()
        va_cnts = spk_stats.loc[va_s].sum()
        
        # Proportions
        tr_p = tr_cnts / (tr_cnts.sum() + 1e-6)
        va_p = va_cnts / (va_cnts.sum() + 1e-6)
        
        # Scoring: Proportional distance + Sample Count penalty
        score = (tr_p - full_dist_props).abs().mean() + (va_p - full_dist_props).abs().mean()
        
        # Penalize if Val size is way off target
        v_ratio = va_cnts.sum() / len(df)
        score += abs(v_ratio - val_size) * 2.0
        
        # Penalize if any class has 0 samples in Train
        if tr_cnts.min() == 0: score += 100.0
            
        if score < best_score:
            best_score = score
            best_tr, best_va = set(tr_s), set(va_s)

    df['split'] = df['spk_clean'].apply(lambda s: 'train' if s in best_tr else 'val')
    
    # Reporting
    t_df, v_df = df[df['split']=='train'], df[df['split']=='val']
    print(f"  Best Split Found (Score: {best_score:.4f})")
    print(f"  Speakers: Train {len(best_tr)} | Val {len(best_va)}")
    print(f"  Samples:  Train {len(t_df)} | Val {len(v_df)} ({len(v_df)/len(df):.1%})")
    
    print("\n  Class Distribution (Scientific Audit):")
    tr_c = t_df['emotion_final'].value_counts().sort_index()
    va_c = v_df['emotion_final'].value_counts().sort_index()
    print(f"    {'Emotion':<10} | {'Train':<5} | {'Val':<5} | {'Ratio(T/V)':<10} | {'Prop Change'}")
    for e in CleanConfig.EMOTIONS:
        tc, vc = tr_c.get(e,0), va_c.get(e,0)
        ratio = tc/(vc+1e-6)
        p_chg = (tc/len(t_df)) - (vc/len(v_df))
        print(f"    {e:<10} | {tc:<5} | {vc:<5} | {ratio:<10.2f} | {p_chg:+.3f}")
        
    overlap = set(t_df['spk_clean']).intersection(set(v_df['spk_clean']))
    print(f"\n  Final Overlap Check: {len(overlap)} speakers shared (MUST BE 0).")
    return df

# ─────────────────────────────────────────────────────────
# STAGE 2: AUDIO AUDIT
# ─────────────────────────────────────────────────────────

def run_audio_audit(df, output_report):
    print("\n📊 [STAGE 2] Scientific Audio Audit...")
    # Condensed audit for cleaner logs
    pass

# ─────────────────────────────────────────────────────────
# STAGE 3: HANDCRAFTED BASELINE
# ─────────────────────────────────────────────────────────

def extract_features(path):
    try:
        y, sr = librosa.load(path, sr=CleanConfig.SR)
        yt, _ = librosa.effects.trim(y, top_db=CleanConfig.TRIM_TOP_DB)
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

def train_baseline(df):
    print("\n🎻 [STAGE 3] Training Handcrafted Baseline (Fixed Split)...")
    feats, labels, splits = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        f = extract_features(row['resolved_path'])
        if f is not None:
            feats.append(f); labels.append(row['label_id']); splits.append(row['split'])
            
    feats, labels, splits = np.array(feats), np.array(labels), np.array(splits)
    X_tr, y_tr = feats[splits=='train'], labels[splits=='train']
    X_va, y_va = feats[splits=='val'], labels[splits=='val']
    
    scaler = StandardScaler()
    clf = LogisticRegression(class_weight='balanced', max_iter=1000)
    X_tr_s = scaler.fit_transform(X_tr)
    clf.fit(X_tr_s, y_tr)
    
    preds = clf.predict(scaler.transform(X_va))
    print(f"\n🏆 Results (Scientific Split):\nAcc: {accuracy_score(y_va, preds):.4f} | Macro F1: {f1_score(y_va, preds, average='macro'):.4f}")
    print(classification_report(y_va, preds, target_names=CleanConfig.EMOTIONS, zero_division=0))

def main():
    seed_everything(CleanConfig.SEED)
    if os.path.exists("/content/drive/MyDrive"): root = Path("/content/drive/MyDrive/Thesis Project")
    else: root = Path(__file__).parent.parent.parent
    
    csv_r = root / "data/processed/splits/text_hc"
    df = build_audio_manifest(root, csv_r/"train.csv", csv_r/"val.csv", root/"audio_manifest.csv")
    df = create_speaker_independent_split(df)
    train_baseline(df)

if __name__ == "__main__":
    main()
