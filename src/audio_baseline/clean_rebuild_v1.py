"""
Egyptian Arabic SER — Clean Rebuild v1
======================================
Stage: Repeated Stratified Group Evaluation (5-Split Stability Check)
"""

import os, sys, time, datetime, random, warnings, zipfile
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, balanced_accuracy_score

class CleanConfig:
    SR = 16000
    SEED = 42
    TRIM_TOP_DB = 40
    N_SPLITS = 5
    VAL_SIZE_TARGET = 0.28 # Targeting ~28%
    EMOTIONS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
    LID = {e: i for i, e in enumerate(EMOTIONS)}

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

# ─────────────────────────────────────────────────────────
# STAGE 1: DATA LOADING
# ─────────────────────────────────────────────────────────

def build_manifest(root, train_csv, val_csv):
    tr = pd.read_csv(train_csv); va = pd.read_csv(val_csv)
    df = pd.concat([tr, va], ignore_index=True)
    df['spk_clean'] = df['speaker_identity'].fillna(df['speaker']).astype(str)
    
    # Path resolution (already verified in v50/51)
    local_root = Path("/content/dataset")
    physical_map = {}
    if not local_root.exists():
        zname = "Thesis_Audio_Full.zip"
        for r, _, files in os.walk("/content/drive/MyDrive"):
            if zname in files:
                with zipfile.ZipFile(Path(r)/zname, 'r') as z: z.extractall(local_root)
                break
    for r in [root, local_root]:
        if r.exists():
            for p in r.rglob("*.wav"):
                parts = p.parts
                for i, pt in enumerate(parts):
                    if pt.startswith("videoplayback"):
                        physical_map["/".join(parts[i:]).replace("\\", "/")] = str(p); break
    
    resolved = []
    for _, row in df.iterrows():
        key = f"{row['folder']}/{row['audio_relpath']}".replace("\\", "/")
        resolved.append(physical_map.get(key))
    df["resolved_path"] = resolved
    df["label_id"] = df["emotion_final"].map(CleanConfig.LID)
    return df[df["resolved_path"].notna()].copy()

# ─────────────────────────────────────────────────────────
# STAGE 1.5: REPEATED STABLE SPLITS
# ─────────────────────────────────────────────────────────

def generate_n_stable_splits(df, n=5):
    print(f"\n⚖️ [STAGE 1.5] Generating {n} Stable Speaker-Independent Splits...")
    speakers = list(df['spk_clean'].unique())
    spk_stats = df.groupby('spk_clean')['emotion_final'].value_counts().unstack(fill_value=0)
    
    valid_splits = []
    attempts = 0
    while len(valid_splits) < n and attempts < 10000:
        random.shuffle(speakers)
        limit = int(len(speakers) * (1 - CleanConfig.VAL_SIZE_TARGET))
        tr_s, va_s = speakers[:limit], speakers[limit:]
        
        tr_cnts = spk_stats.loc[tr_s].sum()
        va_cnts = spk_stats.loc[va_s].sum()
        
        # QUALITY CRITERIA
        # 1. Rare classes must exist in Val (Fear target >= 6 if possible, min 3)
        if va_cnts.min() < 3: attempts += 1; continue
        if tr_cnts.min() < 5: attempts += 1; continue
        
        # 2. Split size range
        v_size = va_cnts.sum() / len(df)
        if not (0.22 <= v_size <= 0.35): attempts += 1; continue
        
        # If valid, save split map
        spk_to_split = {s: 'train' for s in tr_s}
        spk_to_split.update({s: 'val' for s in va_s})
        
        # Avoid duplicate splits
        sig = "".join(sorted(va_s))
        if sig not in [s['sig'] for s in valid_splits]:
            valid_splits.append({
                'map': [spk_to_split[s] for s in df['spk_clean']],
                'spk_map': spk_to_split,
                'sig': sig,
                'tr_c': tr_cnts,
                'va_c': va_cnts
            })
            
    print(f"  Found {len(valid_splits)} candidate splits after {attempts} attempts.")
    return valid_splits

# ─────────────────────────────────────────────────────────
# STAGE 3: HANDCRAFTED BASELINE EVAL
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

def evaluate_split(df, split_info, features):
    df['split'] = df['spk_clean'].map(split_info['map'])
    
    X_tr = features[df['split'] == 'train']
    y_tr = df[df['split'] == 'train']['label_id']
    X_va = features[df['split'] == 'val']
    y_va = df[df['split'] == 'val']['label_id']
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    clf = LogisticRegression(class_weight='balanced', max_iter=1000)
    clf.fit(X_tr_s, y_tr)
    
    preds = clf.predict(scaler.transform(X_va))
    
    metrics = {
        'acc': accuracy_score(y_va, preds),
        'macro_f1': f1_score(y_va, preds, average='macro'),
        'uar': balanced_accuracy_score(y_va, preds),
        'weighted_f1': f1_score(y_va, preds, average='weighted'),
        'report': classification_report(y_va, preds, target_names=CleanConfig.EMOTIONS, output_dict=True, zero_division=0)
    }
    return metrics

def main():
    seed_everything(CleanConfig.SEED)
    if os.path.exists("/content/drive/MyDrive"): root = Path("/content/drive/MyDrive/Thesis Project")
    else: root = Path(__file__).parent.parent.parent
    
    csv_r = root / "data/processed/splits/text_hc"
    # Stage 1
    df = build_manifest(root, csv_r/"train.csv", csv_r/"val.csv")
    
    # Pre-extract features for all samples once
    print("\n📦 Pre-extracting features for all speakers...")
    all_feats = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        all_feats.append(extract_features(row['resolved_path']))
    all_feats = np.array(all_feats)
    
    # Generate 5 stable splits
    splits = generate_n_stable_splits(df, n=CleanConfig.N_SPLITS)
    
    results = []
    for i, s in enumerate(splits):
        print(f"\n🚀 Evaluating Split {i+1}/{len(splits)}...")
        m = evaluate_split(df, s, all_feats)
        results.append(m)
        print(f"   Acc: {m['acc']:.4f} | MacroF1: {m['macro_f1']:.4f} | UAR: {m['uar']:.4f}")
        
    # FINAL STABILITY REPORT
    print("\n🏆 FINAL REPEATED EVALUATION REPORT (Handcrafted Baseline)")
    print("-" * 60)
    
    for metric in ['acc', 'macro_f1', 'uar', 'weighted_f1']:
        vals = [r[metric] for r in results]
        print(f"{metric.upper():<12}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
        
    print("\nMean Per-Class F1-Scores:")
    for e in CleanConfig.EMOTIONS:
        f1s = [r['report'][e]['f1-score'] for r in results]
        print(f"  {e:<10}: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

if __name__ == "__main__":
    main()
