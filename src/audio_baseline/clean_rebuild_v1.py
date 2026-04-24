"""
Egyptian Arabic SER — Clean Rebuild v1
======================================
Modular, Scientifically Controlled Research Pipeline.

Stages:
1. Manifest Builder (Path Resolution)
2. Audio Audit (Statistics & Health)
3. Handcrafted Baseline (MFCC/Prosody + Logistic)
4. Frozen WavLM Baseline
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
from transformers import WavLMModel

# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────

class CleanConfig:
    SR = 16000
    MAX_LEN = 80000
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    TRIM_TOP_DB = 40
    
    # Model config
    BATCH_SIZE = 16
    EPOCHS = 10
    LR = 1e-3
    
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

def build_audio_manifest(colab_root, train_csv, val_csv, output_path):
    print("\n🏗️ [STAGE 1] Building Audio Manifest...")
    
    # Load raw CSVs
    tr_df = pd.read_csv(train_csv)
    va_df = pd.read_csv(val_csv)
    tr_df['split'] = 'train'
    va_df['split'] = 'val'
    df = pd.concat([tr_df, va_df], ignore_index=True)
    
    # 1. Scan Physical Storage
    print("Listing files from Drive and fast local storage...")
    local_root = Path("/content/dataset")
    physical_map = {} # rel_suffix -> abs_path
    
    search_roots = [colab_root, local_root]
    # Check for zip extraction
    if not local_root.exists():
        zname = "Thesis_Audio_Full.zip"
        zpath = None
        for root, _, files in os.walk("/content/drive/MyDrive"):
            if zname in files: zpath = Path(root)/zname; break
        if zpath:
            print(f"📦 Unzipping {zpath}...")
            with zipfile.ZipFile(zpath, 'r') as z: z.extractall(local_root)
            
    for r in search_roots:
        if not r.exists(): continue
        for p in r.rglob("*.wav"):
            # Folder-based unique key: folder/audios/speaker/file.wav
            try:
                # We expect paths like .../videoplayback (1)/audios/...
                parts = p.parts
                for i, part in enumerate(parts):
                    if part.startswith("videoplayback"):
                        key = "/".join(parts[i:]).replace("\\", "/")
                        physical_map[key] = str(p)
                        break
            except: pass

    # 2. Resolve Paths
    resolved, actual_durations, status = [], [], []
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Resolving"):
        folder = str(row["folder"])
        rel = str(row["audio_relpath"]).replace("\\", "/")
        full_rel = f"{folder}/{rel}"
        
        abs_p = physical_map.get(full_rel)
        if abs_p:
            resolved.append(abs_p)
            # duration audit
            try:
                d = librosa.get_duration(path=abs_p)
                actual_durations.append(d)
                status.append("Resolved")
            except:
                actual_durations.append(0.0)
                status.append("Corrupt")
        else:
            resolved.append(None)
            actual_durations.append(0.0)
            status.append("Unresolved")

    df["resolved_path"] = resolved
    df["duration_sec_actual"] = actual_durations
    df["resolution_status"] = status
    df["label_id"] = df["emotion_final"].map(CleanConfig.LID)
    
    unresolved_count = (df["resolution_status"] != "Resolved").sum()
    if unresolved_count > 0:
        print(f"❌ Resolution Failed: {unresolved_count} rows could not be linked.")
        print(df[df["resolution_status"] != "Resolved"][["sample_id", "folder", "audio_relpath"]].head(20))
        raise FileNotFoundError("Manifest build failed due to unresolved paths.")
    
    df.to_csv(output_path, index=False)
    print(f"✅ Manifest saved to {output_path} ({len(df)} rows)")
    return df

# ─────────────────────────────────────────────────────────
# STAGE 2: AUDIO AUDIT
# ─────────────────────────────────────────────────────────

def run_audio_audit(df, output_report):
    print("\n📊 [STAGE 2] Running Audio Audit...")
    
    audit_results = []
    
    # 1. Class Counts
    c_counts = df.groupby(['split', 'emotion_final']).size().unstack(fill_value=0)
    audit_results.append("Class Distribution:\n" + str(c_counts))
    
    # 2. Duration Stats
    dur_stats = df.groupby('split')['duration_sec_actual'].agg(['min', 'mean', 'max'])
    audit_results.append("\nDuration Stats (Actual):\n" + str(dur_stats))
    
    # 3. Short Clips
    short_05 = (df['duration_sec_actual'] < 0.5).sum()
    short_10 = (df['duration_sec_actual'] < 1.0).sum()
    audit_results.append(f"\nShort Clips:\n  <0.5s: {short_05}\n  <1.0s: {short_10}")
    
    # 4. Speaker Overlap
    tr_spk = set(df[df['split']=='train']['speaker_identity'].dropna())
    va_spk = set(df[df['split']=='val']['speaker_identity'].dropna())
    overlap = tr_spk.intersection(va_spk)
    audit_results.append(f"\nSpeaker Audit:\n  Train Speakers: {len(tr_spk)}\n  Val Speakers: {len(va_spk)}\n  Overlap: {len(overlap)}")
    
    # 5. Folder Overlap
    tr_fld = set(df[df['split']=='train']['folder'])
    va_fld = set(df[df['split']=='val']['folder'])
    f_overlap = tr_fld.intersection(va_fld)
    audit_results.append(f"\nFolder Audit:\n  Overlap Folders: {len(f_overlap)}")
    
    report_text = "\n".join(audit_results)
    with open(output_report, "w") as f: f.write(report_text)
    print(report_text)
    
    # 20 samples for user
    print("\n🔍 Identity Spot-Check:")
    cols = ["sample_id", "folder", "audio_relpath", "resolved_path", "emotion_final"]
    print(df[cols].sample(20).to_string())

# ─────────────────────────────────────────────────────────
# STAGE 3: HANDCRAFTED BASELINE
# ─────────────────────────────────────────────────────────

def extract_handcrafted_features(path):
    try:
        y, sr = librosa.load(path, sr=CleanConfig.SR)
        yt, _ = librosa.effects.trim(y, top_db=CleanConfig.TRIM_TOP_DB)
        if len(yt) < 100: return None
        
        # Spectral
        mfcc = librosa.feature.mfcc(y=yt, sr=sr, n_mfcc=13)
        zcr = librosa.feature.zero_crossing_rate(y=yt)
        rms = librosa.feature.rms(y=yt)
        
        # Pitch
        f0 = librosa.yin(yt, fmin=float(librosa.note_to_hz('C2')), fmax=float(librosa.note_to_hz('C7')))
        voiced = f0[~np.isnan(f0)]
        
        feat = [
            len(yt)/sr, # duration
            np.mean(rms), np.std(rms),
            np.mean(zcr), np.std(zcr),
            np.mean(voiced) if len(voiced)>0 else 0,
            np.std(voiced) if len(voiced)>0 else 0,
            len(voiced)/len(f0), # voiced ratio
            (np.max(voiced) - np.min(voiced)) if len(voiced)>0 else 0, # range
        ]
        # MFCC
        feat.extend(np.mean(mfcc, axis=1))
        feat.extend(np.std(mfcc, axis=1))
        return np.nan_to_num(feat)
    except: return None

def train_handcrafted_baseline(df):
    print("\n🎻 [STAGE 3] Training Handcrafted Baseline...")
    
    features, labels, splits = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        f = extract_handcrafted_features(row['resolved_path'])
        if f is not None:
            features.append(f)
            labels.append(row['label_id'])
            splits.append(row['split'])
            
    features = np.array(features)
    labels = np.array(labels)
    splits = np.array(splits)
    
    tr_idx = (splits == 'train')
    va_idx = (splits == 'val')
    
    X_train, y_train = features[tr_idx], labels[tr_idx]
    X_val, y_val = features[va_idx], labels[va_idx]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    clf = LogisticRegression(class_weight='balanced', max_iter=1000)
    clf.fit(X_train_scaled, y_train)
    
    preds = clf.predict(X_val_scaled)
    
    print("\n🏆 Handcrafted Baseline Results:")
    print(f"Acc: {accuracy_score(y_val, preds):.4f}")
    print(f"Macro F1: {f1_score(y_val, preds, average='macro'):.4f}")
    print(classification_report(y_val, preds, target_names=CleanConfig.EMOTIONS))
    return clf

# ─────────────────────────────────────────────────────────
# STAGE 4: FROZEN WAVLM BASELINE
# ─────────────────────────────────────────────────────────

def main():
    seed_everything(CleanConfig.SEED)
    
    # Environment detection
    if os.path.exists("/content/drive/MyDrive"):
        proj_root = Path("/content/drive/MyDrive/Thesis Project")
    else:
        # Local Windows fallback
        proj_root = Path(__file__).parent.parent.parent
    
    csv_root = proj_root / "data/processed/splits/text_hc"
    manifest_p = proj_root / "audio_manifest.csv"
    report_p = proj_root / "audio_audit_report.txt"
    
    print(f"🏠 Project Root: {proj_root}")
    
    # 1. Build Manifest
    df = build_audio_manifest(proj_root, csv_root/"train.csv", csv_root/"val.csv", manifest_p)
    
    # 2. Run Audit
    run_audio_audit(df, report_p)
    
    # 3. Handcrafted Baseline
    train_handcrafted_baseline(df)

if __name__ == "__main__":
    main()
