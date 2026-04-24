"""
Egyptian Arabic SER — Clean Rebuild v1
======================================
Modular, Scientifically Controlled Research Pipeline.

Stages:
1. Manifest Builder (Path Resolution)
1.5 Speaker-Independent Split (The Scientific Fix)
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
            resolved.append(abs_p); actual_durations.append(librosa.get_duration(path=abs_p)); status.append("Resolved")
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
# STAGE 1.5: SPEAKER-INDEPENDENT SPLIT
# ─────────────────────────────────────────────────────────

def create_speaker_independent_split(df, test_size=0.25):
    print("\n⚖️ [STAGE 1.5] Creating Speaker-Independent Split...")
    df['spk_clean'] = df['speaker_identity'].fillna(df['speaker']).astype(str)
    
    speakers = sorted(df['spk_clean'].unique())
    random.shuffle(speakers)
    
    limit = int(len(speakers) * (1 - test_size))
    tr_spks = set(speakers[:limit])
    
    df['split'] = df['spk_clean'].apply(lambda s: 'train' if s in tr_spks else 'val')
    
    t_cnt = len(df[df['split']=='train'])
    v_cnt = len(df[df['split']=='val'])
    print(f"  Speakers: {len(speakers)} | Train Samples: {t_cnt} | Val Samples: {v_cnt}")
    return df

# ─────────────────────────────────────────────────────────
# STAGE 2: AUDIO AUDIT
# ─────────────────────────────────────────────────────────

def run_audio_audit(df, output_report):
    print("\n📊 [STAGE 2] Scientific Audio Audit...")
    audit = []
    
    # Leakage Check
    tr_s = set(df[df['split']=='train']['spk_clean'])
    va_s = set(df[df['split']=='val']['spk_clean'])
    overlap = tr_s.intersection(va_s)
    
    audit.append(f"Split Audit:\n  Train Speakers: {len(tr_s)}\n  Val Speakers: {len(va_s)}\n  OVERLAP: {len(overlap)}")
    
    # Class Check
    c_counts = df.groupby(['split', 'emotion_final']).size().unstack(fill_value=0)
    audit.append("\nClass Balance:\n" + str(c_counts))
    
    report_text = "\n".join(audit)
    with open(output_report, "w") as f: f.write(report_text)
    print(report_text)
    
    # Table Spot Check
    print("\n🔍 Identity Spot-Check (Unseen Voices Only):")
    print(df[df['split']=='val'][["sample_id", "folder", "spk_clean", "emotion_final"]].sample(15).to_string())

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
        f0 = librosa.yin(yt, fmin=65, fmax=2000)
        voiced = f0[~np.isnan(f0)]
        feat = [len(yt)/sr, np.mean(rms), np.std(rms), np.mean(zcr), np.std(zcr), np.mean(voiced) if len(voiced)>0 else 0]
        feat.extend(np.mean(mfcc, axis=1))
        feat.extend(np.std(mfcc, axis=1))
        return np.nan_to_num(feat)
    except: return None

def train_baseline(df):
    print("\n🎻 [STAGE 3] Training Handcrafted Baseline (Unseen Voices)...")
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
    print(f"\n🏆 Results (Speaker-Independent):\nAcc: {accuracy_score(y_va, preds):.4f} | Macro F1: {f1_score(y_va, preds, average='macro'):.4f}")
    print(classification_report(y_va, preds, target_names=CleanConfig.EMOTIONS, zero_division=0))

def main():
    seed_everything(CleanConfig.SEED)
    if os.path.exists("/content/drive/MyDrive"): root = Path("/content/drive/MyDrive/Thesis Project")
    else: root = Path(__file__).parent.parent.parent
    
    csv_r = root / "data/processed/splits/text_hc"
    # Stage 1
    df = build_audio_manifest(root, csv_r/"train.csv", csv_r/"val.csv", root/"audio_manifest.csv")
    # Stage 1.5
    df = create_speaker_independent_split(df)
    # Stage 2
    run_audio_audit(df, root/"audio_audit_report.txt")
    # Stage 3
    train_baseline(df)

if __name__ == "__main__":
    main()
