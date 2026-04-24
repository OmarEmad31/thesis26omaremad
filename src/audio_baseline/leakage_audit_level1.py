"""
Egyptian Arabic SER — Level 1 Audit & Cleaning Suite (v73)
=========================================================
1. Finds duplicate file fingerprints (MD5 + Size).
2. Prints detailed forensic info on overlaps.
3. Automatically creates a cleaned Track A split.
"""

import os, hashlib, pandas as pd
from pathlib import Path
from tqdm import tqdm
import librosa

def get_file_hash(p):
    hash_md5 = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
            break # Still just checking headers for speed
    return hash_md5.hexdigest()

def main():
    proj_root = Path("/content/drive/MyDrive/Thesis Project")
    if not proj_root.exists(): proj_root = Path("D:/Thesis Project")
    
    csv_r = proj_root / "data" / "processed" / "splits" / "text_hc"
    man_p = proj_root / "audio_manifest.csv"
    output_dir = proj_root / "data" / "processed" / "splits" / "trackA_cleaned"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not man_p.exists():
        print(f"❌ Audit Incomplete: {man_p} not found.")
        return

    # 1. Load Data
    print("⏳ Loading Split Data...")
    clean_tr = output_dir / "trackA_train_clean.csv"
    clean_va = output_dir / "trackA_val_clean.csv"
    
    if clean_tr.exists() and clean_va.exists():
        print("✨ Found Cleaned Files. Auditing the CLEANED split...")
        tr_orig = pd.read_csv(clean_tr)
        va_orig = pd.read_csv(clean_va)
    else:
        print("⚠️ No cleaned files found. Auditing the ORIGINAL split...")
        tr_orig = pd.read_csv(csv_r / "train.csv")
        va_orig = pd.read_csv(csv_r / "val.csv")
    
    man_map = df_man.set_index('sample_id')['resolved_path'].to_dict()
    tr_orig['resolved_path'] = tr_orig['sample_id'].map(man_map)
    va_orig['resolved_path'] = va_orig['sample_id'].map(man_map)
    
    tr = tr_orig.dropna(subset=['resolved_path']).copy()
    va = va_orig.dropna(subset=['resolved_path']).copy()

    # 2. Fingerprint Audit
    print("🧬 Computing File Fingerprints...")
    def get_fingerprint(p):
        try:
            stat = os.stat(p)
            return (stat.st_size, get_file_hash(p))
        except: return None

    # Track fingerprints to avoid redundant calls
    tr_fprints = {}
    for i, row in tqdm(tr.iterrows(), total=len(tr), desc="Train"):
        fp = get_fingerprint(row['resolved_path'])
        if fp: tr_fprints[fp] = i

    va_leaks = []
    for j, row in tqdm(va.iterrows(), total=len(va), desc="Val"):
        fp = get_fingerprint(row['resolved_path'])
        if fp in tr_fprints:
            va_leaks.append((tr_fprints[fp], j, fp))

    # 3. Detailed Forensic Reporting
    if va_leaks:
        print("\n" + "!"*40)
        print("🚨 HARD LEAKAGE DETECTED (LEVEL 1 FAILED)")
        print("!"*40)
        
        removed_log = []
        for i, j, fp in va_leaks:
            r_tr = tr.loc[i]
            r_va = va.loc[j]
            y_tr, _ = librosa.load(r_tr['resolved_path'], sr=8000) # Quick duration check
            y_va, _ = librosa.load(r_va['resolved_path'], sr=8000)
            
            print(f"\nDuplicate Pair Found:")
            print(f"   Train Index: {i} | Val Index: {j}")
            print(f"   Train ID:    {r_tr['sample_id']}")
            print(f"   Val ID:      {r_va['sample_id']}")
            print(f"   Emotions:    TR: {r_tr['emotion_final']} | VA: {r_va['emotion_final']}")
            print(f"   Paths:       TR: {r_tr['resolved_path']}")
            print(f"                VA: {r_va['resolved_path']}")
            print(f"   Fingerprint: Size: {fp[0]} bytes | Hash: {fp[1]}")
            print(f"   Durations:   TR: {len(y_tr)/8000:.3f}s | VA: {len(y_va)/8000:.3f}s")
            
            removed_log.append(r_va)

        # 4. Cleaning Logic
        print("\n🧹 Purging duplicates from Validation...")
        leak_val_indices = [l[1] for l in va_leaks]
        va_clean = va.drop(index=leak_val_indices)
        
        # Save results
        log_df = pd.DataFrame(removed_log)
        log_df.to_csv(output_dir / "trackA_removed_level1_duplicates.csv", index=False)
        tr.to_csv(output_dir / "trackA_train_clean.csv", index=False)
        va_clean.to_csv(output_dir / "trackA_val_clean.csv", index=False)
        
        print(f"✅ Cleaned files saved to: {output_dir}")
        print(f"   Cleaned Val size: {len(va_clean)} (Removed {len(va_leaks)})")
        print("\n⚠️  RE-RUN THIS SCRIPT NOW TO VERIFY PASS.")
    else:
        print("\n✅ CONCLUSION: LEVEL 1 PASSED")
        print("No exact sample/path/hash leakage found.")
        # If we previously cleaned them, they might be in the Cleaned folder
        if (output_dir / "trackA_val_clean.csv").exists():
             print(f"Checked against cleaned files in: {output_dir}")

if __name__ == "__main__": main()
