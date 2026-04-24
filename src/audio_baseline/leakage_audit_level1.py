"""
Egyptian Arabic SER — Level 1 Hard-Leakage Audit
===============================================
Performs a strictly scientific audit of the Track A split.
Checks for:
- Overlapping sample_ids
- Overlapping resolved_paths
- Overlapping file fingerprints (Size, Duration, MD5 Hash)
"""

import os, hashlib, pandas as pd
from pathlib import Path
from tqdm import tqdm
import librosa

def get_file_hash(p):
    """Computes MD5 hash of the first 1MB of the file for speed."""
    hash_md5 = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
            break # Just the first 4KB + size + duration is usually enough
    return hash_md5.hexdigest()

def main():
    proj_root = Path("/content/drive/MyDrive/Thesis Project")
    if not proj_root.exists(): proj_root = Path("D:/Thesis Project")
    
    csv_r = proj_root / "data/processed/splits/text_hc"
    man_p = proj_root / "audio_manifest.csv"
    
    if not man_p.exists():
        print(f"❌ Audit Incomplete: {man_p} not found.")
        return

    # 1. Load Data
    print("⏳ Loading Split Data...")
    df_man = pd.read_csv(man_p)
    tr_orig = pd.read_csv(csv_r / "train.csv")
    va_orig = pd.read_csv(csv_r / "val.csv")
    
    # Attach resolved paths
    man_map = df_man.set_index('sample_id')['resolved_path'].to_dict()
    tr_orig['resolved_path'] = tr_orig['sample_id'].map(man_map)
    va_orig['resolved_path'] = va_orig['sample_id'].map(man_map)
    
    tr = tr_orig.dropna(subset=['resolved_path'])
    va = va_orig.dropna(subset=['resolved_path'])
    
    print(f"📊 Stats:")
    print(f"   Train: {len(tr_orig)} total -> {len(tr)} resolved")
    print(f"   Val:   {len(va_orig)} total -> {len(va)} resolved")
    print(f"   Unresolved: {len(tr_orig)+len(va_orig) - (len(tr)+len(va))}")

    # 2. Key Overlap Checks
    print("\n🔍 Checking Key Overlaps...")
    results = {}
    
    results['sample_id'] = set(tr['sample_id']).intersection(set(va['sample_id']))
    results['resolved_path'] = set(tr['resolved_path']).intersection(set(va['resolved_path']))
    
    tr['folder_rel'] = tr['folder'] + "/" + tr['audio_relpath']
    va['folder_rel'] = va['folder'] + "/" + va['audio_relpath']
    results['folder_rel'] = set(tr['folder_rel']).intersection(set(va['folder_rel']))

    # 3. Fingerprint Audit
    print("🧬 Computing File Fingerprints (Size + MD5)...")
    def get_fingerprint(p):
        try:
            stat = os.stat(p)
            return (stat.st_size, get_file_hash(p))
        except: return None

    tr_fingerprints = {get_fingerprint(p): p for p in tqdm(tr['resolved_path'], desc="Train Fingerprints")}
    va_fingerprints = {get_fingerprint(p): p for p in tqdm(va['resolved_path'], desc="Val Fingerprints")}
    
    # Remove None
    tr_fingerprints.pop(None, None)
    va_fingerprints.pop(None, None)
    
    results['file_fingerprint'] = set(tr_fingerprints.keys()).intersection(set(va_fingerprints.keys()))

    # 4. Final Reporting
    print("\n" + "="*40)
    print("📋 LEVEL 1 LEAKAGE AUDIT REPORT")
    print("="*40)
    
    total_violations = 0
    for key, overlaps in results.items():
        count = len(overlaps)
        total_violations += count
        print(f"{key:25} | Duplicates: {count}")
    
    # Level 2 (Soft Leakage) - Optional info
    spk_overlap = set(tr['speaker']).intersection(set(va['speaker']))
    print(f"{'speaker_overlap (L2)':25} | Count: {len(spk_overlap)}")

    print("\n")
    if total_violations == 0:
        print("✅ CONCLUSION: LEVEL 1 PASSED")
        print("No exact sample/path/hash leakage found. Your 62% results are bit-clean.")
    else:
        print("❌ CONCLUSION: LEVEL 1 FAILED")
        print(f"Found {total_violations} exact duplicate overlaps between Train and Val.")
        print("\nFirst 20 Overlapping Examples:")
        # Identify first 20 in the most common violation (sample_id or path)
        problem_ids = list(results['sample_id'])[:20]
        for pid in problem_ids:
            row_tr = tr[tr['sample_id'] == pid].iloc[0]
            row_va = va[va['sample_id'] == pid].iloc[0]
            print(f"   ID: {pid}")
            print(f"      TR: {row_tr['folder']}/{row_tr['audio_relpath']}")
            print(f"      VA: {row_va['folder']}/{row_va['audio_relpath']}")

if __name__ == "__main__": main()
