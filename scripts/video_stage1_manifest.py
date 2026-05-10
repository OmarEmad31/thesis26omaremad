import os
import pandas as pd
from pathlib import Path

def build_video_manifest():
    root = Path(r"d:\Thesis Project")
    data_root = root / "dataset" / "Final Modalink Dataset MERGED"
    
    # Target split directory
    split_dir = root / "data" / "processed" / "splits" / "text_hc"
    
    if not split_dir.exists():
        print(f"ERROR: Split directory {split_dir} not found.")
        return

    print(f"Using split directory: {split_dir}")

    # Load splits
    train_df = pd.read_csv(split_dir / "train.csv")
    val_df = pd.read_csv(split_dir / "val.csv")
    test_df = pd.read_csv(split_dir / "test.csv")
    
    # FILTER: ONLY ELIGIBLE FOR VIDEO
    train_df = train_df[train_df['elig_video'] == 1]
    val_df = val_df[val_df['elig_video'] == 1]
    test_df = test_df[test_df['elig_video'] == 1]
    
    print(f"Initial counts (elig_video=1): Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    full_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
    results = []
    unresolved = []
    
    print(f"Processing {len(full_df)} samples...")
    
    for _, row in full_df.iterrows():
        sample_id = row['sample_id']
        folder = row['folder']
        rel_path = row['video_relpath']
        
        resolved_path = data_root / folder / rel_path
        
        status = "resolved"
        if not resolved_path.exists():
            status = "unresolved"
            resolved_path = "NOT_FOUND"

        res_row = {
            "sample_id": sample_id,
            "split": row['split'],
            "emotion_final": row['emotion_final'],
            "label_id": row.get('label_id', -1),
            "folder": folder,
            "video_relpath": rel_path,
            "resolved_video_path": str(resolved_path),
            "video_exists": 1 if status == "resolved" else 0,
            "resolution_status": status
        }
        results.append(res_row)
        if status == "unresolved":
            unresolved.append(res_row)

    manifest_df = pd.DataFrame(results)
    manifest_df.to_csv(root / "video_manifest_trackA.csv", index=False)
    
    # 4. PRINT UNRESOLVED VIDEOS BY SPLIT AND EMOTION
    if unresolved:
        un_df = pd.DataFrame(unresolved)
        un_df.to_csv(root / "unresolved_video_rows.csv", index=False)
        print("\n[AUDIT] UNRESOLVED VIDEOS BREAKDOWN:")
        print(un_df.groupby(['split', 'emotion_final']).size())
    else:
        print("\n[AUDIT] All videos resolved successfully.")
    
    # 3. VIDEO LEVEL 1 LEAKAGE AUDIT
    print("\n" + "="*50)
    print("STAGE 2: VIDEO LEAKAGE AUDIT (LEVEL 1)")
    print("="*50)
    
    train_ids = set(manifest_df[manifest_df['split'] == 'train']['sample_id'])
    val_ids = set(manifest_df[manifest_df['split'] == 'val']['sample_id'])
    test_ids = set(manifest_df[manifest_df['split'] == 'test']['sample_id'])
    
    overlap_tv = train_ids.intersection(val_ids)
    overlap_tt = train_ids.intersection(test_ids)
    overlap_vt = val_ids.intersection(test_ids)
    
    if not overlap_tv and not overlap_tt and not overlap_vt:
        print("OK: Sample IDs are distinct (No ID Leakage).")
    else:
        print(f"ERROR: ID overlap detected! TV:{len(overlap_tv)} TT:{len(overlap_tt)} VT:{len(overlap_vt)}")

    resolved_df = manifest_df[manifest_df['resolution_status'] == 'resolved']
    tr_paths = set(resolved_df[resolved_df['split'] == 'train']['resolved_video_path'])
    va_paths = set(resolved_df[resolved_df['split'] == 'val']['resolved_video_path'])
    te_paths = set(resolved_df[resolved_df['split'] == 'test']['resolved_video_path'])

    path_overlap_tv = tr_paths.intersection(va_paths)
    path_overlap_tt = tr_paths.intersection(te_paths)
    
    if not path_overlap_tv and not path_overlap_tt:
        print("OK: Resolved paths are distinct (No File Leakage).")
    else:
        print(f"ERROR: Path overlap detected! TV:{len(path_overlap_tv)} TT:{len(path_overlap_tt)}")

    # FINAL ALIGNMENT CONFIRMATION
    print("\nFINAL SPLIT ALIGNMENT (RESOLVED):")
    stats = resolved_df['split'].value_counts()
    print(stats)
    
    target_counts = {'train': 511, 'val': 64, 'test': 44}
    for s, target in target_counts.items():
        current = stats.get(s, 0)
        if current == target:
            print(f"  {s.upper():<6}: MATCH ({current})")
        else:
            print(f"  {s.upper():<6}: MISMATCH (Current: {current} | Target: {target})")

if __name__ == "__main__":
    build_video_manifest()
