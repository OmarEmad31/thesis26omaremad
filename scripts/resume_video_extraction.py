import os
import sys
import csv
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
import cv2
from PIL import Image

def validate_features(manifest_path, feat_dir, models):
    print("\n" + "="*50)
    print("STAGE: VALIDATE FEATURE FILES")
    print("="*50)
    
    df = pd.read_csv(manifest_path)
    df = df[df['resolution_status'] == 'resolved']
    
    all_valid = True
    for model_name in models:
        print(f"\nValidating {model_name}...")
        valid_count = 0
        for _, row in df.iterrows():
            sid = row['sample_id']
            fid = sid.replace("::", "__").replace("/", "_").replace(".mp4", "")
            fpath = feat_dir / f"{fid}_{model_name.lower()}.npy"
            
            if not fpath.exists():
                print(f"  [ERROR] Missing {model_name} file for {sid}")
                all_valid = False
                continue
                
            feat = np.load(fpath)
            if np.isnan(feat).any() or np.isinf(feat).any():
                print(f"  [ERROR] NaN/Inf in {model_name} file for {sid}")
                all_valid = False
                continue
                
            # expected shape is flattened mean+std
            # check 1D
            if len(feat.shape) != 1:
                print(f"  [ERROR] Invalid shape {feat.shape} in {model_name} file for {sid}")
                all_valid = False
                continue
                
            valid_count += 1
        print(f"  {valid_count}/{len(df)} files valid for {model_name}.")
        
    return all_valid

def sample_frames(v_path, num_frames=16):
    cap = cv2.VideoCapture(v_path)
    if not cap.isOpened(): return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0: return []
    
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    indices_set = set(indices)
    frames_dict = {}
    
    current_frame = 0
    while current_frame <= max(indices):
        ret, frame = cap.read()
        if not ret: break
        if current_frame in indices_set:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_dict[current_frame] = Image.fromarray(frame)
        current_frame += 1
    cap.release()
    
    frames = [frames_dict.get(idx) for idx in indices if idx in frames_dict]
    return frames

def get_features(model, data_config, frames, device):
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(data_config['input_size'][1:]),
        transforms.CenterCrop(data_config['input_size'][1:]),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_config['mean'], std=data_config['std']),
    ])
    
    batch = torch.stack([transform(f) for f in frames]).to(device)
    with torch.no_grad():
        features = model(batch)
        if len(features.shape) > 2:
            features = features.mean(dim=[2,3])
        mean_feat = features.mean(dim=0).cpu().numpy()
        std_feat = features.std(dim=0).cpu().numpy()
        combined_feat = np.concatenate([mean_feat, std_feat])
    return combined_feat

def main():
    root = Path(r"d:\Thesis Project")
    manifest_path = root / "video_manifest_trackA.csv"
    feat_dir = root / "data" / "processed" / "features" / "video_v2"
    feat_dir.mkdir(parents=True, exist_ok=True)
    prog_file = root / "video_v2_extraction_progress.json"

    df = pd.read_csv(manifest_path)
    resolved_df = df[df['resolution_status'] == 'resolved']
    
    print("\n" + "="*50)
    print("STAGE 1: INSPECTION & CONFIRMATION")
    print("="*50)
    print(f"FINAL SPLIT SOURCE USED: {manifest_path}")
    counts = resolved_df['split'].value_counts()
    print("Train/Val/Test counts:")
    print(counts)
    
    tr_ids = set(resolved_df[resolved_df['split']=='train']['sample_id'])
    va_ids = set(resolved_df[resolved_df['split']=='val']['sample_id'])
    te_ids = set(resolved_df[resolved_df['split']=='test']['sample_id'])
    overlap_tv = tr_ids.intersection(va_ids)
    overlap_tt = tr_ids.intersection(te_ids)
    overlap_vt = va_ids.intersection(te_ids)
    print(f"Leakage check -> TV overlap: {len(overlap_tv)}, TT: {len(overlap_tt)}, VT: {len(overlap_vt)}")
    if not overlap_tv and not overlap_tt and not overlap_vt:
        print("Confirmed: no train/val/test leakage.")
        
    models_to_extract = [
        ("CLIP", "vit_base_patch32_clip_224"),
        ("DINOv2", "vit_base_patch14_dinov2"),
        ("ResNet50", "resnet50")
    ]
    
    all_models_names = [m[0] for m in models_to_extract]
    
    # Inspection logic
    for model_name, _ in models_to_extract:
        print(f"\nInspecting {model_name}...")
        missing_count = 0
        extracted_by_split = {'train': 0, 'val': 0, 'test': 0}
        missing_ids = []
        for _, row in resolved_df.iterrows():
            sid = row['sample_id']
            fid = sid.replace("::", "__").replace("/", "_").replace(".mp4", "")
            fpath = feat_dir / f"{fid}_{model_name.lower()}.npy"
            if fpath.exists():
                extracted_by_split[row['split']] += 1
            else:
                missing_count += 1
                missing_ids.append(sid)
                
        for s in ['train', 'val', 'test']:
            print(f"  {s.capitalize()} rows extracted: {extracted_by_split[s]} / {counts.get(s, 0)}")
        print(f"  Expected total: {len(resolved_df)}")
        if missing_count == 0:
            print(f"  Status: Complete")
        else:
            print(f"  Status: Incomplete")
            print(f"  Missing IDs ({missing_count}): {missing_ids[:3]}...")
            
    print("\n" + "="*50)
    print("STAGE 2: EXTRACTION RESUME")
    print("="*50)
    
    import timm
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load progress if exists
    progress = {}
    if prog_file.exists():
        with open(prog_file, "r") as f:
            progress = json.load(f)
            
    for model_name, model_id in models_to_extract:
        # check missing
        missing_rows = []
        for _, row in resolved_df.iterrows():
            sid = row['sample_id']
            fid = sid.replace("::", "__").replace("/", "_").replace(".mp4", "")
            fpath = feat_dir / f"{fid}_{model_name.lower()}.npy"
            if not fpath.exists():
                missing_rows.append(row)
                
        if not missing_rows:
            print(f"{model_name} is already complete. Skipping.")
            continue
            
        print(f"\nResuming {model_name} extraction. {len(missing_rows)} missing.")
        
        try:
            model = timm.create_model(model_id, pretrained=True, num_classes=0).to(device)
            model.eval()
            data_config = timm.data.resolve_model_data_config(model)
            
            for i, row in enumerate(tqdm(missing_rows, desc=model_name)):
                sid = row['sample_id']
                v_path = row['resolved_video_path']
                fid = sid.replace("::", "__").replace("/", "_").replace(".mp4", "")
                fpath = feat_dir / f"{fid}_{model_name.lower()}.npy"
                
                frames = sample_frames(v_path, num_frames=16)
                if frames:
                    feat = get_features(model, data_config, frames, device)
                    # Temporary save to temp path first, then rename (atomic)
                    tmp_fpath = fpath.with_suffix('.tmp.npy')
                    np.save(tmp_fpath, feat)
                    tmp_fpath.replace(fpath)
                
                # Checkpoint progress
                progress.setdefault(model_name, 0)
                progress[model_name] += 1
                if i % 10 == 0:
                    with open(prog_file, "w") as f:
                        json.dump(progress, f)
                        
            # Final checkpoint save
            with open(prog_file, "w") as f:
                json.dump(progress, f)
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"FAILED {model_name}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Rebuild Manifest
    print("\nRebuilding feature manifest v2...")
    m_path = root / "video_feature_manifest_v2.csv"
    with open(m_path, "w", newline="", encoding="utf-8") as mf:
        writer = csv.DictWriter(mf, fieldnames=["sample_id", "clip_path", "dinov2_path", "resnet50_path"])
        writer.writeheader()
        for _, row in resolved_df.iterrows():
            s_id = row['sample_id']
            f_id = s_id.replace("::", "__").replace("/", "_").replace(".mp4", "")
            entry = {"sample_id": s_id}
            for m_n in all_models_names:
                p = feat_dir / f"{f_id}_{m_n.lower()}.npy"
                if p.exists():
                    entry[f"{m_n.lower()}_path"] = str(p)
            writer.writerow(entry)

    all_valid = validate_features(manifest_path, feat_dir, all_models_names)
    
    if all_valid:
        print("\n" + "="*50)
        print("STAGE 4: EVALUATION")
        print("="*50)
        # Import and run stage 4 eval
        sys.path.append(str(root / "scripts"))
        import video_stage4_eval_baseline
        video_stage4_eval_baseline.run_eval()
    else:
        print("Validation failed. Aborting evaluation.")

if __name__ == "__main__":
    main()
