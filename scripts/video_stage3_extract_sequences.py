import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
import cv2
from PIL import Image

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
        ret, cap_frame = cap.read()
        if not ret: break
        if current_frame in indices_set:
            cap_frame = cv2.cvtColor(cap_frame, cv2.COLOR_BGR2RGB)
            frames_dict[current_frame] = Image.fromarray(cap_frame)
        current_frame += 1
    cap.release()
    
    frames = [frames_dict.get(idx) for idx in indices if idx in frames_dict]
    return frames

def get_sequence_features(model, data_config, frames, device):
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
        # Return the exact sequence [16, D] without mean pooling!
        seq_feat = features.cpu().numpy()
    return seq_feat

def main():
    root = Path(r"d:\Thesis Project")
    manifest_path = root / "video_manifest_trackA.csv"
    feat_dir = root / "data" / "processed" / "features" / "video_sequences_v1"
    feat_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    resolved_df = df[df['resolution_status'] == 'resolved']
    
    models_to_extract = [
        ("CLIP", "vit_base_patch32_clip_224"),
        ("DINOv2", "vit_base_patch14_dinov2"),
        ("ResNet50", "resnet50")
    ]
    
    import timm
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    for model_name, model_id in models_to_extract:
        missing_rows = []
        for _, row in resolved_df.iterrows():
            sid = row['sample_id']
            fid = sid.replace("::", "__").replace("/", "_").replace(".mp4", "")
            fpath = feat_dir / f"{fid}_{model_name.lower()}_seq.npy"
            if not fpath.exists():
                missing_rows.append(row)
                
        if not missing_rows:
            print(f"{model_name} sequence extraction is already complete. Skipping.")
            continue
            
        print(f"\nExtracting SEQUENCES [16, D] for {model_name}. {len(missing_rows)} missing.")
        
        try:
            model = timm.create_model(model_id, pretrained=True, num_classes=0).to(device)
            model.eval()
            data_config = timm.data.resolve_model_data_config(model)
            
            for i, row in enumerate(tqdm(missing_rows, desc=model_name)):
                sid = row['sample_id']
                v_path = row['resolved_video_path']
                fid = sid.replace("::", "__").replace("/", "_").replace(".mp4", "")
                fpath = feat_dir / f"{fid}_{model_name.lower()}_seq.npy"
                
                frames = sample_frames(v_path, num_frames=16)
                if frames and len(frames) == 16:
                    feat = get_sequence_features(model, data_config, frames, device)
                    tmp_fpath = fpath.with_suffix('.tmp.npy')
                    np.save(tmp_fpath, feat)
                    tmp_fpath.replace(fpath)
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"FAILED {model_name}: {e}")

if __name__ == "__main__":
    main()
