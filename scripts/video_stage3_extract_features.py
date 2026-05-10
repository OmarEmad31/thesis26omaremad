import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import sys
import csv

def sample_frames(v_path, num_frames=16):
    cap = cv2.VideoCapture(v_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return []
    
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    cap.release()
    return frames

def get_features(model, data_config, frames, device):
    import torch
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
        mean_feat = features.mean(dim=0).cpu().numpy()
        std_feat = features.std(dim=0).cpu().numpy()
        combined_feat = np.concatenate([mean_feat, std_feat])
    return combined_feat

def extract_features():
    import torch
    import timm
    
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    
    root = Path(r"d:\Thesis Project")
    manifest_path = root / "video_manifest_trackA.csv"
    
    print(f"Loading manifest from {manifest_path}...")
    resolved_samples = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['resolution_status'] == 'resolved':
                resolved_samples.append(row)
    
    print(f"Found {len(resolved_samples)} resolved samples.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    models_to_extract = [
        ("CLIP", "vit_base_patch32_clip_224"),
        ("DINOv2", "vit_base_patch14_dinov2"),
        ("ResNet50", "resnet50")
    ]

    out_root = root / "data" / "processed" / "features" / "video_v2"
    out_root.mkdir(parents=True, exist_ok=True)

    for model_name, model_id in models_to_extract:
        print(f"\n[INFO] Extracting {model_name} (Mean+Std)...")
        
        try:
            model = timm.create_model(model_id, pretrained=True, num_classes=0).to(device)
            model.eval()
            data_config = timm.data.resolve_model_data_config(model)
            
            for row in tqdm(resolved_samples, desc=model_name):
                v_path = row['resolved_video_path']
                feat_id = row['sample_id'].replace("::", "__").replace("/", "_").replace(".mp4", "")
                out_path = out_root / f"{feat_id}_{model_name.lower()}.npy"
                
                if out_path.exists():
                    continue
                    
                frames = sample_frames(v_path, num_frames=16)
                if not frames:
                    continue
                
                feat = get_features(model, data_config, frames, device)
                np.save(out_path, feat)
                
            del model
            torch.cuda.empty_cache()
            
            # Write partial manifest
            m_path = root / "video_feature_manifest_v2.csv"
            with open(m_path, "w", newline="", encoding="utf-8") as mf:
                writer = csv.DictWriter(mf, fieldnames=["sample_id", "clip_path", "dinov2_path", "resnet50_path"])
                writer.writeheader()
                for r in resolved_samples:
                    s_id = r['sample_id']
                    f_id = s_id.replace("::", "__").replace("/", "_").replace(".mp4", "")
                    entry = {"sample_id": s_id}
                    for m_n, _ in models_to_extract:
                        p = out_root / f"{f_id}_{m_n.lower()}.npy"
                        if p.exists():
                            entry[f"{m_n.lower()}_path"] = str(p)
                    writer.writerow(entry)
            
        except Exception as e:
            print(f"FAILED {model_name}: {e}")

    print(f"\nExtraction complete.")

if __name__ == "__main__":
    try:
        extract_features()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
