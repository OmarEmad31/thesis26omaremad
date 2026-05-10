import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import cv2

def extract_uniform_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret: frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        else: frames.append(frames[-1] if frames else Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)))
    cap.release()
    return frames

def run_feature_extraction():
    root = Path(r"d:\Thesis Project")
    df = pd.read_csv(root / "video_manifest_trackA.csv")
    resolved_df = df[df['resolution_status'] == 'resolved']
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP on {device}...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    feature_dir = root / "data" / "processed" / "video_features_clip"
    feature_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting features for {len(resolved_df)} samples...")
    for _, row in tqdm(resolved_df.iterrows(), total=len(resolved_df)):
        sample_id = row['sample_id'].replace("::", "__").replace("/", "_")
        feat_path = feature_dir / f"{sample_id}.npy"
        
        if feat_path.exists(): continue
        
        frames = extract_uniform_frames(row['resolved_video_path'])
        if frames is None: continue
        
        inputs = processor(images=frames, return_tensors="pt").to(device)
        with torch.no_grad():
            img_features = model.get_image_features(**inputs) # [8, 512]
        
        np.save(feat_path, img_features.cpu().numpy())

    print(f"Extraction complete. Features saved to: {feature_dir}")

if __name__ == "__main__":
    run_feature_extraction()
