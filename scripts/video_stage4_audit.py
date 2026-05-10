import os
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

def extract_uniform_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, 0, 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    if total_frames <= 0:
        cap.release()
        return None, 0, 0

    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            # If a frame fails, try to pad with the last one or black
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
                
    cap.release()
    return frames, duration, total_frames

def run_video_audit():
    root = Path(r"d:\Thesis Project")
    manifest_path = root / "video_manifest_trackA.csv"
    if not manifest_path.exists():
        print("ERROR: Manifest not found.")
        return

    df = pd.read_csv(manifest_path)
    resolved_df = df[df['resolution_status'] == 'resolved']
    
    output_dir = root / "outputs" / "video_audit"
    contact_dir = output_dir / "contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)
    
    audit_results = []
    print(f"Starting Preprocessing Audit on {len(resolved_df)} videos...")
    
    # Audit a representative sample (or all if small)
    # Since we have ~600, let's audit ALL to be thesis-ready.
    for i, row in tqdm(resolved_df.iterrows(), total=len(resolved_df)):
        v_path = row['resolved_video_path']
        frames, duration, total_frames = extract_uniform_frames(v_path, num_frames=8)
        
        status = "ok" if frames is not None else "failed"
        
        audit_results.append({
            "sample_id": row['sample_id'],
            "split": row['split'],
            "emotion": row['emotion_final'],
            "duration": duration,
            "total_frames": total_frames,
            "status": status
        })
        
        # Save first 20 as contact sheets
        if i < 20 and frames is not None:
            fig, axes = plt.subplots(2, 4, figsize=(15, 8))
            for ax, f in zip(axes.flatten(), frames):
                ax.imshow(f)
                ax.axis('off')
            plt.suptitle(f"Sample: {row['sample_id']} | Emotion: {row['emotion_final']}")
            plt.savefig(contact_dir / f"contact_{i:02d}.png")
            plt.close()

    audit_df = pd.DataFrame(audit_results)
    audit_df.to_csv(output_dir / "video_audit_report.csv", index=False)
    
    # Summary
    print("\n" + "="*50)
    print("VIDEO PREPROCESSING AUDIT SUMMARY")
    print("="*50)
    
    for split in ['train', 'val', 'test']:
        s_df = audit_df[audit_df['split'] == split]
        if len(s_df) > 0:
            print(f"\nSPLIT: {split.upper()}")
            print(f"  Readable: {len(s_df[s_df['status'] == 'ok'])} / {len(s_df)}")
            print(f"  Duration: Min {s_df['duration'].min():.2f}s | Mean {s_df['duration'].mean():.2f}s | Max {s_df['duration'].max():.2f}s")
            print(f"  Total Frames: Min {s_df['total_frames'].min()} | Mean {s_df['total_frames'].mean():.1f}")

    print(f"\nAudit Complete. Results saved to: {output_dir}")

if __name__ == "__main__":
    run_video_audit()
