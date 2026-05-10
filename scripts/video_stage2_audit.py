import os
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
import json

def audit_videos():
    root = Path(r"d:\Thesis Project")
    manifest_path = root / "video_manifest_trackA.csv"
    
    if not manifest_path.exists():
        print("ERROR: Run video_stage1_manifest.py first.")
        return

    df = pd.read_csv(manifest_path)
    resolved_df = df[df['resolution_status'] == 'resolved'].copy()
    
    print(f"Auditing {len(resolved_df)} resolved videos...")
    
    audit_results = []
    
    for _, row in tqdm(resolved_df.iterrows(), total=len(resolved_df)):
        v_path = row['resolved_video_path']
        cap = cv2.VideoCapture(v_path)
        
        if not cap.isOpened():
            audit_results.append({
                "sample_id": row['sample_id'],
                "status": "corrupt",
                "width": 0, "height": 0, "fps": 0, "frames": 0, "duration": 0
            })
            continue
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        audit_results.append({
            "sample_id": row['sample_id'],
            "status": "ok",
            "width": width,
            "height": height,
            "fps": fps,
            "frames": frame_count,
            "duration": duration
        })
        cap.release()

    audit_df = pd.DataFrame(audit_results)
    final_df = df.merge(audit_df, on="sample_id", how="left")
    
    report_path = root / "video_audit_report.csv"
    final_df.to_csv(report_path, index=False)
    print(f"Audit report saved to {report_path}")
    
    # Summary
    print("\nAUDIT SUMMARY:")
    print(f"Total Samples: {len(df)}")
    print(f"Resolved: {len(resolved_df)}")
    print(f"Corrupt: {len(audit_df[audit_df['status'] == 'corrupt'])}")
    
    print("\nClass Distribution (Resolved):")
    print(resolved_df['emotion_final'].value_counts())
    
    print("\nResolution Stats:")
    print(final_df[final_df['status'] == 'ok'][['width', 'height']].value_counts())

if __name__ == "__main__":
    audit_videos()
