"""
Egyptian Arabic SER — Self-Healing Manifest Builder
===================================================
Scans the actual filesystem of the machine and updates the 
audio_manifest.csv with current working paths.
"""

import os, pandas as pd, zipfile
from pathlib import Path
from tqdm import tqdm

def main():
    proj_root = Path("/content/drive/MyDrive/Thesis Project")
    if not proj_root.exists(): proj_root = Path("D:/Thesis Project")
    
    csv_r = proj_root / "data/processed/splits/text_hc"
    man_p = proj_root / "audio_manifest.csv"
    
    # 1. Discover ALL wav files on the machine
    print("🔭 Scanning filesystem for WAV files (this might take a minute)...")
    search_roots = [Path("/content/dataset"), proj_root]
    
    # Check for zip extraction if dataset is empty
    local_ds = Path("/content/dataset")
    if not local_ds.exists() or len(list(local_ds.rglob("*.wav"))) < 100:
        zname = "Thesis_Audio_Full.zip"
        zpath = None
        for root, _, files in os.walk("/content/drive/MyDrive"):
            if zname in files: zpath = Path(root)/zname; break
        if zpath:
            print(f"📦 Found Zip at {zpath}. Extracting to /content/dataset...")
            with zipfile.ZipFile(zpath, 'r') as z: z.extractall(local_ds)
    
    physical_map = {}
    for r in search_roots:
        if not r.exists(): continue
        for p in r.rglob("*.wav"):
            # Key: folder/audios/speaker/file.wav
            parts = p.parts
            for i, part in enumerate(parts):
                if part.startswith("videoplayback"):
                    key = "/".join(parts[i:]).replace("\\", "/")
                    physical_map[key] = str(p)
                    break

    print(f"✅ Found {len(physical_map)} unique video playback segments on disk.")

    # 2. Update Manifest
    print("📝 Re-linking manifest with working paths...")
    tr = pd.read_csv(csv_r/"train.csv")
    va = pd.read_csv(csv_r/"val.csv")
    df = pd.concat([tr, va], ignore_index=True)
    
    resolved, status = [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        key = f"{row['folder']}/{row['audio_relpath']}".replace("\\", "/")
        abs_p = physical_map.get(key)
        resolved.append(abs_p)
        status.append("Resolved" if abs_p else "Unresolved")
        
    df["resolved_path"] = resolved
    df["resolution_status"] = status
    df["label_id"] = df["emotion_final"].map({'Anger':0, 'Disgust':1, 'Fear':2, 'Happiness':3, 'Neutral':4, 'Sadness':5, 'Surprise':6})
    
    df = df[df["resolution_status"] == "Resolved"].copy()
    df.to_csv(man_p, index=False)
    print(f"\n✨ Manifest Healed! {len(df)} samples linked successfully.")
    print(f"   Saved to: {man_p}")

if __name__ == "__main__": main()
