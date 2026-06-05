"""
zip_ssl_pool_videos.py
======================
Zips the raw .mp4 files for the exact 1500 unlabelled SSL pool samples
(same seed=42 selection as fusion_contrastive_v3.py).

Output: d:/Thesis Project/ssl_pool_videos.zip
Upload this to Google Drive root as: ssl_pool_videos.zip
Colab will extract and use it in Cell 2.
"""

import re, pandas as pd, zipfile
from pathlib import Path
from tqdm import tqdm

XLSX_PATH    = Path(r"d:\Thesis Project\dataset\Final Modalink Dataset MERGED\all_segments.xlsx")
DATASET_ROOT = Path(r"d:\Thesis Project\dataset\Final Modalink Dataset MERGED")
SPLITS_DIR   = Path(r"d:\Thesis Project\thesis26omaremad\data\processed\splits\multimodal_eligible")
OUT_ZIP      = Path(r"d:\Thesis Project\ssl_pool_videos.zip")
UNLABELLED_N = 1500
SEED         = 42

# Mirror load_unlabelled_pool exactly
df = pd.read_excel(str(XLSX_PATH))
unlabelled = df[df["Final Overall (majority of modalities)"].isna()].copy()
unlabelled["folder"]    = unlabelled["Folder"]
unlabelled["sample_id"] = unlabelled["Folder"] + "::" + unlabelled["video_file"]
unlabelled = unlabelled[unlabelled["transcript"].apply(
    lambda t: isinstance(t, str) and len(t.strip()) > 2)].reset_index(drop=True)

va_te = set(pd.concat([
    pd.read_csv(SPLITS_DIR / "val.csv"),
    pd.read_csv(SPLITS_DIR / "test.csv"),
])["sample_id"].values)
unlabelled = unlabelled[~unlabelled["sample_id"].isin(va_te)].reset_index(drop=True)
pool = unlabelled.sample(n=min(UNLABELLED_N, len(unlabelled)), random_state=SEED).reset_index(drop=True)
print(f"SSL pool: {len(pool)} samples")

# Collect video paths
found, missing = [], []
for _, row in pool.iterrows():
    folder = str(row["Folder"]).strip()
    vfile  = str(row["video_file"]).strip()
    p = DATASET_ROOT / folder / vfile
    if p.exists():
        found.append((p, f"ssl_pool_videos/{folder}/{vfile}"))
    else:
        missing.append(str(p))

print(f"Found: {len(found)}  |  Missing: {len(missing)}")
if missing:
    print("First 3 missing:", missing[:3])

# Zip
print(f"\nZipping {len(found)} video files → {OUT_ZIP}")
tmp = OUT_ZIP.with_suffix(".tmp.zip")
with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
    for abs_path, arcname in tqdm(found, desc="zipping"):
        zf.write(abs_path, arcname)
tmp.replace(OUT_ZIP)

mb = OUT_ZIP.stat().st_size / 1024**2
gb = mb / 1024
print(f"\nDone!  {OUT_ZIP}")
print(f"Size : {mb:.0f} MB  ({gb:.2f} GB)")
print(f"\nNEXT: Upload ssl_pool_videos.zip to Google Drive root.")
