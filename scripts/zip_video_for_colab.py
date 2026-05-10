"""
zip_video_for_colab.py
======================
Packs ONLY the video clips referenced by the video_eligible splits
(train / val / test) plus the split CSVs and the colab_train.py script.

Output:  D:/Thesis_Video_Colab.zip   (~varies by dataset size)

Structure inside the zip:
  Thesis_Video/
    splits/
      train.csv
      val.csv
      test.csv
    colab_train.py
    dataset/
      videoplayback (1)/
        videos/
          SPEAKER_00/
            SPEAKER_00_segment_0000.mp4
          ...
      videoplayback (2)/
        ...
"""

import csv
import os
import zipfile
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(r"D:\Thesis Project")
DATASET_ROOT  = PROJECT_ROOT / "dataset" / "Final Modalink Dataset MERGED"
SPLITS_DIR    = PROJECT_ROOT / "data" / "processed" / "splits" / "video_eligible"
COLAB_SCRIPT  = PROJECT_ROOT / "src" / "video_swin" / "colab_train.py"
OUTPUT_ZIP    = Path(r"D:\ThesisWork\Thesis_Video_Colab.zip")
# ─────────────────────────────────────────────────────────────────────────────


def collect_video_paths():
    """Return set of (absolute_path, arcname) for every video in all splits."""
    paths = []
    seen  = set()
    missing = []

    for split in ["train", "val", "test"]:
        csv_file = SPLITS_DIR / f"{split}.csv"
        with open(csv_file, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("elig_video", "0").strip() != "1":
                    continue
                folder  = row.get("folder", "").strip()
                relpath = row.get("video_relpath", "").strip()
                if not folder or not relpath:
                    continue
                abs_path = DATASET_ROOT / folder / relpath
                if abs_path in seen:
                    continue
                seen.add(abs_path)
                if abs_path.exists():
                    # arcname inside zip: dataset/videoplayback (X)/videos/...
                    arcname = f"Thesis_Video/dataset/{folder}/{relpath}"
                    paths.append((abs_path, arcname))
                else:
                    missing.append(str(abs_path))

    if missing:
        print(f"WARNING: {len(missing)} files not found on disk (will be skipped):")
        for m in missing[:10]:
            print(f"  {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    return paths


def main():
    if OUTPUT_ZIP.exists():
        print(f"Removing old zip: {OUTPUT_ZIP}")
        OUTPUT_ZIP.unlink()

    video_paths = collect_video_paths()
    print(f"\nCollected {len(video_paths)} unique video files across all splits.")

    print(f"Creating zip: {OUTPUT_ZIP}")
    added = 0

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:

        # 1. Split CSVs
        for split in ["train", "val", "test"]:
            src = SPLITS_DIR / f"{split}.csv"
            zf.write(src, f"Thesis_Video/splits/{split}.csv")
            print(f"  Added splits/{split}.csv")

        # 2. Colab training script
        zf.write(COLAB_SCRIPT, "Thesis_Video/colab_train.py")
        print(f"  Added colab_train.py")

        # 3. Video files
        print(f"\nZipping {len(video_paths)} video files (this may take a few minutes)...")
        for abs_path, arcname in video_paths:
            zf.write(abs_path, arcname)
            added += 1
            if added % 200 == 0:
                size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
                print(f"  {added}/{len(video_paths)} videos zipped  ({size_mb:.0f} MB so far)")

    final_size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    final_size_gb = final_size_mb / 1024

    print(f"\n{'='*60}")
    print(f"SUCCESS!")
    print(f"  Videos packed : {added}")
    print(f"  Output file   : {OUTPUT_ZIP}")
    print(f"  Size          : {final_size_mb:.0f} MB  ({final_size_gb:.2f} GB)")
    print(f"{'='*60}")
    print(f"\nNEXT STEPS:")
    print(f"  1. Upload Thesis_Video_Colab.zip to the ROOT of your Google Drive")
    print(f"  2. Open Colab → paste the cells from colab_notebook_cells.py")
    print(f"  3. Run all cells in order")


if __name__ == "__main__":
    main()
