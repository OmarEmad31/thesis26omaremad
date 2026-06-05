"""
rebuild_video_sequences_zip.py
==============================
1. Finds the 202 samples in the multimodal_eligible splits that are missing
   from data/processed/features/video_sequences_v1/ (CLIP+DINOv2+ResNet50 seq).
2. Extracts [16, D] sequence features for each missing sample.
3. Rebuilds d:/Thesis Project/video_sequences_v1.zip from the full directory.

Upload the zip to the ROOT of Google Drive as  video_sequences_v1.zip
The Colab auto_detect() will find it at /content/drive/MyDrive/video_sequences_v1.zip
and extract it automatically before training starts.
"""

import numpy as np
import pandas as pd
import torch
import cv2
import zipfile
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
SPLITS_DIR   = Path(r"d:\Thesis Project\thesis26omaremad\data\processed\splits\multimodal_eligible")
DATASET_ROOT = Path(r"d:\Thesis Project\dataset\Final Modalink Dataset MERGED")
FEAT_DIR     = Path(r"d:\Thesis Project\data\processed\features\video_sequences_v1")
OUT_ZIP      = Path(r"d:\Thesis Project\video_sequences_v1.zip")
NUM_FRAMES   = 16
MODELS_CFG   = [
    ("clip",     "vit_base_patch32_clip_224"),
    ("dinov2",   "vit_base_patch14_dinov2"),
    ("resnet50", "resnet50"),
]
# ─────────────────────────────────────────────────────────────────────────────


def sid_to_fid(sid: str) -> str:
    return sid.replace("::", "__").replace("/", "_").replace(".mp4", "")


def resolve_video(row) -> Path | None:
    folder = str(row.get("folder", "")).strip()
    relpath = str(row.get("video_relpath", "")).strip()
    if not relpath:
        return None
    for base in [DATASET_ROOT / folder, DATASET_ROOT]:
        p = base / relpath
        if p.exists():
            return p
    return None


def sample_frames(v_path: Path, n: int = NUM_FRAMES) -> list:
    cap = cv2.VideoCapture(str(v_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    idxs = set(np.linspace(0, total - 1, n).astype(int))
    buf, cur = {}, 0
    while cur <= max(idxs):
        ret, f = cap.read()
        if not ret:
            break
        if cur in idxs:
            buf[cur] = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cur += 1
    cap.release()
    frames = [buf[i] for i in sorted(buf.keys())]
    while len(frames) < n:
        frames.append(frames[-1] if frames else Image.new("RGB", (224, 224)))
    return frames[:n]


def extract_missing():
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Collect all split rows
    all_df = pd.concat(
        [pd.read_csv(SPLITS_DIR / f"{s}.csv") for s in ("train", "val", "test")],
        ignore_index=True,
    )

    # Find missing (check only clip_seq — if it exists, all three assumed present)
    missing = all_df[
        ~all_df["sample_id"].apply(
            lambda s: (FEAT_DIR / f"{sid_to_fid(s)}_clip_seq.npy").exists()
        )
    ].reset_index(drop=True)

    if missing.empty:
        print("All samples already extracted — nothing to do.")
        return

    print(f"Missing: {len(missing)} / {len(all_df)} samples")

    import timm
    from torchvision import transforms

    for mname, mid in MODELS_CFG:
        print(f"\n[{mname}] Loading {mid} ...")
        model = timm.create_model(mid, pretrained=True, num_classes=0).to(device)
        model.eval()
        cfg = timm.data.resolve_model_data_config(model)
        tf = transforms.Compose([
            transforms.Resize(cfg["input_size"][1:]),
            transforms.CenterCrop(cfg["input_size"][1:]),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg["mean"], std=cfg["std"]),
        ])

        ok = skip = 0
        for _, row in tqdm(missing.iterrows(), total=len(missing), desc=mname):
            fid   = sid_to_fid(row["sample_id"])
            fpath = FEAT_DIR / f"{fid}_{mname}_seq.npy"
            if fpath.exists():
                ok += 1
                continue
            vp = resolve_video(row)
            if vp is None:
                skip += 1
                continue
            frames = sample_frames(vp)
            if not frames:
                skip += 1
                continue
            batch = torch.stack([tf(f) for f in frames]).to(device)
            with torch.no_grad():
                feat = model(batch)
                if feat.dim() > 2:
                    feat = feat.mean(dim=[2, 3])
            tmp = fpath.with_suffix(".tmp.npy")
            np.save(str(tmp), feat.cpu().numpy())
            tmp.replace(fpath)
            ok += 1

        print(f"  [{mname}] saved={ok}  skipped={skip}")
        del model
        torch.cuda.empty_cache()

    print("\nExtraction complete.")


def rebuild_zip():
    print(f"\nRebuilding zip: {OUT_ZIP}")
    all_npy = sorted(FEAT_DIR.glob("*.npy"))
    print(f"  Files to zip: {len(all_npy)}")

    tmp_zip = OUT_ZIP.with_suffix(".tmp.zip")
    with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for i, p in enumerate(tqdm(all_npy, desc="zipping"), 1):
            zf.write(p, f"video_sequences_v1/{p.name}")
            if i % 500 == 0:
                mb = tmp_zip.stat().st_size / 1024**2
                print(f"  {i}/{len(all_npy)} files  ({mb:.0f} MB so far)")

    tmp_zip.replace(OUT_ZIP)
    mb = OUT_ZIP.stat().st_size / 1024**2
    clip_count = sum(1 for p in all_npy if p.name.endswith("_clip_seq.npy"))
    print(f"\n{'='*60}")
    print(f"  Zip rebuilt: {OUT_ZIP}")
    print(f"  Size       : {mb:.0f} MB")
    print(f"  Samples    : {clip_count} (clip_seq count)")
    print(f"{'='*60}")
    print()
    print("NEXT STEP:")
    print("  Upload  video_sequences_v1.zip  to the ROOT of your Google Drive.")
    print("  (should land at: My Drive/video_sequences_v1.zip)")
    print("  Colab auto_detect() will find and extract it automatically.")


if __name__ == "__main__":
    extract_missing()
    rebuild_zip()
