"""
extract_ssl_pool_video_features.py
====================================
Extracts CLIP+DINOv2+ResNet50 sequence features [16, D] for the exact 1500
unlabelled SSL pool samples that fusion_contrastive_v3.py will use (same
filtering logic and random_state=42 seed — deterministic selection).

Output:
  d:/Thesis Project/ssl_video_features/   <-- per-sample .npy files
  d:/Thesis Project/ssl_video_features.zip <-- upload this to Google Drive root

In Colab:  place at  My Drive/ssl_video_features.zip
fusion_contrastive_v3.py will auto-extract it to /content/ssl_video_features/
before the SSL phase starts.
"""

import re
import numpy as np
import pandas as pd
import torch
import cv2
import zipfile
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
XLSX_PATH    = Path(r"d:\Thesis Project\dataset\Final Modalink Dataset MERGED\all_segments.xlsx")
DATASET_ROOT = Path(r"d:\Thesis Project\dataset\Final Modalink Dataset MERGED")
SPLITS_DIR   = Path(r"d:\Thesis Project\thesis26omaremad\data\processed\splits\multimodal_eligible")
OUT_DIR      = Path(r"d:\Thesis Project\ssl_video_features")
OUT_ZIP      = Path(r"d:\Thesis Project\ssl_video_features.zip")
UNLABELLED_N = 1500
SEED         = 42
NUM_FRAMES   = 16
MODELS_CFG   = [
    ("clip",     "vit_base_patch32_clip_224"),
    ("dinov2",   "vit_base_patch14_dinov2"),
    ("resnet50", "resnet50"),
]
# ─────────────────────────────────────────────────────────────────────────────

_FILLERS = re.compile(
    r'\b(اه|ايه|يعني|بص|كده|كدا|اهو|والله|عشان|بقا|بقى|يا|اوه|هاه|اوكي|اوكى|تمام|صح|ايوه|لا|مش|ميش)\b'
)


def sid_to_fid(sid: str) -> str:
    return sid.replace("::", "__").replace("/", "_").replace(".mp4", "")


def resolve_video(row) -> Path | None:
    folder = str(row.get("folder", row.get("Folder", ""))).strip()
    vfile  = str(row.get("video_file", "")).strip()
    if not vfile:
        return None
    for base in [DATASET_ROOT / folder, DATASET_ROOT]:
        p = base / vfile
        if p.exists():
            return p
        p2 = base / folder / vfile
        if p2.exists():
            return p2
    return None


def load_ssl_pool():
    """Exactly mirrors load_unlabelled_pool() in fusion_contrastive_v3.py."""
    df = pd.read_excel(str(XLSX_PATH))
    unlabelled = df[df["Final Overall (majority of modalities)"].isna()].copy()
    unlabelled["folder"]        = unlabelled["Folder"]
    unlabelled["audio_relpath"] = unlabelled["audio_file"]
    unlabelled["sample_id"]     = unlabelled["Folder"] + "::" + unlabelled["video_file"]
    unlabelled = unlabelled[
        unlabelled["transcript"].apply(
            lambda t: isinstance(t, str) and len(t.strip()) > 2
        )
    ].reset_index(drop=True)

    # Exclude val+test IDs (same guard as in v3)
    all_splits = pd.concat(
        [pd.read_csv(SPLITS_DIR / f"{s}.csv") for s in ("train", "val", "test")],
        ignore_index=True,
    )
    va_te_ids = set(
        pd.concat([
            pd.read_csv(SPLITS_DIR / "val.csv"),
            pd.read_csv(SPLITS_DIR / "test.csv"),
        ])["sample_id"].values
    )
    before = len(unlabelled)
    unlabelled = unlabelled[~unlabelled["sample_id"].isin(va_te_ids)].reset_index(drop=True)
    removed = before - len(unlabelled)
    if removed:
        print(f"  Removed {removed} val/test IDs from unlabelled pool")

    n_use = min(UNLABELLED_N, len(unlabelled))
    pool  = unlabelled.sample(n=n_use, random_state=SEED).reset_index(drop=True)
    print(f"  SSL pool: {n_use} samples (same selection as Colab)")
    return pool


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


def extract_features(pool: pd.DataFrame):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Check how many already done
    done = sum(1 for _, r in pool.iterrows()
               if (OUT_DIR / f"{sid_to_fid(r['sample_id'])}_clip_seq.npy").exists())
    print(f"Already extracted: {done}/{len(pool)}")
    if done == len(pool):
        print("All features already present — skipping extraction.")
        return

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
        for _, row in tqdm(pool.iterrows(), total=len(pool), desc=mname):
            fid   = sid_to_fid(row["sample_id"])
            fpath = OUT_DIR / f"{fid}_{mname}_seq.npy"
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


def build_zip():
    print(f"\nBuilding zip: {OUT_ZIP}")
    all_npy = sorted(OUT_DIR.glob("*.npy"))
    clip_count = sum(1 for p in all_npy if p.name.endswith("_clip_seq.npy"))
    print(f"  Files to zip: {len(all_npy)}  ({clip_count} samples)")

    tmp_zip = OUT_ZIP.with_suffix(".tmp.zip")
    with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for i, p in enumerate(tqdm(all_npy, desc="zipping"), 1):
            zf.write(p, f"ssl_video_features/{p.name}")
            if i % 500 == 0:
                mb = tmp_zip.stat().st_size / 1024**2
                print(f"  {i}/{len(all_npy)} files  ({mb:.0f} MB so far)")

    tmp_zip.replace(OUT_ZIP)
    mb = OUT_ZIP.stat().st_size / 1024**2
    print(f"\n{'='*60}")
    print(f"  Zip built  : {OUT_ZIP}")
    print(f"  Size       : {mb:.0f} MB")
    print(f"  Samples    : {clip_count}")
    print(f"{'='*60}")
    print()
    print("NEXT STEP:")
    print("  Upload  ssl_video_features.zip  to the ROOT of your Google Drive.")
    print("  (My Drive/ssl_video_features.zip)")
    print("  fusion_contrastive_v3.py will auto-extract it before SSL training.")


if __name__ == "__main__":
    print("Loading SSL pool (same 1500 samples Colab will use)...")
    pool = load_ssl_pool()
    extract_features(pool)
    build_zip()
