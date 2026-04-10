#!/usr/bin/env python3
"""
Stratified train/val/test split on emotion_final from manifest.csv.
Run from project root: python scripts/split_dataset.py
"""

from __future__ import annotations

import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "manifest.csv"
OUTPUT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "manifest_with_split.csv"
SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
RANDOM_SEED = 42


def _truthy_exists(val: str) -> bool:
    v = (val or "").strip().lower()
    return v in ("1", "true", "yes")


def load_manifest() -> list[dict[str, str]]:
    if not MANIFEST_PATH.is_file():
        print(f"ERROR: manifest not found: {MANIFEST_PATH}", file=sys.stderr)
        sys.exit(1)
    with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def filter_emotion_final(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for r in rows:
        ef = (r.get("emotion_final") or "").strip()
        if ef:
            r = dict(r)
            r["emotion_final"] = ef
            out.append(r)
    return out


def stratified_train_val_test(
    labels: list[str],
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
    seed: int = RANDOM_SEED,
) -> list[str]:
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train/val/test fractions must sum to 1")

    n = len(labels)
    rng = random.Random(seed)
    by_label: dict[str, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        by_label[lab].append(i)

    split_names: list[str] = [""] * n

    for lab in sorted(by_label.keys()):
        idxs = by_label[lab][:]
        rng.shuffle(idxs)
        n_c = len(idxs)
        n_train = int(round(n_c * train_frac))
        n_train = max(0, min(n_c, n_train))
        rem = n_c - n_train
        if rem == 0:
            n_val, n_test = 0, 0
        elif rem == 1:
            # single remainder: assign to val (arbitrary but deterministic)
            n_val, n_test = 1, 0
        else:
            n_val = int(round(rem * (val_frac / (val_frac + test_frac))))
            n_val = max(0, min(rem, n_val))
            n_test = rem - n_val

        train_ids = idxs[:n_train]
        mid = idxs[n_train : n_train + n_val]
        test_ids = idxs[n_train + n_val :]

        for i in train_ids:
            split_names[i] = "train"
        for i in mid:
            split_names[i] = "val"
        for i in test_ids:
            split_names[i] = "test"

    if any(s == "" for s in split_names):
        raise RuntimeError("internal error: missing split assignment")

    return split_names


def add_eligibility(rows: list[dict[str, str]]) -> None:
    for r in rows:
        tr = (r.get("transcript") or "").strip()
        r["elig_text"] = "1" if tr else "0"
        r["elig_audio"] = "1" if _truthy_exists(r.get("audio_exists", "")) else "0"
        r["elig_video"] = "1" if _truthy_exists(r.get("video_exists", "")) else "0"
        multimodal = r["elig_text"] == "1" and r["elig_audio"] == "1" and r["elig_video"] == "1"
        r["elig_multimodal"] = "1" if multimodal else "0"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def print_summary(rows: list[dict[str, str]], title: str) -> None:
    print(f"\n--- {title} ---")
    by_split = Counter((r.get("split") or "").strip() for r in rows)
    for s in ("train", "val", "test"):
        print(f"  {s}: {by_split.get(s, 0)}")
    labels = sorted({r.get("emotion_final", "") for r in rows})
    print("  per label (split x emotion_final):")
    for lab in labels:
        parts = [f"{s}={sum(1 for r in rows if r.get('split')==s and r.get('emotion_final')==lab)}" for s in ("train", "val", "test")]
        print(f"    {lab}: " + ", ".join(parts))


def main() -> None:
    raw = load_manifest()
    filtered = filter_emotion_final(raw)
    if not filtered:
        print("ERROR: no rows with non-empty emotion_final.", file=sys.stderr)
        sys.exit(1)

    labels = [r["emotion_final"] for r in filtered]
    splits = stratified_train_val_test(labels)
    for r, sp in zip(filtered, splits):
        r["split"] = sp

    add_eligibility(filtered)

    extra = ["split", "elig_text", "elig_audio", "elig_video", "elig_multimodal"]
    with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as f:
        base_fields = list(csv.DictReader(f).fieldnames or [])
    # Preserve manifest column order; append new fields
    all_fields = list(dict.fromkeys(list(base_fields) + extra))

    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_MANIFEST, all_fields, filtered)

    # Subsets: same split column; filter by eligibility
    def subset_rows(pred) -> list[dict[str, str]]:
        return [r for r in filtered if pred(r)]

    subsets = {
        "text_eligible": lambda r: r["elig_text"] == "1",
        "audio_eligible": lambda r: r["elig_audio"] == "1",
        "video_eligible": lambda r: r["elig_video"] == "1",
        "multimodal_eligible": lambda r: r["elig_multimodal"] == "1",
    }

    for name, pred in subsets.items():
        sub = subset_rows(pred)
        d = SPLITS_DIR / name
        for sp in ("train", "val", "test"):
            part = [r for r in sub if r.get("split") == sp]
            write_csv(d / f"{sp}.csv", all_fields, part)

    # Summary
    print("=== Split generation ===")
    print(f"Manifest in:  {MANIFEST_PATH}")
    print(f"Manifest out: {OUTPUT_MANIFEST}")
    print(f"Splits dir:   {SPLITS_DIR}")
    print(f"Rows (all with emotion_final): {len(filtered)}")
    print(f"Split fractions: train={TRAIN_FRAC}, val={VAL_FRAC}, test={TEST_FRAC}, seed={RANDOM_SEED}")

    print_summary(filtered, "All labeled rows")
    for name, pred in subsets.items():
        sub = subset_rows(pred)
        print_summary(sub, f"{name} ({len(sub)} rows)")

    # Label distribution check (global)
    print("\n--- emotion_final counts (all labeled) ---")
    for lab, c in sorted(Counter(labels).items(), key=lambda x: -x[1]):
        print(f"  {lab}: {c}")

    print("\nDone.")


if __name__ == "__main__":
    main()
