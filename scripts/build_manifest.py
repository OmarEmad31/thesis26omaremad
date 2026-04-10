#!/usr/bin/env python3
"""
Build manifest.csv from annotations + dataset folder layout.
Run from project root: python scripts/build_manifest.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

# Paths relative to project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS_CSV = PROJECT_ROOT / "annotations" / "dataset bachelor.csv"
DATASET_ROOT = PROJECT_ROOT / "dataset" / "Final Modalink Dataset MERGED"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "manifest.csv"

# Source column -> clean manifest column name
COLUMN_RENAME = {
    "Folder": "folder",
    "segment_id": "segment_id",
    "speaker": "speaker",
    "speaker_segment_id": "speaker_segment_id",
    "start_time": "start_time_sec",
    "end_time": "end_time_sec",
    "duration": "duration_sec",
    "video_file": "video_relpath",
    "audio_file": "audio_relpath",
    "transcript": "transcript",
    "Emotion Audio (final)": "emotion_audio",
    "Text Emotion (final)": "emotion_text",
    "Video Emotion (final)": "emotion_video",
    "Final Overall (majority of modalities)": "emotion_final",
    "Audio Clarity (final)": "audio_clarity",
    "Speaker Gender (any)": "speaker_gender",
    "Speaker Identity (any)": "speaker_identity",
}


def trim_cell(s: str | None) -> str:
    if s is None:
        return ""
    return s.strip()


def load_rows() -> list[dict[str, str]]:
    if not ANNOTATIONS_CSV.is_file():
        print(f"ERROR: annotations not found: {ANNOTATIONS_CSV}", file=sys.stderr)
        sys.exit(1)
    with ANNOTATIONS_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print("ERROR: CSV has no header row.", file=sys.stderr)
            sys.exit(1)
        missing = set(COLUMN_RENAME) - set(reader.fieldnames)
        if missing:
            print(f"ERROR: CSV missing expected columns: {sorted(missing)}", file=sys.stderr)
            sys.exit(1)
        return [dict(row) for row in reader]


def build_manifest_rows(raw_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in raw_rows:
        folder = trim_cell(row["Folder"])
        vrel = trim_cell(row["video_file"])
        arel = trim_cell(row["audio_file"])

        video_path = (DATASET_ROOT / folder / vrel).resolve()
        audio_path = (DATASET_ROOT / folder / arel).resolve()

        video_exists = video_path.is_file()
        audio_exists = audio_path.is_file()

        sample_id = f"{folder}::{vrel}"

        clean: dict[str, str] = {
            "sample_id": sample_id,
        }
        for src, dst in COLUMN_RENAME.items():
            clean[dst] = trim_cell(row.get(src, ""))

        clean["video_path"] = str(video_path)
        clean["audio_path"] = str(audio_path)
        clean["video_exists"] = "1" if video_exists else "0"
        clean["audio_exists"] = "1" if audio_exists else "0"

        out.append(clean)
    return out


def write_manifest(rows: list[dict[str, str]]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("ERROR: no rows to write.", file=sys.stderr)
        sys.exit(1)

    fieldnames = [
        "sample_id",
        "folder",
        "segment_id",
        "speaker",
        "speaker_segment_id",
        "start_time_sec",
        "end_time_sec",
        "duration_sec",
        "video_relpath",
        "audio_relpath",
        "transcript",
        "emotion_audio",
        "emotion_text",
        "emotion_video",
        "emotion_final",
        "audio_clarity",
        "speaker_gender",
        "speaker_identity",
        "video_path",
        "audio_path",
        "video_exists",
        "audio_exists",
    ]

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def print_summary(rows: list[dict[str, str]]) -> None:
    n = len(rows)
    v_ok = sum(1 for r in rows if r["video_exists"] == "1")
    a_ok = sum(1 for r in rows if r["audio_exists"] == "1")
    both = sum(1 for r in rows if r["video_exists"] == "1" and r["audio_exists"] == "1")
    v_only_miss = n - v_ok
    a_only_miss = n - a_ok

    print("=== Manifest validation summary ===")
    print(f"Project root:     {PROJECT_ROOT}")
    print(f"Annotations:      {ANNOTATIONS_CSV}")
    print(f"Dataset root:     {DATASET_ROOT}")
    print(f"Output:           {OUTPUT_PATH}")
    print(f"Rows written:     {n}")
    print(f"Unique sample_id: {len({r['sample_id'] for r in rows})}")
    print()
    print(f"video_exists:     {v_ok} ({100.0 * v_ok / n:.1f}%)")
    print(f"audio_exists:     {a_ok} ({100.0 * a_ok / n:.1f}%)")
    print(f"both exist:       {both} ({100.0 * both / n:.1f}%)")
    print(f"missing video:    {v_only_miss}")
    print(f"missing audio:    {a_only_miss}")
    print()
    print(f"DATASET_ROOT exists: {DATASET_ROOT.is_dir()}")
    print("Done.")


def main() -> None:
    raw = load_rows()
    manifest = build_manifest_rows(raw)
    write_manifest(manifest)
    print_summary(manifest)


if __name__ == "__main__":
    main()
