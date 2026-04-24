"""
Build Audio High-Confidence (HC) Splits.

Filtering rules for Audio:
R1. emotion_final is a valid 7-class label.
R2. audio_clarity == "Clear" (Crucial for Audio SER).
R3. cross-modal consensus: at least 2 out of 3 annotator labels
    match emotion_final. Guards against noisy labels.
    
Crucially, this EXCLUDES the Text length rule (R1 from text_hc). 
An intense scream or sob might have 0 words but perfectly valid audio emotion. 
Filtering by text length actively throws away the best audio files!
"""

import csv
from collections import Counter
from pathlib import Path

SRC_DIR  = Path("data/processed/splits/audio_eligible")
DST_DIR  = Path("data/processed/splits/audio_hc")
DST_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "val", "test"]

VALID_LABELS = {
    "Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"
}

def count_annotator_agreement(row: dict, final: str) -> tuple[int, int]:
    """Return (matching_votes, total_non_empty_annotators)."""
    votes = [
        row.get("emotion_audio", "").strip(),
        row.get("emotion_text",  "").strip(),
        row.get("emotion_video", "").strip(),
    ]
    non_empty = [v for v in votes if v]
    matching  = [v for v in non_empty if v == final]
    return len(matching), len(non_empty)

def passes_filters(row: dict) -> tuple[bool, str]:
    final      = (row.get("emotion_final") or "").strip()
    clarity    = (row.get("audio_clarity") or "").strip()

    # R1 – valid label
    if final not in VALID_LABELS:
        return False, "invalid_label"

    # R2 – audio clarity
    if clarity != "Clear":
        return False, "noisy_audio"

    # R3 – cross-modal consensus (≥2 annotators agree with emotion_final)
    matching, total = count_annotator_agreement(row, final)
    if total < 2 or matching < 2:
        return False, f"low_consensus(match={matching},total={total})"

    # Notice: No Word Count filtering!
    return True, "ok"

for split in SPLITS:
    src = SRC_DIR / f"{split}.csv"
    if not src.exists():
        print(f"Skipping {split}, {src} not found.")
        continue
        
    dst = DST_DIR / f"{split}.csv"

    with src.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        all_rows   = list(reader)

    kept, skip_counts = [], Counter()
    for row in all_rows:
        ok, reason = passes_filters(row)
        if ok:
            kept.append(row)
        else:
            skip_counts[reason] += 1

    with dst.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    label_dist = Counter(r["emotion_final"] for r in kept)
    print(f"\n=== {split}: {len(all_rows)} -> {len(kept)} kept "
          f"({len(all_rows)-len(kept)} dropped) ===")
    print(f"  Skip reasons: {dict(skip_counts)}")
    print(f"  Label dist:   {dict(sorted(label_dist.items()))}")

print(f"\nDone. HC Audio splits written to: {DST_DIR.resolve()}")
