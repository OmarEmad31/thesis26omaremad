"""
Build high-confidence (HC) text splits.

Filtering rules (applied to train, val, and test independently):

R1. transcript >= 5 words after strip  (text must carry real content)
R2. emotion_final is a known 7-class label (not empty, not "Amiguous")
R3. audio_clarity == "Clear"            (noisy audio → unreliable transcription)
R4. cross-modal consensus: at least 2 out of 3 annotator labels
    {emotion_audio, emotion_text, emotion_video} are non-empty AND
    match emotion_final                  (guards against single-annotator labels)

Output: data/processed/splits/text_hc/{train,val,test}.csv
"""

import csv
from collections import Counter
from pathlib import Path

SRC_DIR  = Path("data/processed/splits/text_eligible")
DST_DIR  = Path("data/processed/splits/text_hc")
DST_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "val", "test"]

VALID_LABELS = {
    "Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"
}

MIN_WORDS = 5


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
    transcript = (row.get("transcript") or "").strip()
    final      = (row.get("emotion_final") or "").strip()
    clarity    = (row.get("audio_clarity") or "").strip()

    # R1 – transcript length
    if len(transcript.split()) < MIN_WORDS:
        return False, "short_transcript"

    # R2 – valid label
    if final not in VALID_LABELS:
        return False, "invalid_label"

    # R3 – audio clarity
    if clarity != "Clear":
        return False, "noisy_audio"

    # R4 – cross-modal consensus (≥2 annotators agree with emotion_final)
    matching, total = count_annotator_agreement(row, final)
    if total < 2 or matching < 2:
        return False, f"low_consensus(match={matching},total={total})"

    return True, "ok"


for split in SPLITS:
    src = SRC_DIR / f"{split}.csv"
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
    print(f"\n=== {split}: {len(all_rows)} → {len(kept)} kept "
          f"({len(all_rows)-len(kept)} dropped) ===")
    print(f"  Skip reasons: {dict(skip_counts)}")
    print(f"  Label dist:   {dict(sorted(label_dist.items()))}")

print(f"\nDone. HC splits written to: {DST_DIR.resolve()}")
