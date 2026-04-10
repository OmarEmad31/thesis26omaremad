"""Load CSV splits and build label mappings."""

from __future__ import annotations

import csv
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Arabic normalization
# ---------------------------------------------------------------------------

# Tatweel (kashida elongation character)
_TATWEEL = re.compile(r"\u0640")

# Arabic diacritics / tashkeel (harakat + related marks)
_DIACRITICS = re.compile(
    r"[\u064B-\u065F"   # fathatan … sukun
    r"\u0610-\u061A"   # extended Arabic marks
    r"\u06D6-\u06DC"   # Quranic annotation signs
    r"\u06DF-\u06E4"   # more annotation signs
    r"\u06E7\u06E8"
    r"\u06EA-\u06ED]"  # maddah etc.
)

# Alef variants → bare Alef
_ALEF_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})

# Collapse any run of whitespace to a single space
_MULTI_SPACE = re.compile(r"\s+")

# Minimum character length to keep a transcript (after normalization)
_MIN_TEXT_LEN = 3


def normalize_arabic(text: str) -> str:
    """Apply minimal Arabic normalization before MARBERT tokenization."""
    text = _TATWEEL.sub("", text)          # 1. remove tatweel
    text = _DIACRITICS.sub("", text)       # 2. remove diacritics
    text = text.translate(_ALEF_MAP)       # 3. normalize Alef variants → ا
    text = text.replace("ى", "ي")          # 4. normalize Alef Maqsura → ي
    text = _MULTI_SPACE.sub(" ", text)     # 5. collapse internal whitespace
    return text.strip()


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_split_csv(path: Path, text_col: str, label_col: str) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    skipped = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = normalize_arabic(row.get(text_col) or "")
            y = (row.get(label_col) or "").strip()
            if not y:
                skipped += 1
                continue
            if len(t) < _MIN_TEXT_LEN:   # drop empty / extremely short transcripts
                skipped += 1
                continue
            texts.append(t)
            labels.append(y)
    if skipped:
        print(f"  [data] skipped {skipped} rows (no label / transcript too short) in {path.name}")
    return texts, labels


def build_label2id(train_labels: list[str]) -> dict[str, int]:
    unique = sorted(set(train_labels))
    return {lab: i for i, lab in enumerate(unique)}


def encode_labels(labels: list[str], label2id: dict[str, int]) -> list[int]:
    return [label2id[y] for y in labels]


def labels_in_order(label2id: dict[str, int]) -> list[str]:
    return [lab for lab, _ in sorted(label2id.items(), key=lambda x: x[1])]
