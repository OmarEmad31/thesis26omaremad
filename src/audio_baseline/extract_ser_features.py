"""
Extract emotion-discriminative embeddings from superb/wav2vec2-large-superb-er.
This model was explicitly fine-tuned on IEMOCAP for Speech Emotion Recognition.
Its internal representations are already emotion-clustered — no training needed.

Run: python -m src.audio_baseline.extract_ser_features
"""

import pandas as pd
import numpy as np
import torch
import librosa
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel
from pathlib import Path

import sys
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.audio_baseline import config

SER_MODEL    = "superb/wav2vec2-large-superb-er"
MAX_SAMPLES  = 4 * 16000   # 4 seconds — sufficient to capture a full emotional utterance

def main():
    print(f"🚀 Extracting SER-expert embeddings from: {SER_MODEL}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device.upper()}")

    print(f"   Loading model...")
    # AutoModel gives us the encoder WITHOUT the classification head,
    # preserving the fine-tuned internal representations.
    extractor = AutoFeatureExtractor.from_pretrained(SER_MODEL)
    model     = AutoModel.from_pretrained(SER_MODEL).to(device)
    model.eval()

    # Load all splits (including test)
    train_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv"); train_df["split"] = "train"
    val_df   = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv");   val_df["split"]   = "val"
    test_df  = pd.read_csv(config.SPLIT_CSV_DIR / "test.csv");  test_df["split"]  = "test"
    all_df   = pd.concat([train_df, val_df, test_df], ignore_index=True)
    print(f"   Total rows: {len(all_df)}")

    # Build audio map (bulletproof file finder)
    print("   Mapping .wav files on disk...")
    audio_map = {}
    search_dir = Path("/content") if Path("/content").exists() else Path(config.DATA_ROOT).parent
    for p in search_dir.rglob("*.wav"):
        audio_map[p.name] = p
    print(f"   Locked onto {len(audio_map)} audio tracks.")

    features_list, valid_indices = [], []

    with torch.no_grad():
        for idx, row in tqdm(all_df.iterrows(), total=len(all_df)):
            basename   = Path(str(row["audio_relpath"]).replace("\\", "/")).name
            audio_path = audio_map.get(basename)
            if audio_path is None:
                continue

            try:
                audio, _ = librosa.load(audio_path, sr=16000, mono=True)
            except Exception as e:
                print(f"[Warn] {basename}: {e}")
                continue

            # Truncate / pad to 4 seconds
            if len(audio) > MAX_SAMPLES:
                audio = audio[:MAX_SAMPLES]
            else:
                audio = np.pad(audio, (0, MAX_SAMPLES - len(audio)))

            inputs = extractor(audio.astype(np.float32), sampling_rate=16000,
                               return_tensors="pt", padding=False,
                               max_length=MAX_SAMPLES, truncation=True)
            input_values = inputs.input_values.to(device)

            outputs      = model(input_values=input_values)
            # last_hidden_state: [1, T, 1024]  →  mean pool  →  [1024]
            pooled = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

            features_list.append(pooled)
            valid_indices.append(idx)

    out_df              = all_df.iloc[valid_indices].copy()
    out_df["ser_features"] = features_list

    save_path = config.SER_FEATURES_DIR / "ser_dataset.pkl"
    out_df.to_pickle(save_path)

    print(f"\n✅  SER Expert Extractor Complete!")
    print(f"   Saved → {save_path}")
    print(f"   Success: {len(out_df)} / {len(all_df)} files")

if __name__ == "__main__":
    main()
