"""
Classical Acoustic Feature Extraction for Egyptian Arabic SER.
Language-agnostic: MFCCs, pitch, energy, spectral features.
No pretrained model required. Runs on CPU in seconds.

Run: python -m src.audio_baseline.extract_acoustic_features
"""

import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import sys
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.audio_baseline import config

SR          = 16000
MAX_SAMPLES = 6 * SR   # 6 seconds — slightly longer to capture full utterances
N_MFCC      = 40
N_MELS      = 128
HOP         = 256
N_FFT       = 1024


def extract_features(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    Extract a comprehensive, language-agnostic acoustic feature vector.
    All features are computed as (mean, std) over the time axis.
    Final dimensionality: ~280 features.
    """
    feats = []

    # 1. MFCC (40 coeffs) + delta + delta-delta → mean & std
    mfcc        = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, hop_length=HOP, n_fft=N_FFT)
    delta_mfcc  = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    for arr in [mfcc, delta_mfcc, delta2_mfcc]:
        feats.extend(arr.mean(axis=1))
        feats.extend(arr.std(axis=1))

    # 2. Mel-spectrogram (128 bins) → mean & std
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS, hop_length=HOP, n_fft=N_FFT)
    log_mel = librosa.power_to_db(mel)
    feats.extend(log_mel.mean(axis=1))
    feats.extend(log_mel.std(axis=1))

    # 3. Chroma → mean & std (12 pitch classes)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=HOP, n_fft=N_FFT)
    feats.extend(chroma.mean(axis=1))
    feats.extend(chroma.std(axis=1))

    # 4. Spectral contrast (7 bands) → mean & std
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, hop_length=HOP, n_fft=N_FFT)
    feats.extend(contrast.mean(axis=1))
    feats.extend(contrast.std(axis=1))

    # 5. Tonnetz (6 dims) → mean & std
    y_harm = librosa.effects.harmonic(audio)
    tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
    feats.extend(tonnetz.mean(axis=1))
    feats.extend(tonnetz.std(axis=1))

    # 6. ZCR → mean and std
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=HOP)
    feats.append(float(zcr.mean()))
    feats.append(float(zcr.std()))

    # 7. RMS energy → mean and std
    rms = librosa.feature.rms(y=audio, hop_length=HOP)
    feats.append(float(rms.mean()))
    feats.append(float(rms.std()))

    # 8. Spectral centroid, bandwidth, rolloff → mean and std
    for fn in [librosa.feature.spectral_centroid,
               librosa.feature.spectral_bandwidth,
               librosa.feature.spectral_rolloff]:
        arr = fn(y=audio, sr=sr, hop_length=HOP, n_fft=N_FFT)
        feats.append(float(arr.mean()))
        feats.append(float(arr.std()))

    # 9. Pitch (F0) via YIN → mean, std, range
    try:
        f0 = librosa.yin(audio, fmin=50, fmax=400, sr=sr, hop_length=HOP)
        f0_voiced = f0[f0 > 0]
        if len(f0_voiced) > 0:
            feats.extend([float(f0_voiced.mean()), float(f0_voiced.std()),
                          float(f0_voiced.max() - f0_voiced.min())])
        else:
            feats.extend([0.0, 0.0, 0.0])
    except Exception:
        feats.extend([0.0, 0.0, 0.0])

    return np.array(feats, dtype=np.float32)


def main():
    print("🚀  Classical Acoustic Feature Extraction")
    print(f"   Features: MFCC×3 + MelSpec + Chroma + Contrast + Tonnetz + ZCR + RMS + Centroid + Pitch")
    print(f"   Language-agnostic: works directly on Egyptian Arabic without pretrained models\n")

    train_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv"); train_df["split"] = "train"
    val_df   = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv");   val_df["split"]   = "val"
    test_df  = pd.read_csv(config.SPLIT_CSV_DIR / "test.csv");  test_df["split"]  = "test"
    all_df   = pd.concat([train_df, val_df, test_df], ignore_index=True)
    print(f"   Total rows: {len(all_df)}")

    audio_map = {}
    search_dir = Path("/content") if Path("/content").exists() else Path(config.DATA_ROOT).parent
    for p in search_dir.rglob("*.wav"):
        audio_map[p.name] = p
    print(f"   Mapped {len(audio_map)} audio tracks.\n")

    features_list, valid_indices = [], []

    for idx, row in tqdm(all_df.iterrows(), total=len(all_df)):
        basename   = Path(str(row["audio_relpath"]).replace("\\", "/")).name
        audio_path = audio_map.get(basename)
        if audio_path is None:
            continue

        try:
            audio, _ = librosa.load(audio_path, sr=SR, mono=True)
        except Exception:
            continue

        if len(audio) > MAX_SAMPLES:
            audio = audio[:MAX_SAMPLES]
        elif len(audio) < SR // 2:      # skip clips shorter than 0.5 seconds
            continue
        else:
            audio = np.pad(audio, (0, MAX_SAMPLES - len(audio)))

        feats = extract_features(audio)
        features_list.append(feats)
        valid_indices.append(idx)

    out_df = all_df.iloc[valid_indices].copy()
    out_df["acoustic_features"] = features_list

    save_path = config.SER_FEATURES_DIR / "acoustic_dataset.pkl"
    out_df.to_pickle(save_path)

    dim = len(features_list[0]) if features_list else 0
    print(f"\n✅  Done! Feature dimensionality: {dim}")
    print(f"   Saved → {save_path}")
    print(f"   Success: {len(out_df)} / {len(all_df)} files")


if __name__ == "__main__":
    main()
