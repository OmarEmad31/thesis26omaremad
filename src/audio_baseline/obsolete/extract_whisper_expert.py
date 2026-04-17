import pandas as pd
import numpy as np
import torch
import librosa
from tqdm import tqdm
from transformers import AutoFeatureExtractor, WhisperModel
from pathlib import Path

import sys
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.audio_baseline import config

def main():
    print(f"🚀 Initializing Arabic-Native Whisper Offline Feature Extraction...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading '{config.MODEL_NAME}' natively to {device}...")
    extractor = AutoFeatureExtractor.from_pretrained(config.MODEL_NAME)
    
    # We heavily isolate the encoder to ONLY extract massive acoustic intelligence.
    model = WhisperModel.from_pretrained(config.MODEL_NAME).to(device)
    model.eval()
    
    train_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv")
    val_df = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv")
    test_df = pd.read_csv(config.SPLIT_CSV_DIR / "test.csv")
    
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    print("Mapping audio files securely natively on disk...")
    audio_map = {}
    search_dir = Path("/content") if Path("/content").exists() else Path(config.DATA_ROOT).parent
    for p in search_dir.rglob("*.wav"):
        audio_map[p.name] = p
        
    print(f"Locked onto {len(audio_map)} physical audio tracks.")
    
    features_list = []
    valid_indices = []
    
    with torch.no_grad():
        for idx, row in tqdm(all_df.iterrows(), total=len(all_df)):
            rel = str(row["audio_relpath"]).replace("\\", "/").lstrip("/")
            basename = Path(rel).name
            
            if basename not in audio_map:
                continue
                
            audio_path = audio_map[basename]
            
            try:
                # Whisper inherently mandates 16kHz
                audio, _ = librosa.load(audio_path, sr=16000, mono=True)
                audio = audio[:config.MAX_AUDIO_SAMPLES].astype(np.float32)
                
                # Push audio into the acoustic feature array (Length: 30 Seconds padded)
                inputs = extractor(audio, sampling_rate=16000, return_tensors="pt")
                input_features = inputs.input_features.to(device)
                
                # Execute the Whisper Acoustic Encoder specifically.
                encoder_outputs = model.encoder(input_features)
                last_hidden_state = encoder_outputs.last_hidden_state
                
                # MATHEMATICAL FIX: Padding Dilution Isolation
                # Whisper outputs exactly 50 sequence frames per second.
                # We calculate the precise fraction of seconds in this specific clip and slice the array to entirely avoid 0-padded features.
                duration_seconds = len(audio) / 16000.0
                active_frames = min(1500, int(duration_seconds * 50))
                
                if active_frames < 1:
                    active_frames = 1  # Failsafe for mathematically glitched silent files
                    
                # Extract only the physical vocal array: [batch=1, active_frames, dim=768]
                active_state = last_hidden_state[0, :active_frames, :]
                
                # Crush sequential dimension -> [768] Concentrated Dense Vector
                pooled = active_state.mean(dim=0).squeeze().cpu().numpy()
                
                features_list.append(pooled)
                valid_indices.append(idx)
                
            except Exception as e:
                print(f"[Fatal] {audio_path}: {e}")
                
    out_df = all_df.iloc[valid_indices].copy()
    out_df["whisper_features"] = features_list
    
    save_path = config.OFFLINE_FEATURES_DIR / "whisper_dataset.pkl"
    out_df.to_pickle(save_path)
    
    print(f"\n✅ Arabic Whisper Extractor Complete! Pure Acoustic matrices saved to {save_path}")
    print(f"Final Count: {out_df.shape[0]} / {len(all_df)} arrays encoded successfully.")

if __name__ == "__main__":
    main()
