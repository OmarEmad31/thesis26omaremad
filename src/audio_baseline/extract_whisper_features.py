import pandas as pd
import numpy as np
import torch
import librosa
from tqdm import tqdm
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from pathlib import Path

# Fix python path dynamically if running locally
import sys
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.audio_baseline import config

def main():
    print("🚀 Initializing Offline Egyptian Whisper Extraction...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Whisper Model explicitly mapping to its Acoustic Encoder
    print(f"Loading '{config.MODEL_NAME}' to {device}...")
    extractor = AutoFeatureExtractor.from_pretrained(config.MODEL_NAME)
    
    # We load the classification model but only keep the base encoder
    base_model = AutoModelForAudioClassification.from_pretrained(config.MODEL_NAME).to(device)
    base_model.eval() # Freeze completely
    
    # Depending on model architecture, the encoder is usually nested. 
    # For WhisperForAudioClassification, it sits in `base_model.whisper.encoder` or `base_model.audio_model.encoder`.
    # Let's interact with it safely:
    if hasattr(base_model, "whisper"):
        encoder = base_model.whisper.encoder
    elif hasattr(base_model, "audio_model"):
        encoder = base_model.audio_model.encoder
    else:
        # Fallback to the overarching base_model if directly mapping
        encoder = base_model

    # 2. Combine train/val to extract everything in one massive pass
    train_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv")
    val_df = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv")
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    all_df = pd.concat([train_df, val_df], ignore_index=True)
    
    print(f"Processing {len(all_df)} total audio files...")
    
    # 3. Extract Loop
    features_list = []
    valid_indices = []
    
    with torch.no_grad():
        for idx, row in tqdm(all_df.iterrows(), total=len(all_df)):
            fldr = str(row["folder"]).strip()
            rel = str(row["audio_relpath"]).replace("\\", "/").lstrip("/")
            audio_path = config.DATA_ROOT / fldr / rel
            
            if not audio_path.exists():
                print(f"[Warning] Missing file: {audio_path}")
                continue
                
            try:
                audio, _ = librosa.load(audio_path, sr=config.SAMPLING_RATE, mono=True)
                audio = audio[:config.MAX_AUDIO_SAMPLES].astype(np.float32)
                
                # Extract Whisper 30s padding
                inputs = extractor(audio, sampling_rate=config.SAMPLING_RATE, return_tensors="pt").to(device)
                
                # Fetch acoustic representation (last_hidden_state)
                # Output dim format: [1, 1500, 768] (Frames vs Features)
                if "input_features" in inputs:
                    enc_out = encoder(inputs.input_features).last_hidden_state
                else:
                    enc_out = encoder(inputs.input_values).last_hidden_state
                    
                # We Mean-Pool the 1500 acoustic frames into one singular 768-dimension vector
                # that represents the entire Egyptian emotional cadence of the audio clip.
                pooled = enc_out.mean(dim=1).squeeze().cpu().numpy()
                
                features_list.append(pooled)
                valid_indices.append(idx)
                
            except Exception as e:
                print(f"[Error] Failed to process {audio_path}: {e}")
                
    # 4. Filter the dataframe and attach features
    out_df = all_df.iloc[valid_indices].copy()
    out_df["whisper_features"] = features_list
    
    # 5. Save massively compressed database globally
    save_path = config.OFFLINE_FEATURES_DIR / "whisper_dataset.pkl"
    out_df.to_pickle(save_path)
    
    print(f"\n✅ 100% Complete! Offline Dataset saved securely to {save_path}")
    print(f"Final Count: {out_df.shape[0]} / {len(all_df)} files encoded successfully.")

if __name__ == "__main__":
    main()
