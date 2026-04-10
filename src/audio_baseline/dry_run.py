"""Sanity check for the Audio Baseline.
Run: python -m src.audio_baseline.dry_run
"""

import torch
from transformers import Wav2Vec2FeatureExtractor
from src.audio_baseline import config
from src.audio_baseline.model import Emotion2VecBaseline
from src.audio_baseline.data import AudioEmotionDataset, collate_audio_fn
import pandas as pd
from torch.utils.data import DataLoader

def dry_run():
    print("--- 🧪 Starting Audio Baseline Dry Run ---")
    
    # 1. Setup
    device = torch.device("cpu") # Dry run on CPU for safety
    label2id = {"Neutral": 0, "Anger": 1, "Happiness": 2, "Sadness": 3, "Fear": 4, "Disgust": 5, "Surprise": 6}
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config.MODEL_NAME)
    
    # 2. Check individual sample loading
    print(f"Checking data mapping from {config.SPLIT_CSV_DIR/'test.csv'}...")
    test_df = pd.read_csv(config.SPLIT_CSV_DIR / "test.csv")
    
    ds = AudioEmotionDataset(
        csv_path=None,
        data_root=config.DATA_ROOT,
        feature_extractor=feature_extractor,
        label2id=label2id,
        max_samples=config.MAX_AUDIO_SAMPLES
    )
    ds.df = test_df.head(2) # Just take 2 samples
    
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_audio_fn)
    batch = next(iter(loader))
    
    print(f"  Batch Loaded! Input shape: {batch['input_values'].shape}")
    print(f"  Labels: {batch['labels']}")
    
    # 3. Model Inference
    print(f"Loading {config.MODEL_NAME} (Base version)...")
    model = Emotion2VecBaseline(config.MODEL_NAME, num_labels=len(label2id))
    model.eval()
    
    with torch.no_grad():
        logits = model(batch["input_values"].to(device), attention_mask=batch["attention_mask"].to(device))
        print(f"  Inference Success! Logits shape: {logits.shape}")
        
    print("\n--- ✅ Dry Run Complete: Audio Pipeline is Valid! ---")

if __name__ == "__main__":
    dry_run()
