import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def create_final_sanitized_splits():
    root = Path("/content/drive/MyDrive/Thesis Project")
    clean_p = root / "data/processed/splits/trackA_cleaned"
    
    # Load the base sanitized pool
    tr_pool = pd.read_csv(clean_p / "trackA_train_clean.csv")
    va_pool = pd.read_csv(clean_p / "trackA_val_clean.csv")
    
    # The 511 samples are always training
    train_final = tr_pool 
    
    # Split the 108 eval samples EXACTLY as we did for Audio (Seed 42, 40% test)
    val_final, test_final = train_test_split(
        va_pool, 
        test_size=0.4, 
        random_state=42, 
        stratify=va_pool['emotion_final']
    )
    
    # Save the Final Locked-Down CSVs
    final_dir = root / "data/processed/splits/final_sanitized"
    final_dir.mkdir(parents=True, exist_ok=True)
    
    train_final.to_csv(final_dir / "train.csv", index=False)
    val_final.to_csv(final_dir / "val.csv", index=False)
    test_final.to_csv(final_dir / "test.csv", index=False)
    
    print(f"✅ Final Splits Created in: {final_dir}")
    print(f"📊 Train: {len(train_final)} | Val: {len(val_final)} | Test: {len(test_final)}")

if __name__ == "__main__": create_final_sanitized_splits()
