"""
Egyptian Arabic Text Booster (v95)
==================================
Identifies 'Fear', 'Surprise', and 'Happiness' classes.
Applies Random Insertion, Swap, and Deletion (EDA) to balance the training set.
"""

import pandas as pd
import random
from pathlib import Path

def augment_text(text, n=2):
    """
    Simple EDA: Random Swap and Random Deletion.
    Effective for boosting small datasets without complex translation APIs.
    """
    words = text.split()
    if len(words) < 3: return [text]
    
    augmented = []
    
    # 1. Random Swap
    for _ in range(n):
        new_words = words.copy()
        idx1, idx2 = random.sample(range(len(new_words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        augmented.append(" ".join(new_words))
        
    # 2. Random Deletion
    for _ in range(n):
        if len(words) > 2:
            new_words = words.copy()
            idx = random.randint(0, len(new_words)-1)
            del new_words[idx]
            augmented.append(" ".join(new_words))
            
    return list(set(augmented))

def main():
    root = Path("/content/drive/MyDrive/Thesis Project")
    data_path = root / "data/processed/splits/final_sanitized/train.csv"
    
    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        return
        
    df = pd.read_csv(data_path)
    original_size = len(df)
    
    # Target rare classes
    rare_classes = ['Fear', 'Surprise', 'Happiness']
    new_rows = []
    
    for _, row in df.iterrows():
        if row['emotion_final'] in rare_classes:
            augs = augment_text(row['transcript'], n=3)
            for aug in augs:
                new_row = row.copy()
                new_row['transcript'] = aug
                new_rows.append(new_row)
                
    # Combine original and augmented
    augmented_df = pd.concat([df, pd.DataFrame(new_rows)]).reset_index(drop=True)
    
    # Save
    out_path = root / "data/processed/splits/final_sanitized/train_augmented.csv"
    augmented_df.to_csv(out_path, index=False)
    
    print(f"✅ Data Boosting Complete!")
    print(f"📊 Original: {original_size} -> Augmented: {len(augmented_df)}")
    print(f"📁 Saved to: {out_path}")

if __name__ == "__main__":
    main()
