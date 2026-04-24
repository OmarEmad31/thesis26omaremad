import pandas as pd
import sys
from pathlib import Path

csv_path = Path("D:/Thesis Project/data/processed/splits/audio_eligible/train.csv")
print(f"Checking CSV: {csv_path}")

try:
    # Attempt 1: Default (UTF-8)
    print("Attempt 1: Default encoding...")
    df = pd.read_csv(csv_path)
    print(f"Success! Rows: {len(df)}")
except Exception as e:
    print(f"Attempt 1 Failed: {e}")
    try:
        # Attempt 2: latin-1 (common fallback)
        print("Attempt 2: latin-1...")
        df = pd.read_csv(csv_path, encoding='latin-1')
        print(f"Success with latin-1! Rows: {len(df)}")
    except Exception as e2:
        print(f"Attempt 2 Failed: {e2}")
        try:
            # Attempt 3: utf-16
            print("Attempt 3: utf-16...")
            df = pd.read_csv(csv_path, encoding='utf-16')
            print(f"Success with utf-16! Rows: {len(df)}")
        except Exception as e3:
            print(f"Attempt 3 Failed: {e3}")
