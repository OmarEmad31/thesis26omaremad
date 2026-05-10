import os
import pandas as pd
from pathlib import Path

def find_target_split():
    root = Path(r"d:\Thesis Project")
    print(f"Searching for Test=44 and Train=511 in {root}...")
    
    for p in root.rglob("test.csv"):
        try:
            te_df = pd.read_csv(p)
            if len(te_df) == 44:
                parent = p.parent
                tr_path = parent / "train.csv"
                va_path = parent / "val.csv"
                
                if tr_path.exists() and va_path.exists():
                    tr_df = pd.read_csv(tr_path)
                    va_df = pd.read_csv(va_path)
                    
                    if len(tr_df) == 511 and len(va_df) == 64:
                        print(f"\n[MATCH FOUND!]")
                        print(f"Path: {parent}")
                        print(f"Train: {len(tr_df)}")
                        print(f"Val:   {len(va_df)}")
                        print(f"Test:  {len(te_df)}")
                        return parent
        except:
            continue
    
    print("\nNo exact match found in project directory.")
    return None

if __name__ == "__main__":
    find_target_split()
