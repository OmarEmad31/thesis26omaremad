import pandas as pd
from pathlib import Path

def scan_splits():
    root = Path(r"d:\Thesis Project")
    splits_dir = root / "data" / "processed" / "splits"
    
    print(f"Scanning splits in: {splits_dir}\n")
    results = []
    
    for p in splits_dir.glob("**/train.csv"):
        parent = p.parent
        try:
            tr_df = pd.read_csv(p)
            va_path = parent / "val.csv"
            te_path = parent / "test.csv"
            
            va_count = len(pd.read_csv(va_path)) if va_path.exists() else 0
            te_count = len(pd.read_csv(te_path)) if te_path.exists() else 0
            
            tr_count = len(tr_df)
            cols = tr_df.columns.tolist()
            
            # Identify ID and Label columns
            sample_id_col = "sample_id" if "sample_id" in cols else (cols[0] if len(cols)>0 else "N/A")
            label_col = "emotion_final" if "emotion_final" in cols else ("label" if "label" in cols else "N/A")
            
            results.append({
                "folder": str(parent.relative_to(splits_dir)),
                "train": tr_count,
                "val": va_count,
                "test": te_count,
                "sample_id_col": sample_id_col,
                "label_col": label_col
            })
        except Exception as e:
            print(f"Error reading {parent}: {e}")

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # Identify match
    match = df[(df['train'] == 511) & (df['val'] == 64) & (df['test'] == 44)]
    if not match.empty:
        print("\n[MATCH FOUND!]")
        print(match.to_string(index=False))
    else:
        # Check for close matches
        print("\n[NO EXACT MATCH (511, 64, 44)]")
        close = df[(df['train'] > 500) & (df['test'] < 60)]
        if not close.empty:
            print("Close matches (Train > 500, Test < 60):")
            print(close.to_string(index=False))

if __name__ == "__main__":
    scan_splits()
