import os, sys
from pathlib import Path

# Step 1: Paths and Imports
print("Step 1: Imports...")
try:
    import torch
    import pandas as pd
    from transformers import WavLMModel
    print("Imports OK")
except Exception as e:
    print(f"Import Failed: {e}")
    sys.exit(1)

# Step 2: Config
print("Step 2: Config...")
try:
    project_root = str(Path(__file__).parent.parent.absolute())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.audio_baseline import config
    print(f"Config OK. Split Dir: {config.SPLIT_CSV_DIR}")
except Exception as e:
    print(f"Config Failed: {e}")
    sys.exit(1)

# Step 3: Loading CSV
print("Step 3: CSV Load...")
try:
    tr_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv")
    print(f"CSV OK. Labels: {tr_df['emotion_final'].unique()}")
except Exception as e:
    print(f"CSV Failed: {e}")
    sys.exit(1)

# Step 4: Model Load
print("Step 4: Model Init (CUDA)...")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model_name = "microsoft/wavlm-base-plus"
    os.environ["HF_HOME"] = "D:/HuggingFaceCache"
    # Load model
    model = WavLMModel.from_pretrained(model_name).to(device)
    print("Model Loaded OK")
except Exception as e:
    print(f"Model Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("ALL DIAGNOSTICS PASSED")
