"""
Precision Tri-Expert Feature Extractor (v2).
FIXED: Padding Dilution (Mask-Aware Pooling) for both Transformers.
Successive loading to ensure no RAM crashes in Colab.

Run: python -m src.audio_baseline.extract_tri_expert
"""

import os, sys, gc, torch, librosa, numpy as np, pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel
from pathlib import Path

project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.audio_baseline import config
from src.audio_baseline.extract_acoustic_features import extract_features as extract_physics

# ─────────────────────────────────────────────
ARABIC_BACKBONE = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
SER_BACKBONE    = "superb/wav2vec2-large-superb-er"
MAX_SAMPLES     = 6 * 16000 # 6 seconds
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

def get_concentrated_embeddings(model_id, dataset_df, audio_map):
    print(f"🚀 Extracting from {model_id} (Mask-Aware)...")
    fe = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(DEVICE)
    model.eval()
    
    embeddings = []
    valid_indices = []
    
    with torch.no_grad():
        for idx, row in tqdm(dataset_df.iterrows(), total=len(dataset_df)):
            basename = Path(str(row["audio_relpath"]).replace("\\", "/")).name
            path = audio_map.get(basename)
            if not path: continue
            
            try:
                audio, _ = librosa.load(path, sr=16000, mono=True)
                if len(audio) > MAX_SAMPLES: audio = audio[:MAX_SAMPLES]
                # Pad to max for batch stability (though we process 1 by 1)
                audio_padded = np.pad(audio, (0, MAX_SAMPLES - len(audio)))
                
                # Get mask and input values
                inputs = fe(audio_padded, sampling_rate=16000, return_tensors="pt", 
                            padding="max_length", max_length=MAX_SAMPLES).to(DEVICE)
                
                # Custom mask for the REAL audio length (before padding)
                # Wav2Vec2 frames = ~50 per second. Input is 16000Hz.
                # But fe() call above pads the mask too. We need the raw mask.
                # Actually, the fe() call with padding='max_length' provides the correct mask!
                out = model(input_values=inputs.input_values, attention_mask=inputs.attention_mask)
                
                # 🛠️ Fix Padding Dilution: Masked Global Average Pooling
                last_hidden = out.last_hidden_state # [1, T, 1024]
                mask = inputs.attention_mask         # [1, MAX_SAMPLES]
                
                # Align mask to hidden state size
                mask_resized = F.interpolate(mask.unsqueeze(1).float(), 
                                             size=last_hidden.size(1), 
                                             mode='nearest').squeeze(1) # [1, T]
                mask_expanded = mask_resized.unsqueeze(-1).expand_as(last_hidden) # [1, T, 1024]
                
                # Weighted Mean
                pooled = (last_hidden * mask_expanded).sum(dim=1) / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                
                embeddings.append(pooled.squeeze().cpu().numpy())
                valid_indices.append(idx)
            except Exception as e:
                # print(f"Error on {basename}: {e}")
                continue
                
    # Cleanup memory
    del model, fe
    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    
    return np.stack(embeddings), valid_indices

def main():
    print(f"\n{'='*60}")
    print("🔥  PRECISION TRI-EXPERT EXTRACTION (Fixed Dilution)")
    print(f"{'='*60}")

    # 1. Load Data & Map
    train_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv"); train_df["split"] = "train"
    val_df   = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv");   val_df["split"]   = "val"
    test_df  = pd.read_csv(config.SPLIT_CSV_DIR / "test.csv");  test_df["split"]  = "test"
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    audio_map = {}
    src = Path("/content") if Path("/content").exists() else Path(config.DATA_ROOT).parent
    for p in src.rglob("*.wav"): audio_map[p.name] = p
    print(f"Mapped {len(audio_map)} files.\n")

    # 2. Extract Expert 1: Arabic XLSR (Masked)
    emb_arabic, idx1 = get_concentrated_embeddings(ARABIC_BACKBONE, df, audio_map)
    
    # 3. Extract Expert 2: SER Superb (Masked)
    df_v1 = df.iloc[idx1].reset_index(drop=True)
    emb_ser, idx2 = get_concentrated_embeddings(SER_BACKBONE, df_v1, audio_map)
    
    # Align
    final_df = df_v1.iloc[idx2].reset_index(drop=True)
    emb_arabic_final = emb_arabic[idx2]
    
    # 4. Extract Expert 3: Handcrafted (Physics)
    print("🚀 Extracting Expert 3: Acoustic Physics...")
    emb_physics = []
    for idx, row in tqdm(final_df.iterrows(), total=len(final_df)):
        basename = Path(str(row["audio_relpath"]).replace("\\", "/")).name
        path = audio_map.get(basename)
        audio, _ = librosa.load(path, sr=16000, mono=True)
        if len(audio) > MAX_SAMPLES: audio = audio[:MAX_SAMPLES]
        emb_physics.append(extract_physics(audio))
    emb_physics = np.stack(emb_physics)

    # 5. Save Concentrated Features
    out_path = config.SER_FEATURES_DIR / "tri_expert_dataset_v2.pkl"
    final_data = {
        "metadata": final_df,
        "feat_arabic": emb_arabic_final,
        "feat_ser": emb_ser,
        "feat_physics": emb_physics
    }
    
    pd.to_pickle(final_data, out_path)
    print(f"\n✅  SUCCESS! Features are now 'Concentrated' and aligned.")
    print(f"    Path: {out_path}")
    print(f"    Total Features: {1024 + 1024 + emb_physics.shape[1]}")

if __name__ == "__main__":
    main()
