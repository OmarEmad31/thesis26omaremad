"""
Tri-Expert Feature Extractor for Egyptian Arabic SER.
Sequential extraction to save RAM.

Expert 1: jonatasgrosman/wav2vec2-large-xlsr-53-arabic (Dialect knowledge)
Expert 2: superb/wav2vec2-large-superb-er (Acoustic Emotion knowledge)
Expert 3: Handcrafted Acoustic (Pitch, Energy, MFCC)

Run: python -m src.audio_baseline.extract_tri_expert
"""

import os, sys, gc, torch, librosa, numpy as np, pandas as pd
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
MAX_SAMPLES     = 6 * 16000
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

def get_transformer_embeddings(model_id, dataset_df, audio_map):
    print(f"🚀 Extracting from {model_id}...")
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
                else: audio = np.pad(audio, (0, MAX_SAMPLES - len(audio)))
                
                inputs = fe(audio, sampling_rate=16000, return_tensors="pt", padding=True).to(DEVICE)
                out = model(**inputs)
                # Mean pool the last hidden state
                pooled = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                embeddings.append(pooled)
                valid_indices.append(idx)
            except Exception as e:
                continue
                
    # Cleanup memory
    del model, fe
    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    
    return np.stack(embeddings), valid_indices

def main():
    print(f"\n{'='*60}")
    print("🔥  TRI-EXPERT FEATURE EXTRACTION  (Cap: 58-60%)")
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

    # 2. Extract Expert 1: Arabic XLSR
    emb_arabic, idx1 = get_transformer_embeddings(ARABIC_BACKBONE, df, audio_map)
    
    # 3. Extract Expert 2: SER Superb
    # (Note: Using the same indices to ensure alignment)
    df_v1 = df.iloc[idx1].reset_index(drop=True)
    emb_ser, idx2 = get_transformer_embeddings(SER_BACKBONE, df_v1, audio_map)
    
    # Final alignment: Ensure all experts have the exact same rows
    final_df = df_v1.iloc[idx2].reset_index(drop=True)
    emb_arabic_final = emb_arabic[idx2]
    emb_ser_final    = emb_ser
    
    # 4. Extract Expert 3: Handcrafted Physics
    print("🚀 Extracting Expert 3: Acoustic Physics (MFCC, Pitch)...")
    emb_physics = []
    for idx, row in tqdm(final_df.iterrows(), total=len(final_df)):
        basename = Path(str(row["audio_relpath"]).replace("\\", "/")).name
        path = audio_map.get(basename)
        audio, _ = librosa.load(path, sr=16000, mono=True)
        if len(audio) > MAX_SAMPLES: audio = audio[:MAX_SAMPLES]
        emb_physics.append(extract_physics(audio))
    emb_physics = np.stack(emb_physics)

    # 5. Combine & Save
    # We store experts individually in the PKL so the trainer can weight them
    out_path = config.SER_FEATURES_DIR / "tri_expert_dataset.pkl"
    final_data = {
        "metadata": final_df,
        "feat_arabic": emb_arabic_final,
        "feat_ser": emb_ser_final,
        "feat_physics": emb_physics
    }
    
    pd.to_pickle(final_data, out_path)
    print(f"\n✅  COMPLETED! Tri-Expert Features Saved.")
    print(f"    Path: {out_path}")
    print(f"    Shapes: Arabic={emb_arabic_final.shape} | SER={emb_ser_final.shape} | Physics={emb_physics.shape}")
    print(f"    Total combined dimensionality: {1024 + 1024 + emb_physics.shape[1]}")

if __name__ == "__main__":
    main()
