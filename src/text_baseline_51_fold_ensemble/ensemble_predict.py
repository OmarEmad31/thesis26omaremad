import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from datasets import Dataset

from src.text_baseline import config
from src.text_baseline.model import MARBERTWithMultiSampleDropout
from src.text_baseline.data import load_split_csv, encode_labels, labels_in_order, build_label2id
from src.text_baseline.metrics_utils import evaluate_predictions, confusion_matrix_labels

def print_confusion_matrix(cm: np.ndarray, label_names: list[str], title: str) -> None:
    print(f"\n--- Confusion matrix ({title}) ---")
    header = "true\\pred".ljust(14) + "".join(f"{n[:10]:>12}" for n in label_names)
    print(header)
    for i, row_name in enumerate(label_names):
        line = f"{row_name[:12]:<14}" + "".join(f"{cm[i, j]:>12d}" for j in range(cm.shape[1]))
        print(line)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    test_texts, test_labels = load_split_csv(config.TEST_CSV, config.TEXT_COLUMN, config.LABEL_COLUMN)
    
    # Load label2id from any fold (they are the same)
    label2id_path = config.CHECKPOINT_DIR / "label2id.json"
    with open(label2id_path, "r", encoding="utf-8") as f:
        label2id = json.load(f)
    
    id2label = {v: k for k, v in label2id.items()}
    label_names = labels_in_order(label2id)
    num_labels = len(label2id)
    test_ids = encode_labels(test_labels, label2id)
    y_true = np.array(test_ids)

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=config.MAX_LENGTH)
    
    test_ds = Dataset.from_dict({"text": test_texts, "labels": test_ids})
    test_ds = test_ds.map(tokenize, batched=True, remove_columns=["text"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, collate_fn=collator)

    # 2. Collect Predictions from all Available Folds
    all_fold_logits = []
    
    # Automatically detect completed folds
    fold_dirs = sorted(list(config.CHECKPOINT_DIR.glob("fold_*")))
    completed_folds = []
    for d in fold_dirs:
        # Check for either .bin or .safetensors
        if (d / "best_model" / "pytorch_model.bin").exists() or (d / "best_model" / "model.safetensors").exists():
            completed_folds.append(d)
    
    if not completed_folds:
        print(f"Error: No completed folds found in {config.CHECKPOINT_DIR}")
        return

    print(f"Detected {len(completed_folds)} completed folds. Starting Ensemble prediction...")

    for fold_dir in completed_folds:
        best_model_path = fold_dir / "best_model"
        print(f"Loading Model from {fold_dir.name}...")
        
        model = MARBERTWithMultiSampleDropout(
            config.MODEL_NAME, 
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        )
        
        # Load weights (handle both formats)
        bin_path = best_model_path / "pytorch_model.bin"
        safe_path = best_model_path / "model.safetensors"
        
        if safe_path.exists():
            import safetensors.torch
            state_dict = safetensors.torch.load_file(safe_path, device="cpu")
        else:
            state_dict = torch.load(bin_path, map_location=device, weights_only=True)
            
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        fold_logits = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                fold_logits.append(outputs.logits.cpu().numpy())
        
        all_fold_logits.append(np.vstack(fold_logits))

    # 3. Ensemble (Soft-Voting / Averaging Probabilities)
    # Convert logits to probabilities and then average
    all_probs = [F.softmax(torch.from_numpy(logits), dim=-1).numpy() for logits in all_fold_logits]
    avg_probs = np.mean(all_probs, axis=0)
    final_preds = np.argmax(avg_probs, axis=-1)

    # 4. Final Evaluation
    metrics = evaluate_predictions(y_true, final_preds)
    print("\n" + "="*40)
    print(f"      FINAL {len(completed_folds)}-FOLD ENSEMBLE RESULTS      ")
    print("="*40)
    print(f"Accuracy:     {metrics['accuracy']:.4f}")
    print(f"Macro F1:     {metrics['f1_macro']:.4f}")
    print(f"Weighted F1: {metrics['f1_weighted']:.4f}")
    print("="*40)
    
    cm = confusion_matrix_labels(y_true, final_preds, num_labels)
    print_confusion_matrix(cm, label_names, "Ensemble")

if __name__ == "__main__":
    main()
