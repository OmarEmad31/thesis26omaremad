import csv
import shutil
import os
from pathlib import Path

def prepare_mini_dataset():
    # 1. Paths
    project_root = Path.cwd()
    splits_dir = project_root / "data" / "processed" / "splits" / "text_hc"
    local_data_root = project_root / "dataset" / "Final Modalink Dataset MERGED"
    
    # OUTPUT FOLDER
    output_root = project_root / "Thesis_Mini_Upload"
    output_dataset_dir = output_root / "dataset" / "Final Modalink Dataset MERGED"
    output_splits_dir = output_root / "data" / "processed" / "splits" / "text_hc"
    
    print(f"--- 🚀 Starting Mini-Dataset Preparation (Path-Fix Version) ---")
    
    # 2. Collect all required paths from CSVs
    required_rel_paths = set()
    csv_files = ["train.csv", "val.csv", "test.csv"]
    
    for csv_file in csv_files:
        csv_path = splits_dir / csv_file
        if not csv_path.exists():
            print(f"❌ Error: Could not find {csv_path}")
            continue
            
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Combine video_id_short + audio_relpath
                # Example: "videoplayback (106)" + "audios/SPEAKER_00/segment_0001.wav"
                v_id = row.get('video_id_short', '').strip()
                a_rel = row.get('audio_relpath', '').strip()
                
                if v_id and a_rel:
                    # Construct the path that exists on disk
                    full_rel = os.path.join(v_id, a_rel)
                    required_rel_paths.add(full_rel)
            
    print(f"Found {len(required_rel_paths)} unique audio files needed.")
    
    # 3. Create structure and copy files
    if output_root.exists():
        print(f"Cleaning existing output folder...")
        shutil.rmtree(output_root)
        
    os.makedirs(output_dataset_dir, exist_ok=True)
    os.makedirs(output_splits_dir, exist_ok=True)
    
    # Copy the CSV splits themselves
    for csv_file in csv_files:
        shutil.copy2(splits_dir / csv_file, output_splits_dir / csv_file)
    
    # Copy the audio files
    print(f"Copying files (this might take a minute)...")
    copied_count = 0
    missing_count = 0
    
    for rel_path in required_rel_paths:
        src_path = local_data_root / rel_path
        dest_path = output_dataset_dir / rel_path
        
        if src_path.exists():
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dest_path)
            copied_count += 1
        else:
            # Fallback: Sometimes the folder name on disk doesn't have the space or something
            # But let's check first
            missing_count += 1
            
    print(f"\n--- ✅ Success! ---")
    print(f"Copied: {copied_count} audio files")
    if missing_count > 0:
        print(f"Missing: {missing_count} files")
        
    final_size = get_dir_size(output_root)
    print(f"\nYour mini dataset is ready at: {output_root}")
    print(f"Total size: {final_size / (1024*1024):.2f} MB")
    print(f"\n--- 📤 NEXT STEPS ---")
    print(f"1. Delete the current 'Thesis Project' folder on your Google Drive.")
    print(f"2. Rename the folder '{output_root.name}' to 'Thesis Project'.")
    print(f"3. Upload that ONE folder to your Google Drive root.")

def get_dir_size(path='.'):
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            total += os.path.getsize(fp)
    return total

if __name__ == "__main__":
    prepare_mini_dataset()
