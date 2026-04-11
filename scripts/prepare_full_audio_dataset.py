import shutil
import os
from pathlib import Path

def prepare_full_audio_dataset():
    # 1. Paths
    project_root = Path.cwd()
    data_dir = project_root / "data"
    local_data_root = project_root / "dataset" / "Final Modalink Dataset MERGED"
    
    # OUTPUT FOLDER
    output_root = project_root / "Thesis_Full_Audio_Upload"
    output_dataset_dir = output_root / "dataset" / "Final Modalink Dataset MERGED"
    output_data_dir = output_root / "data"
    
    print(f"--- 🚀 Starting Full-Audio Dataset Preparation ---")
    
    # 2. Create structure
    if output_root.exists():
        print(f"Cleaning existing output folder...")
        shutil.rmtree(output_root)
        
    os.makedirs(output_dataset_dir, exist_ok=True)
    
    # 3. Copy ALL Data Splits (The whole 'data' folder)
    if data_dir.exists():
        print(f"Copying all data splits and manifests...")
        shutil.copytree(data_dir, output_data_dir)
    
    # 4. Copy EVERY .wav file in the dataset
    print(f"Scanning for all audio files (this will take a minute)...")
    copied_count = 0
    
    # Walk through the dataset folder
    for root, dirs, files in os.walk(local_data_root):
        for file in files:
            if file.endswith(".wav"):
                src_path = Path(root) / file
                
                # Maintain relative structure: dataset/Final Modalink.../videoplayback (X)/audios/Y.wav
                rel_parts = src_path.relative_to(local_data_root)
                dest_path = output_dataset_dir / rel_parts
                
                # Create destination folder
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dest_path)
                copied_count += 1
                
                if copied_count % 500 == 0:
                    print(f"Copied {copied_count} files...")
            
    print(f"\n--- ✅ Success! ---")
    print(f"Copied: {copied_count} audio files (100% of the dataset)")
        
    final_size = get_dir_size(output_root)
    print(f"\nYour complete audio dataset is ready at: {output_root}")
    print(f"Total size: {final_size / (1024*1024):.2f} MB")
    print(f"\n--- 📤 NEXT STEPS ---")
    print(f"1. Delete any old 'Thesis Project' folders on your Google Drive.")
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
    prepare_full_audio_dataset()
