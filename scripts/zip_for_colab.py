import shutil
import os
from pathlib import Path

def zip_for_colab():
    # 1. Paths
    root = Path("D:/Thesis Project")
    data_processed = root / "data" / "processed"
    dataset_folder = Path("D:/Thesis Project/dataset")
    output_zip = Path("D:/Thesis_Project_Colab_Ready.zip")

    print(f"Starting compression for Colab migration...")
    
    # We only need the splits and the audio files
    # Zip the dataset folder (the wav files)
    print(f"Zipping audio files (this may take 2-5 mins)...")
    shutil.make_archive(str(output_zip.with_suffix('')), 'zip', root_dir=root, base_dir='dataset')
    
    print(f"\nSUCCESS! Your file is ready at: {output_zip}")
    print(f"Size: {os.path.getsize(output_zip) / (1024*1024):.2f} MB")
    print("Next step: Upload this file to your Google Drive.")

if __name__ == "__main__":
    zip_for_colab()
