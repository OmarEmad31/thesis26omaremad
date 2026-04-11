import zipfile
import os
from pathlib import Path

def zip_full_audio_dataset():
    # 1. Paths
    project_root = Path.cwd()
    data_dir = project_root / "data"
    local_data_root = project_root / "dataset" / "Final Modalink Dataset MERGED"
    
    # OUTPUT ON D: DRIVE
    output_folder = Path("D:/ThesisWork")
    if not output_folder.exists():
        try:
            os.makedirs(output_folder, exist_ok=True)
            print(f"Created folder: {output_folder}")
        except Exception as e:
            print(f"⚠️ Could not create folder on D: drive, falling back to C: Desktop. Error: {e}")
            output_folder = project_root
            
    output_zip = output_folder / "Thesis_Audio_Full.zip"
    
    print(f"--- 📦 Starting Super-Zip Preparation ---")
    print(f"Goal: 100% of Audio + 100% of Data Splits")
    
    if output_zip.exists():
        os.remove(output_zip)
        
    copied_count = 0
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 1. Add ALL Data Splits (The whole 'data' folder)
        if data_dir.exists():
            print(f"Adding all data splits and manifests...")
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = os.path.relpath(file_path, project_root)
                    zipf.write(file_path, arcname)
        
        # 2. Add EVERY .wav file in the dataset
        print(f"Scanning and zipping all audio files...")
        for root, dirs, files in os.walk(local_data_root):
            for file in files:
                if file.endswith(".wav"):
                    src_path = Path(root) / file
                    
                    # Store as: Thesis Project/dataset/Final Modalink.../videoplayback (X)/audios/Y.wav
                    # This ensures it matches our config.py paths
                    arcname = os.path.join("Thesis Project", os.path.relpath(src_path, project_root))
                    zipf.write(src_path, arcname)
                    copied_count += 1
                    
                    if copied_count % 500 == 0:
                        print(f"Zipped {copied_count} files...")

    print(f"\n--- ✅ Success! ---")
    print(f"Total: {copied_count} audio files zipped.")
    print(f"Zip File Location: {output_zip}")
    print(f"Final Size: {os.path.getsize(output_zip) / (1024*1024):.2f} MB")
    
    print(f"\n--- 📤 HOW TO UPLOAD ---")
    print(f"1. Upload 'Thesis_Audio_Full.zip' to the root of your Google Drive.")
    print(f"2. When the upload is finished, let me know. I will give you a 1-line command to 'exploding' the zip inside Colab!")

if __name__ == "__main__":
    zip_full_audio_dataset()
