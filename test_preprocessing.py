import os
import cv2
import numpy as np
from src.preprocessing import preprocess_image, IMAGE_SIZE

# --- CONFIG ---
RAW_DIR = "data/raw/test_user"  
PROCESSED_DIR = "data/processed/test_user"

os.makedirs(PROCESSED_DIR, exist_ok=True)

def process_folder(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR):
    """Process all images in a folder and save the processed faces."""
    files = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not files:
        print(f"No images found in {raw_dir}")
        return

    for file_name in files:
        img_path = os.path.join(raw_dir, file_name)
        image = cv2.imread(img_path)

        if image is None:
            print(f"⚠️ Could not read {file_name}, skipping.")
            continue

        processed = preprocess_image(image)
        if processed is None:
            print(f"⚠️ No face detected in {file_name}, skipping.")
            continue

        # Convert float image back to uint8 for saving
        save_img = (processed * 255).astype(np.uint8)
        save_path = os.path.join(processed_dir, file_name)
        cv2.imwrite(save_path, save_img)
        print(f"✅ Processed and saved {file_name}")

    print("✅ All images processed.")


if __name__ == "__main__":
    process_folder()