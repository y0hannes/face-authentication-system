import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_image(img_path, size=(64, 64)):
    """
    Load an image, apply contrast enhancement, resize, and normalize for CNN.
    """
    # 1. Read as Grayscale (Standard for basic FaceAuth)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found or unreadable: {img_path}")

    # 2. Feature Engineering: Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # 3. Resize to a consistent square dimension
    img = cv2.resize(img, size)

    # 4. Normalization: Scale pixels to [0, 1] range
    img = img.astype('float32') / 255.0

    # 5. Add Channel Dimension: CNNs expect (Height, Width, Channels)
    img = np.expand_dims(img, axis=-1)
    
    return img

def create_dataset(data_path="data/raw"):
    """
    Builds the dataset for the FaceAuth CNN model.
    """
    X, y = [], []
    
    if not os.path.exists(data_path):
        print(f"Error: Path '{data_path}' does not exist.")
        return None, None

    users = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]

    for username in users:
        user_dir = os.path.join(data_path, username)
        
        for img_file in os.listdir(user_dir):
            # Filtering for common image formats
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(user_dir, img_file)
            try:
                features = preprocess_image(img_path)
                X.append(features)
                y.append(username)
            except Exception as e:
                print(f"Skipping {img_path}: {e}")

    # Convert lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)

    # Label Encoding: Convert string names to integers (e.g., "John" -> 0, "Sara" -> 1)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print("-" * 30)
    print(f"Dataset summary for FaceAuth:")
    print(f"Total Samples: {X.shape[0]}")
    print(f"Input Shape: {X.shape[1:]} (Height, Width, Channels)")
    print(f"Unique Users: {len(np.unique(y_encoded))}")
    print("-" * 30)

    return X, y_encoded, le

