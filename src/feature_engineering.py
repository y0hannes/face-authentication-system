import os
import pickle
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder

from config import (
    DATA_PATH, 
    MODEL_PATH,
    HOG_ORIENTATIONS,
    HOG_PIXELS_PER_CELL,
    HOG_CELLS_PER_BLOCK,
    HOG_BLOCK_NORM,
    HOG_VISUALIZE,
    HOG_MULTICHANNEL
)
from src.preprocessing import preprocess_image


# Path where the label encoder is persisted alongside the model
_ENCODER_PATH = MODEL_PATH.replace(".pkl", "_labels.pkl")


def extract_features(processed_image: np.ndarray) -> np.ndarray:
    """
    Extract HOG features from a preprocessed image.
    
    Args:
        processed_image: Grayscale [0, 1] face image.
        
    Returns:
        Flattened HOG feature vector.
    """
    from config import HOG_TRANSFORM_SQRT
    features = hog(
        processed_image,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        visualize=HOG_VISUALIZE,
        channel_axis=None if not HOG_MULTICHANNEL else -1,
        transform_sqrt=HOG_TRANSFORM_SQRT
    )
    return features


def create_dataset(
    data_path: str = None,
    progress_callback=None,
) -> tuple:
    root = data_path or DATA_PATH

    if not os.path.exists(root):
        print(f"[feature_engineering] Data path '{root}' does not exist.")
        return None, None, None

    users = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))

    if not users:
        print(f"[feature_engineering] No user directories found in '{root}'.")
        return None, None, None

    # Collect all (img_path, username) pairs first so we know the total
    all_items = []
    for username in users:
        user_dir = os.path.join(root, username)
        for img_file in os.listdir(user_dir):
            if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                all_items.append((os.path.join(user_dir, img_file), username))

    total = len(all_items)
    if total == 0:
        print("[feature_engineering] No image files found.")
        return None, None, None

    X, y = [], []
    for idx, (img_path, username) in enumerate(all_items):
        if progress_callback is not None:
            progress_callback(idx + 1, total)

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"[feature_engineering] Skipping unreadable file: {img_path}")
            continue

        processed = preprocess_image(img_bgr)
        if processed is None:
            print(f"[feature_engineering] No face detected in: {img_path}")
            continue

        features = extract_features(processed)
        X.append(features)
        y.append(username)

    if len(X) == 0:
        print("[feature_engineering] No valid face images after preprocessing.")
        return None, None, None

    X = np.array(X, dtype="float32")
    y = np.array(y)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(
        f"[feature_engineering] Dataset ready – "
        f"{X.shape[0]} samples, {len(le.classes_)} users: {list(le.classes_)}"
    )
    return X, y_encoded, le


def save_label_encoder(le: LabelEncoder) -> None:
    """Persist the label encoder next to the model file."""
    os.makedirs(os.path.dirname(_ENCODER_PATH), exist_ok=True)
    with open(_ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)
    print(f"[feature_engineering] Label encoder saved to '{_ENCODER_PATH}'.")


def get_label_encoder() -> LabelEncoder | None:

    if not os.path.exists(_ENCODER_PATH):
        return None
    with open(_ENCODER_PATH, "rb") as f:
        return pickle.load(f)


def load_features(progress_callback=None) -> tuple:

    X, y, le = create_dataset(progress_callback=progress_callback)
    if X is None:
        return None, None
    save_label_encoder(le)
    return X, y