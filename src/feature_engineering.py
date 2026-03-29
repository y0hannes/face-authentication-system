# src/feature_engineering.py
import numpy as np
from config import IMAGE_SIZE

def load_features():
    """
    Dummy data for testing ML pipeline.
    Returns:
        X -> features (flattened)
        y -> labels (user IDs)
    """
    print("No images found, using dummy data for testing...")

    # 3 users, 3 images each, flattened 64x64
    X = np.array([
        [1]*IMAGE_SIZE[0]*IMAGE_SIZE[1],
        [1]*IMAGE_SIZE[0]*IMAGE_SIZE[1],
        [1]*IMAGE_SIZE[0]*IMAGE_SIZE[1],
        [2]*IMAGE_SIZE[0]*IMAGE_SIZE[1],
        [2]*IMAGE_SIZE[0]*IMAGE_SIZE[1],
        [2]*IMAGE_SIZE[0]*IMAGE_SIZE[1],
        [3]*IMAGE_SIZE[0]*IMAGE_SIZE[1],
        [3]*IMAGE_SIZE[0]*IMAGE_SIZE[1],
        [3]*IMAGE_SIZE[0]*IMAGE_SIZE[1],
    ])
    y = np.array([0,0,0,1,1,1,2,2,2])
    return X, y