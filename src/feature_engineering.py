"""
TEMPORARY MODULE

This file uses simulated data for testing the ML pipeline.
It will be replaced by the actual feature extraction pipeline.
"""

import numpy as np
from config import IMAGE_SIZE

def load_features():
    print("Using realistic simulated data for testing...")

    num_users = 3
    samples_per_user = 10
    feature_size = IMAGE_SIZE[0] * IMAGE_SIZE[1]

    X = []
    y = []

    for user in range(num_users):
        for _ in range(samples_per_user):
            # Fully random features per sample (no identical base)
            sample = np.random.rand(feature_size)
            X.append(sample)
            y.append(user)

    return np.array(X), np.array(y)