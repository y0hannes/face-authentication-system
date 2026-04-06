import os

# Project Root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Paths
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "face_model.pkl")

# Image parameters
IMAGE_SIZE = (64, 64)

# Capture parameters
CAPTURE_COUNT = 60
CAPTURE_DELAY = 0.15  # seconds between captures for a high-intensity stream

# Quality thresholds
BLUR_THRESHOLD = 50.0
BRIGHTNESS_THRESHOLD = 80

# Training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
KNN_NEIGHBORS = 3

# Face detection parameters
HAAR_SCALE_FACTOR = 1.1
HAAR_MIN_NEIGHBORS = 6
HAAR_MIN_SIZE = (100, 100)
HAAR_EYE_MIN_NEIGHBORS = 10

# Prediction parameters
# Using HOG, cosine distance is usually more robust than Euclidean.
# A threshold of 0.30-0.40 for cosine distance is typical for HOG.
THRESHOLD = 0.35

# HOG (Histogram of Oriented Gradients) parameters
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"
HOG_VISUALIZE = False
HOG_MULTICHANNEL = False
HOG_TRANSFORM_SQRT = True  # Power-law compression (gamma correction) for better performance
