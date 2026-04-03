import os

# Project Root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Paths
DATA_PATH  = os.path.join(PROJECT_ROOT, "data", "raw")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "face_model.pkl")

# Image parameters
IMAGE_SIZE = (64, 64)

# Capture parameters
CAPTURE_COUNT = 15
CAPTURE_DELAY = 1.5   # seconds between captures

# Quality thresholds
BLUR_THRESHOLD       = 50.0
BRIGHTNESS_THRESHOLD = 80

# Training parameters
TEST_SIZE    = 0.2
RANDOM_STATE = 42
KNN_NEIGHBORS = 3

# Face detection parameters
HAAR_SCALE_FACTOR    = 1.1
HAAR_MIN_NEIGHBORS   = 6
HAAR_MIN_SIZE        = (100, 100)
HAAR_EYE_MIN_NEIGHBORS = 10

# Prediction parameters
THRESHOLD = 5000.0