DATA_PATH = "data/raw/"
MODEL_PATH = "models/face_model.pkl"

IMAGE_SIZE = (64, 64)

# Capture parameters
CAPTURE_COUNT = 15
CAPTURE_DELAY = 1.5  

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