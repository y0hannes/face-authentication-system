import pickle
from config import MODEL_PATH, THRESHOLD
from src.preprocessing import preprocess_image

# Load trained model (will be saved later by Hemen)
try:
    model = pickle.load(open(MODEL_PATH, "rb"))
except FileNotFoundError:
    model = None
    print("Model not trained yet. Hemen will train and save face_model.pkl")

def predict_face(image):
    if model is None:
        return "Model Not Ready"
    processed = preprocess_image(image)
    if processed is None:
        return "No Face Detected"
    flattened = processed.flatten()
    prediction = model.predict([flattened])
    distances, _ = model.kneighbors([flattened])
    if distances[0][0] < THRESHOLD:
        return prediction[0]
    else:
        return "Unknown"