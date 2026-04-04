import pickle
import numpy as np

from config import MODEL_PATH, THRESHOLD
from src.preprocessing import preprocess_image
from src.feature_engineering import get_label_encoder

_model = None
_label_encoder = None


def get_model():
    """Load and cache the KNN model. Returns None if not trained yet."""
    global _model
    if _model is None:
        try:
            with open(MODEL_PATH, "rb") as f:
                _model = pickle.load(f)
            print(f"[predict] Model loaded from '{MODEL_PATH}'.")
        except FileNotFoundError:
            _model = None
    return _model


def reload_model():
    """Force a fresh reload of the model and label encoder (call after re-training)."""
    global _model, _label_encoder
    _model = None
    _label_encoder = None
    # Pre-emptively load them to verify availability
    get_model()
    _get_label_encoder()
    return _model


def _get_label_encoder():
    global _label_encoder
    # If the file exists but we haven't loaded it yet (or it was cleared)
    if _label_encoder is None:
        _label_encoder = get_label_encoder()
    return _label_encoder


def predict_face(image: np.ndarray) -> dict:

    model = get_model()
    if model is None:
        return {
            "status": "model_not_ready",
            "username": None,
            "distance": None,
            "message": "Model has not been trained yet. Please train the model first.",
        }

    processed = preprocess_image(image)
    if processed is None:
        return {
            "status": "no_face",
            "username": None,
            "distance": None,
            "message": "No face detected in the frame.",
        }

    flattened = processed.flatten().reshape(1, -1)

    raw_prediction = model.predict(flattened)[0]
    distances, _ = model.kneighbors(flattened)
    distance = float(distances[0][0])

    # Decode integer label to username string if encoder is available
    le = _get_label_encoder()
    if le is not None:
        try:
            username = str(le.inverse_transform([raw_prediction])[0])
        except Exception:
            username = str(raw_prediction)
    else:
        username = str(raw_prediction)

    if distance > THRESHOLD:
        return {
            "status": "unknown",
            "username": None,
            "distance": distance,
            "message": f"Face not recognised (distance {distance:.1f} > threshold {THRESHOLD}).",
        }

    return {
        "status": "authenticated",
        "username": username,
        "distance": distance,
        "message": f"Welcome, {username}! (distance {distance:.1f})",
    }


def is_model_ready() -> bool:
    """Quick check for the UI to gate access to the Login page."""
    return get_model() is not None
