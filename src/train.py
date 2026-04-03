import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from config import MODEL_PATH, TEST_SIZE, RANDOM_STATE, KNN_NEIGHBORS
from src.feature_engineering import load_features
from src.evaluate import evaluate_model


def train_model(X, y) -> KNeighborsClassifier:
    """
    Train a KNN classifier.

    Returns:
        A fitted KNeighborsClassifier.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    knn = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS)
    knn.fit(X_train, y_train)
    print("[train] Model trained successfully.")
    return knn, X_test, y_test


def save_model(model, path: str = MODEL_PATH) -> None:
    """Persist the trained model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[train] Model saved to '{path}'.")


def train_and_save(progress_callback=None) -> dict:
    # 1. Load dataset
    X, y = load_features(progress_callback=progress_callback)

    if X is None or len(X) == 0:
        return {
            "success": False,
            "message": "Training failed: no training data found. "
                       "Please register at least one user first.",
            "accuracy": None,
            "report": None,
            "fig": None,
        }

    # 2. Train
    model, X_test, y_test = train_model(X, y)

    # 3. Evaluate
    eval_results = evaluate_model(model, X_test, y_test)

    # 4. Save
    save_model(model)

    return {
        "success": True,
        "message": (
            f"Training complete. "
            f"Accuracy: {eval_results['accuracy'] * 100:.2f}%  |  "
            f"Model saved to '{MODEL_PATH}'."
        ),
        "accuracy": eval_results["accuracy"],
        "report": eval_results["report"],
        "fig": eval_results["fig"],
    }


if __name__ == "__main__":
    result = train_and_save()
    print(result["message"])