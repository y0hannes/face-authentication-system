# src/train.py
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from .feature_engineering import load_features
from .evaluate import evaluate_model
from config import MODEL_PATH, TEST_SIZE, RANDOM_STATE, KNN_NEIGHBORS


def train_model(X, y, k=KNN_NEIGHBORS):
    """
    Reusable training function.
    Trains a KNN model with configurable k.
    
    Returns:
        model: trained KNN classifier
        X_test, y_test: for evaluation
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    return model, X_test, y_test


def main():
    # Load features
    X, y = load_features()

    # Train model using K from config
    model, X_test, y_test = train_model(X, y)
    print("Model trained successfully!")

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved at {MODEL_PATH}")


if __name__ == "__main__":
    main()