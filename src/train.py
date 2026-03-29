import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from .feature_engineering import load_features
from .evaluate import evaluate_model
from config import MODEL_PATH, TEST_SIZE, RANDOM_STATE, KNN_NEIGHBORS

def train_model(X, y):
    """Trains a KNN model and returns it."""
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS)
    knn.fit(X_train, y_train)
    print("Model trained successfully!")
    
    # Evaluate
    evaluate_model(knn, X_test, y_test)
    
    return knn

def save_model(model, path):
    """Saves the trained model to the specified path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Trained model saved at {path}")

def main():
    # Load features & labels
    X, y = load_features()
    
    # Train and Evaluate
    model = train_model(X, y)
    
    # Save model
    save_model(model, MODEL_PATH)

if __name__ == "__main__":
    main()