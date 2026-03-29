# src/train.py
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from .feature_engineering import load_features
from .evaluate import evaluate_model
from config import MODEL_PATH

def main():
    # Load features & labels
    X, y = load_features()
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    print("Model trained successfully!")
    
    # Evaluate
    evaluate_model(knn, X_test, y_test)
    
    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(knn, f)
    print(f"Trained model saved at {MODEL_PATH}")

if __name__ == "__main__":
    main()