# src/evaluate.py
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model and print:
    - Overall accuracy
    - Confusion matrix
    - Per-class metrics (precision, recall, f1-score)
    """
    y_pred = model.predict(X_test)
    
    # Overall accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {acc*100:.2f}%\n")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm, "\n")
    
    # Per-class metrics
    report = classification_report(y_test, y_pred, digits=2)
    print("Per-class Metrics:")
    print(report)