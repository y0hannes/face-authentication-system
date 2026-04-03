import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Console fallback (useful when running from CLI / train.py)
    print(f"[evaluate] Accuracy: {acc * 100:.2f}%")
    print(f"[evaluate] Classification Report:\n{report}")
    print(f"[evaluate] Confusion Matrix:\n{cm}")

    # Build a non-blocking figure the UI can render
    fig = _build_cm_figure(cm)

    return {
        "accuracy": acc,
        "report": report,
        "cm": cm,
        "fig": fig,
    }


def _build_cm_figure(cm: np.ndarray):
    try:
        import matplotlib
        matplotlib.use("Agg")          # non-interactive backend — no GUI window
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        return fig

    except ImportError:
        return None