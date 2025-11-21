# models/neural_net.py
"""
Incremental warm_start neural net for live decision boundary visualization.
"""

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

def parse_layers(val):
    if isinstance(val, str):
        parts = [p.strip() for p in val.split(",") if p.strip()]
        return tuple(int(x) for x in parts) if parts else (50,)
    if isinstance(val, (list, tuple)):
        return tuple(int(x) for x in val)
    return (50,)

class NeuralNetModel:
    def __init__(self, hyperparams=None):
        hyperparams = hyperparams or {}
        layers = parse_layers(hyperparams.get("hidden_layer_sizes", "50,"))
        lr = float(hyperparams.get("learning_rate_init", 0.001))
        alpha = float(hyperparams.get("alpha", 0.0001))

        self.model = MLPClassifier(
            hidden_layer_sizes=layers,
            learning_rate_init=lr,
            alpha=alpha,
            warm_start=True,
            max_iter=1,
            solver="adam",
            verbose=False
        )

        self.le = LabelEncoder()
        self.fitted = False

    def train_step(self, X, y):
        """
        One incremental training epoch (max_iter=1 + warm_start=True)
        """
        # Force numpy arrays to remove Pylance warnings
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        # Encode labels
        self.le.fit(y)
        y_enc = np.asarray(self.le.transform(y), dtype=int)

        # Shuffle for SGD-like behavior
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)

        # One training iteration
        self.model.fit(X[idx], y_enc[idx])
        self.fitted = True

    def predict(self, X):
        if not self.fitted:
            raise RuntimeError("Call train_step first.")

        X = np.asarray(X, dtype=float)
        preds = self.model.predict(X)
        return self.le.inverse_transform(preds)

    def predict_proba(self, X):
        if not self.fitted:
            raise RuntimeError("Call train_step first.")

        X = np.asarray(X, dtype=float)
        return self.model.predict_proba(X)
