# models/logistic_reg.py
"""
Incremental Logistic Regression using SGDClassifier with 'log_loss'
(works like online logistic regression). Perfect for per-iteration
decision boundary visualization.
"""

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder


class LogisticReg:
    def __init__(self, hyperparams=None):
        hyperparams = hyperparams or {}

        self.lr = float(hyperparams.get("learning_rate_init", 0.01))
        self.alpha = float(hyperparams.get("alpha", 0.0001))

        # SGDClassifier using logistic regression loss
        self.model = SGDClassifier(
            loss="log_loss",                # logistic regression loss
            learning_rate="constant",
            eta0=self.lr,
            alpha=self.alpha,
            warm_start=True
        )

        self.le = LabelEncoder()
        self.is_initialized = False
        self.classes_ = None

    def _ensure_initialized(self, y):
        """Initialize label encoder + partial_fit classes on first call."""
        if not self.is_initialized:
            y = np.asarray(y)

            self.le.fit(y)
            self.classes_ = np.unique(self.le.transform(y))

            # First call to partial_fit() MUST include classes=...
            # We pass a dummy batch just to initialize weights.
            dummy_X = np.zeros((1, 2))
            dummy_y = np.asarray([self.classes_[0]])

            self.model.partial_fit(dummy_X, dummy_y, classes=self.classes_)
            self.is_initialized = True

    def train_step(self, X, y):
        """
        Perform one incremental training step (online logistic regression).
        This is fast + produces smooth decision boundary updates.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        # Initialize encoder & model if needed
        self._ensure_initialized(y)

        y_enc = np.asarray(self.le.transform(y), dtype=int)

        # Shuffle for stochasticity
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)

        # Incremental logistic regression update
        self.model.partial_fit(X[idx], y_enc[idx], classes=self.classes_)

    def predict(self, X):
        if not self.is_initialized:
            raise RuntimeError("Call train_step() before predict().")

        X = np.asarray(X, dtype=float)
        preds = self.model.predict(X)

        return self.le.inverse_transform(preds)

    def predict_proba(self, X):
        if not self.is_initialized:
            raise RuntimeError("Call train_step() before predict_proba().")

        X = np.asarray(X, dtype=float)

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)

        # Fallback probability (very rare case)
        scores = self.model.decision_function(X)

        # Binary or multi-class softmax
        if scores.ndim == 1:
            # binary
            probs = 1 / (1 + np.exp(-scores))
            return np.vstack([1 - probs, probs]).T
        else:
            # multi-class
            exp_s = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            return exp_s / exp_s.sum(axis=1, keepdims=True)
