# models/svm.py
"""
Incremental SVM wrapper with support for:
- Linear SVM via SGDClassifier (true incremental)
- RBF & Poly SVM via Random Fourier Features (incremental kernel approximation)
Fully Pylance-safe and Render-friendly.
"""

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder


class SVMModel:
    def __init__(self, hyperparams=None):
        hyperparams = hyperparams or {}

        self.kernel = hyperparams.get("kernel", "linear")
        self.C = float(hyperparams.get("C", 1.0))
        self.gamma = float(hyperparams.get("gamma", 0.5))

        self.le = LabelEncoder()
        self.is_initialized = False
        self.fitted = False

        # Random Fourier Features for RBF/poly
        self.rff_dim = 200
        self.W = None
        self.b = None

    # ---------------------------------------------------
    # Random Fourier Features (RBF approximation)
    # ---------------------------------------------------
    def _build_rff(self, X):
        if self.W is None:
            np.random.seed(42)
            self.W = np.random.normal(
                scale=np.sqrt(2 * self.gamma),
                size=(X.shape[1], self.rff_dim)
            )
            self.b = np.random.uniform(0, 2*np.pi, self.rff_dim)

    def _rff_transform(self, X):
        proj = X @ self.W + self.b
        return np.sqrt(2 / self.rff_dim) * np.cos(proj)

    # ---------------------------------------------------
    # Initialization
    # ---------------------------------------------------
    def _ensure_initialized(self, y, X):
        if not self.is_initialized:

            # Encode labels
            y = np.asarray(y)
            self.le.fit(y)
            y_enc = np.asarray(self.le.transform(y), dtype=int)
            self.classes_ = np.unique(y_enc)

            # Build RFF before initializing model
            if self.kernel in ("rbf", "poly"):
                self._build_rff(X)
                X = self._rff_transform(X)

            # Create SGD linear SVM classifier
            alpha = 1.0 / self.C
            self.model = SGDClassifier(
                loss="hinge",
                alpha=alpha,
                learning_rate="constant",
                eta0=0.001,
                warm_start=True
            )

            # Dummy init with correct feature dimension
            dummy_X = np.zeros((1, X.shape[1]))
            dummy_y = np.asarray([self.classes_[0]], dtype=int)

            self.model.partial_fit(dummy_X, dummy_y, classes=self.classes_)
            self.is_initialized = True

    # ---------------------------------------------------
    # Training step (incremental)
    # ---------------------------------------------------
    def train_step(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        # Initialize
        self._ensure_initialized(y, X)

        # Encode labels
        y_enc = np.asarray(self.le.transform(y), dtype=int)

        # Transform non-linear kernels
        if self.kernel in ("rbf", "poly"):
            X = self._rff_transform(X)

        # Shuffle
        idx = np.arange(len(X))
        np.random.shuffle(idx)

        # Incremental update
        self.model.partial_fit(X[idx], y_enc[idx], classes=self.classes_)
        self.fitted = True

    # ---------------------------------------------------
    # Prediction
    # ---------------------------------------------------
    def predict(self, X):
        if not self.fitted:
            raise RuntimeError("Call train_step first.")

        X = np.asarray(X, dtype=float)
        if self.kernel in ("rbf", "poly"):
            X = self._rff_transform(X)

        preds = self.model.predict(X)
        return self.le.inverse_transform(preds)

    # ---------------------------------------------------
    # Probability prediction
    # ---------------------------------------------------
    def predict_proba(self, X):
        if not self.fitted:
            raise RuntimeError("Call train_step first.")

        X = np.asarray(X, dtype=float)
        if self.kernel in ("rbf", "poly"):
            X = self._rff_transform(X)

        # Logistic/softmax fallback via decision_function
        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X)

            # Binary logistic approximation
            if scores.ndim == 1:
                probs = 1 / (1 + np.exp(-scores))
                return np.vstack([1 - probs, probs]).T

            # Multi-class softmax
            exp_s = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            return exp_s / exp_s.sum(axis=1, keepdims=True)

        # One-hot fallback
        preds = self.model.predict(X)
        probs = np.zeros((len(preds), len(self.classes_)))

        transform = self.le.transform

        for i, lab in enumerate(preds):
            cls_idx = int(np.asarray(transform([lab]), dtype=int)[0])  # FINAL FIX
            probs[i, cls_idx] = 1.0

        return probs

