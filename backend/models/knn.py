# models/knn.py
"""
KNN wrapper. KNN has no training stage; train_step is a no-op.
We provide predict and predict_proba through sklearn's KNeighborsClassifier.
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

class KNNModel:
    def __init__(self, hyperparams=None):
        hyperparams = hyperparams or {}
        self.k = int(hyperparams.get("n_neighbors", 5))
        self.model = KNeighborsClassifier(n_neighbors=self.k)
        self.le = LabelEncoder()
        self.fitted = False

    def train_step(self, X, y):
        # Fit the entire dataset (fast for small 2D toy datasets)
        self.le.fit(y)
        y_enc = self.le.transform(y)
        self.model.set_params(n_neighbors=self.k)
        self.model.fit(X, y_enc)
        self.fitted = True

    def predict(self, X):
        if not self.fitted:
            raise RuntimeError("KNN model not fitted. Call train_step first.")
        preds = self.model.predict(X)
        return self.le.inverse_transform(preds)

    def predict_proba(self, X):
        if not self.fitted:
            raise RuntimeError("KNN model not fitted. Call train_step first.")
        return self.model.predict_proba(X)
