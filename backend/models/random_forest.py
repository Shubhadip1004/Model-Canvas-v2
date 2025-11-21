# models/random_forest.py
"""
RandomForest wrapper using warm_start to add trees incrementally.
- Each train_step increases n_estimators by 1 (or by step size).
- This produces progressively better boundaries as trees are added.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

class RandomForestModel:
    def __init__(self, hyperparams=None):
        hyperparams = hyperparams or {}
        self.base_trees = int(hyperparams.get("n_estimators", 50))
        self.step = int(hyperparams.get("step", 1)) or 1
        self.current_estimators = 1
        self.le = LabelEncoder()
        # create RF with warm_start so we can incrementally add trees
        self.model = RandomForestClassifier(n_estimators=self.current_estimators, warm_start=True)
        self.fitted = False

    def train_step(self, X, y):
        self.le.fit(y)
        y_enc = self.le.transform(y)
        # increase estimators gradually
        next_estimators = min(self.current_estimators + self.step, self.base_trees)
        self.model.set_params(n_estimators=next_estimators)
        # calling fit with warm_start will add the new trees to the existing ensemble
        self.model.fit(X, y_enc)
        self.current_estimators = next_estimators
        self.fitted = True

    def predict(self, X):
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call train_step first.")
        preds = self.model.predict(X)
        return self.le.inverse_transform(preds)

    def predict_proba(self, X):
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call train_step first.")
        return self.model.predict_proba(X)