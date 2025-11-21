# models/decision_tree.py
"""
DecisionTree wrapper that simulates incremental training by increasing max_depth each step.
- At iteration 1 => depth = 1
- Next iteration => depth = 2
- Continues until base_max_depth
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

class DecisionTreeModel:
    def __init__(self, hyperparams=None):
        hyperparams = hyperparams or {}
        self.base_max_depth = int(hyperparams.get("max_depth", 5))
        
        self.current_depth = 1
        self.model = None   # IMPORTANT: initialize to None
        self.le = LabelEncoder()
        self.fitted = False

    def train_step(self, X, y):
        """
        Fit a tree with increasing depth each iteration.
        This ensures progressive visualization.
        """
        # Fit label encoder every time (safe)
        self.le.fit(y)
        y_enc = self.le.transform(y)

        # Determine the depth for this iteration
        depth = min(self.current_depth, self.base_max_depth)

        # CREATE THE MODEL HERE — this fixes your error
        self.model = DecisionTreeClassifier(max_depth=depth)

        # Fit model
        self.model.fit(X, y_enc)

        # Mark as trained
        self.fitted = True
        
        # Increase depth for next iteration
        self.current_depth += 1

    def predict(self, X):
        if not self.fitted or self.model is None:
            raise RuntimeError("Model not trained. Call train_step() before predict().")
        
        X = np.atleast_2d(X)
        preds = self.model.predict(X)
        return self.le.inverse_transform(preds)

    def predict_proba(self, X):
        if not self.fitted or self.model is None:
            raise RuntimeError("Model not trained. Call train_step() before predict().")

        X = np.atleast_2d(X)
        return self.model.predict_proba(X)
