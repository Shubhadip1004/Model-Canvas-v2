# utils/data_loader.py
"""
Model Canvas 2.0
Dataset loader — returns ONLY (X_scaled, y, labels)
Fully Pylance-safe.
"""

from typing import cast
import numpy as np
from sklearn.utils import Bunch
from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    load_diabetes,
    make_moons,
    make_circles,
    make_blobs
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def load_dataset(name: str):
    name = name.lower()

    # ---------------------------------------------------------
    # SKLEARN BUILT-IN DATASETS (Cast is required for Pylance)
    # ---------------------------------------------------------

    if name == "iris":
        ds = cast(Bunch, load_iris(return_X_y=False, as_frame=False))
        X = np.asarray(ds.data, dtype=float)
        y = np.asarray(ds.target)
        labels = list(ds.target_names)

    elif name == "wine":
        ds = cast(Bunch, load_wine(return_X_y=False, as_frame=False))
        X = np.asarray(ds.data, dtype=float)
        y = np.asarray(ds.target)
        labels = list(ds.target_names)

    elif name == "breast_cancer":
        ds = cast(Bunch, load_breast_cancer(return_X_y=False, as_frame=False))
        X = np.asarray(ds.data, dtype=float)
        y = np.asarray(ds.target)
        labels = list(ds.target_names)

    # ---------------------------------------------------------
    # DIABETES (Regression target -> convert to 3 classes)
    # ---------------------------------------------------------

    elif name == "diabetes":
        ds = cast(Bunch, load_diabetes(return_X_y=False, as_frame=False))

        X_raw = np.asarray(ds.data, dtype=float)
        y_raw = np.asarray(ds.target, dtype=float)

        # Convert regression outputs into classes
        cuts = np.quantile(y_raw, [0.33, 0.66])
        y = np.digitize(y_raw, bins=cuts)  # Produces 0,1,2

        labels = ["Low", "Medium", "High"]

        # PCA to reduce to 2D for visualization
        pca = PCA(n_components=2)
        X = pca.fit_transform(X_raw)

    # ---------------------------------------------------------
    # SYNTHETIC DATASETS
    # ---------------------------------------------------------

    elif name == "moons":
        X, y = make_moons(n_samples=500, noise=0.15, random_state=42)
        labels = ["Class 0", "Class 1"]

    elif name == "circles":
        X, y = make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=42)
        labels = ["Class 0", "Class 1"]

    elif name == "blobs":
        blob_data = make_blobs(n_samples=600, centers=3, cluster_std=2.0, random_state=42)
        X = blob_data[0]  # Pylance-safe
        y = blob_data[1]
        labels = ["Class 0", "Class 1", "Class 2"]

    else:
        raise ValueError(f"Unknown dataset: {name}")

    # ---------------------------------------------------------
    # STANDARD SCALING (Needed for plotting & training)
    # ---------------------------------------------------------

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, labels
