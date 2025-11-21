# backend/backend/utils/boundary_plot.py
"""
Compute a meshgrid and predictions for visualization.
Provides a JSON-friendly dict with 'nx','ny','preds','x_min','x_max','y_min','y_max'.
"""

from typing import Any, Dict
import numpy as np


def compute_grid_for_frontend(model: Any, X: np.ndarray, resolution: int = 120, padding: float = 0.4) -> Dict[str, Any]:
    """
    Given a fitted model (or a model that can predict on arbitrary points),
    and the original 2D data X (n_samples, 2), computes a meshgrid and predicts
    labels for every grid point. Returns a dict suitable for JSON serialization.

    Args:
        model: object with .predict(X_grid) or .predict_proba(X_grid)
        X: ndarray shape (n_samples, 2)
        resolution: number of points per axis
        padding: extra margin around data range

    Returns:
        dict with keys: nx, ny, preds (flat list), x_min, x_max, y_min, y_max
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X must be (n_samples, 2) for 2D visualization")

    x_min, x_max = float(X[:, 0].min() - padding), float(X[:, 0].max() + padding)
    y_min, y_max = float(X[:, 1].min() - padding), float(X[:, 1].max() + padding)

    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(y_min, y_max, resolution)
    xx, yy = np.meshgrid(xs, ys)  # yy rows, xx cols

    pts = np.c_[xx.ravel(), yy.ravel()]  # (resolution*resolution, 2)

    # If the model expects transformed features (RFF etc.), model should handle it in predict().
    try:
        preds = model.predict(pts)
    except Exception:
        # last-resort: try predict_proba -> take argmax
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(pts)
            preds = np.argmax(probs, axis=1)
        else:
            # if model can't predict, return zeros
            preds = np.zeros(pts.shape[0], dtype=int)

    preds_arr = np.asarray(preds).astype(int)
    # reshape to (ny, nx) i.e. same shape as yy, xx
    try:
        preds_grid = preds_arr.reshape((len(ys), len(xs)))
    except Exception:
        # fallback: try row-major reshape
        preds_grid = preds_arr.reshape(yy.shape)

    return {
        "nx": int(preds_grid.shape[1]),
        "ny": int(preds_grid.shape[0]),
        "preds": preds_grid.ravel(order="C").tolist(),
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
    }
