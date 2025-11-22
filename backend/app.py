from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
import os
import json
import time
import uuid
import queue
import threading
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss

# Do NOT modify this file per user request
from utils.data_loader import load_dataset

# sklearn models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


# ------------------------------------------------------------
# Flask setup
# ------------------------------------------------------------
app = Flask(__name__, static_folder=None)
CORS(app, origins="*")

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

SESSIONS = {}
SESSIONS_LOCK = threading.Lock()


# ------------------------------------------------------------
# Build classifier from name
# ------------------------------------------------------------
def build_classifier(algo: str, hyper: dict):
    algo = algo.lower()
    mode = hyper.get("mode", "educational")

    if algo == "knn":
        return KNeighborsClassifier(n_neighbors=int(hyper.get("n_neighbors", 5)))

    elif algo == "svm":
        return SVC(
            kernel=hyper.get("kernel", "rbf"),
            C=float(hyper.get("C", 1.0)),
            probability=False
        )

    elif algo in ("decision_tree", "dt"):
        md = hyper.get("max_depth", None)
        md = None if md in (None, "None") else int(md)
        return DecisionTreeClassifier(max_depth=md)

    elif algo in ("random_forest", "rf"):
        n = int(hyper.get("n_estimators", 100))
        return RandomForestClassifier(n_estimators=n)

    elif algo in ("neural_network", "neural_net", "mlp"):
        hidden = hyper.get("hidden_layer_sizes", "50")
        # robust parsing: allow "50," etc.
        hidden = tuple(int(x.strip()) for x in hidden.split(",") if x.strip() != "")
        lr = float(hyper.get("learning_rate_init", 0.001))
        act_func = hyper.get("activation_function", 'relu')
        if mode == "educational":
            # slow, step-by-step learning
            return MLPClassifier(
                activation=act_func,
                hidden_layer_sizes=hidden,
                learning_rate_init=lr,
                warm_start=True,
                max_iter=1  # runs 1 step each iteration
            )
        else:  # optimized mode
            return MLPClassifier(
                activation=act_func,
                hidden_layer_sizes=hidden,
                learning_rate_init=lr,
                max_iter=50,  # trains much more each iteration
                warm_start=True
            )

    else:
        # Logistic regression family
        if mode == 'optimized':
            max_iter = 200
        else:
            max_iter = 20
        return LogisticRegression(max_iter=max_iter)


# ------------------------------------------------------------
# Utility: Convert dataset into scatter format
# ------------------------------------------------------------
def pack_points(X, y):
    if X.shape[1] >= 2:
        return {
            "x": X[:, 0].astype(float).tolist(),
            "y": X[:, 1].astype(float).tolist(),
            "labels": y.tolist(),
        }
    else:
        return {
            "x": X[:, 0].astype(float).tolist(),
            "y": [0.0] * len(X),
            "labels": y.tolist(),
        }


# ------------------------------------------------------------
# Build 2D decision grid
# ------------------------------------------------------------
def build_grid(model, X, NX=120, NY=80):
    if X.shape[1] < 2:
        x0 = X[:, 0]
        x_min, x_max = x0.min() - 0.5, x0.max() + 0.5
        y_min, y_max = -1, 1
    else:
        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        y_min, y_max = X[:, 1].min(), X[:, 1].max()
        dx = 0.2 * (x_max - x_min)
        dy = 0.2 * (y_max - y_min)
        x_min, x_max = x_min - dx, x_max + dx
        y_min, y_max = y_min - dy, y_max + dy

    xs = np.linspace(x_min, x_max, NX)
    ys = np.linspace(y_min, y_max, NY)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.c_[xx.ravel(), yy.ravel()]

    if X.shape[1] > 2:
        extra = np.zeros((pts.shape[0], X.shape[1] - 2))
        pts_full = np.hstack((pts, extra))
    else:
        pts_full = pts

    try:
        preds = model.predict(pts_full).tolist()
    except Exception:
        preds = [0] * (NX * NY)

    return {
        "nx": NX,
        "ny": NY,
        "x_min": float(x_min),
        "x_max": float(x_max),
        "y_min": float(y_min),
        "y_max": float(y_max),
        "preds": preds,
    }


# ------------------------------------------------------------
# Dataset Preview Endpoint
# ------------------------------------------------------------
@app.route("/preview_dataset")
def preview_dataset():
    ds = request.args.get("dataset", "iris")
    X, y, labels = load_dataset(ds)

    try:
        Xtr, Xte, ytr, yte = train_test_split(
            X,
            y,
            test_size=0.33,
            random_state=42,
            stratify=y if len(np.unique(y)) > 1 else None,
        )
    except Exception:
        N = len(X)
        c = int(N * 0.67)
        Xtr, Xte = X[:c], X[c:]
        ytr, yte = y[:c], y[c:]

    return jsonify({"train": pack_points(Xtr, ytr), "test": pack_points(Xte, yte)})


# ------------------------------------------------------------
# Worker thread for training
# ------------------------------------------------------------
def session_worker(sid):

    with SESSIONS_LOCK:
        cfg = SESSIONS[sid]

    dataset = cfg["dataset"]
    mode = cfg["mode"]
    algo = cfg["algo"]
    hyper = cfg["hyper"]
    epochs = cfg["epochs"]
    interval = cfg["interval"] / 1000
    q = cfg["queue"]

    X, y, labels = load_dataset(dataset)
    X = X.astype(float)

    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    try:
        class_names = [str(c) for c in labels]
    except Exception:
        class_names = [str(c) for c in np.unique(y)]

    p = np.random.permutation(len(Xtr))
    Xtr, ytr = Xtr[p], ytr[p]

    model = build_classifier(algo, hyper)
    classes = np.unique(ytr)
    N = len(Xtr)

    for it in range(1, epochs + 1):

        with SESSIONS_LOCK:
            if not SESSIONS[sid]["running"]:
                q.put(json.dumps({"status": "stopped"}))
                return

        # ---- incremental slice for educational mode ----
        n_cur = max(2, int(N * it / epochs))
        Xs, ys = Xtr[:n_cur], ytr[:n_cur]
        # --------------------------------------------------

        # ---- full dataset always shown in optimized mode ----
        X_plot, y_plot = (Xtr, ytr) if mode == "optimized" else (Xs, ys)
        # ------------------------------------------------------

        # Multi-class safety
        if len(np.unique(ys)) < 2:
            q.put(json.dumps({
                "iteration": it,
                "mode": mode,
                "acc": 0,
                "loss": 0,
                "train": pack_points(X_plot, y_plot),
                "test": {"x": [], "y": [], "labels": [], "preds": []},
                "grid": None,
                "confusion": None,
                "class_names": class_names,
                "feature_names": ["Feature 1", "Feature 2"],
                "note": "Skipped iteration — only one class present",
            }))
            time.sleep(interval)
            continue

        if algo == "knn":
            k = int(hyper.get("n_neighbors", 5))
            if len(ys) < k:
                q.put(json.dumps({
                    "iteration": it,
                    "mode": mode,
                    "acc": 0,
                    "loss": 0,
                    "train": pack_points(X_plot, y_plot),
                    "test": {"x": [], "y": [], "labels": [], "preds": []},
                    "grid": None,
                    "confusion": None,
                    "class_names": class_names,
                    "feature_names": ["Feature 1", "Feature 2"],
                    "note": f"Skipped — KNN requires at least k={k} samples",
                }))
                time.sleep(interval)
                continue

        try:
            if mode == "optimized":
                model.fit(Xtr, ytr)        # full training
            else:
                model.fit(Xs, ys)          # incremental training
        except Exception as e:
            q.put(json.dumps({
                "iteration": it,
                "mode": mode,
                "acc": 0,
                "loss": 0,
                "train": pack_points(X_plot, y_plot),
                "test": {"x": [], "y": [], "labels": [], "preds": []},
                "grid": None,
                "confusion": None,
                "class_names": class_names,
                "feature_names": ["Feature 1", "Feature 2"],
            }))
            time.sleep(interval)
            continue

        try:
            preds = model.predict(Xte)
        except ValueError as e:
            if "n_neighbors" in str(e):
                preds = np.zeros_like(yte)
            else:
                raise

        acc = float(accuracy_score(yte, preds))

        loss = 0.0
        if hasattr(model, "predict_proba"):
            try:
                prob = model.predict_proba(Xte)
                loss = float(log_loss(yte, prob, labels=classes))
            except Exception:
                pass

        try:
            conf = confusion_matrix(yte, preds).tolist()
        except Exception:
            conf = None

        grid = build_grid(model, X_plot)  # grid must match plotted data

        frame = {
            "iteration": it,
            "mode": mode,
            "acc": acc,
            "loss": loss,
            "train": pack_points(X_plot, y_plot),   # <-- FULL DATA SENT WHEN OPTIMIZED
            "test": {
                **pack_points(Xte, yte),
                "preds": preds.tolist(),
            },
            "grid": grid,
            "confusion": conf,
            "class_names": class_names,
            "feature_names": ["Feature 1", "Feature 2"],
        }

        q.put(json.dumps(frame))
        time.sleep(interval)

    q.put(json.dumps({"status": "done"}))


# ------------------------------------------------------------
# Start training
# ------------------------------------------------------------
@app.route("/start_training", methods=["POST"])
def start_training():
    body = request.get_json(force=True)

    sid = str(uuid.uuid4())
    q = queue.Queue()

    SESSIONS[sid] = {
        "dataset": body.get("dataset", "iris"),
        "mode": body.get("mode", "educational"),
        "algo": body.get("algo", "knn"),
        "hyper": body.get("hyperparams", {}),
        "epochs": int(body.get("epochs", 50)),
        "interval": int(body.get("interval_ms", 200)),
        "queue": q,
        "running": True,
    }

    threading.Thread(target=session_worker, args=(sid,), daemon=True).start()
    return jsonify({"session_id": sid})


# ------------------------------------------------------------
# SSE stream
# ------------------------------------------------------------
@app.route("/stream_updates")
def stream_updates():
    sid = request.args.get("session_id")
    if sid not in SESSIONS:
        return "invalid session", 400

    q = SESSIONS[sid]["queue"]

    def gen():
        last = time.time()
        while True:
            try:
                msg = q.get(timeout=0.5)
                yield f"data: {msg}\n\n"
            except queue.Empty:
                if not SESSIONS[sid]["running"] and q.empty():
                    break
                if time.time() - last > 15:
                    yield ": keepalive\n\n"
                    last = time.time()
        yield "data: {\"type\": \"closed\"}\n\n"

    return Response(gen(), mimetype="text/event-stream")


# ------------------------------------------------------------
# Stop training
# ------------------------------------------------------------
@app.route("/stop_training")
def stop_training():
    sid = request.args.get("session_id")
    if sid in SESSIONS:
        SESSIONS[sid]["running"] = False
        return "ok"
    return "invalid session", 400


# ------------------------------------------------------------
# Metadata
# ------------------------------------------------------------
@app.route("/metadata")
def metadata():
    return jsonify({
        "datasets": ["iris", "moons", "circles", "wine"],
        "algorithms": [
            "knn",
            "svm",
            "logistic",
            "decision_tree",
            "random_forest",
            "neural_network",
        ],
    })


# ------------------------------------------------------------
# Serve frontend for local development
# ------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(FRONTEND_DIR, path)


# ------------------------------------------------------------
# Run locally
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
