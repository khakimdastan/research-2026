"""Classification metrics + edge-efficiency measurements."""

from __future__ import annotations
import time
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           class_names: list[str]) -> dict:
    labels = list(range(len(class_names)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(
            f1_score(y_true, y_pred, labels=labels,
                     average="macro", zero_division=0)
        ),
        "precision_macro": float(
            precision_score(y_true, y_pred, labels=labels,
                            average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, labels=labels,
                         average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels
        ).tolist(),
        "report": classification_report(
            y_true, y_pred,
            labels=labels, target_names=class_names,
            digits=4, output_dict=True, zero_division=0,
        ),
    }


def time_classical_inference(model, feature_fn, X_window: np.ndarray,
                              fs: int, n_repeats: int = 50) -> dict:
    """Returns ms per single window for the *full* pipeline (feat + predict)."""
    if len(X_window) == 0:
        return {"feature_ms": 0.0, "predict_ms": 0.0, "total_ms": 0.0}
    sample = X_window[:1]
    t_feat = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        feats = feature_fn(sample, fs)
        t_feat.append((time.perf_counter() - t0) * 1000)
    t_pred = []
    feats = feature_fn(sample, fs)
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        _ = model.predict(feats)
        t_pred.append((time.perf_counter() - t0) * 1000)
    feat_ms = float(np.median(t_feat))
    pred_ms = float(np.median(t_pred))
    return {
        "feature_ms": feat_ms,
        "predict_ms": pred_ms,
        "total_ms":   feat_ms + pred_ms,
    }


def time_cnn_inference(model, X_window: np.ndarray, device: str,
                        n_repeats: int = 50) -> dict:
    if len(X_window) == 0:
        return {"forward_ms": 0.0, "total_ms": 0.0}
    import torch
    model.eval()
    x = torch.tensor(X_window[:1, None, :], dtype=torch.float32, device=device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
        ts = []
        for _ in range(n_repeats):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
    forward_ms = float(np.median(ts))
    return {"forward_ms": forward_ms, "total_ms": forward_ms}
