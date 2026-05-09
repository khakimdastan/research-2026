"""Unified training & evaluation routines for classical and CNN models."""

from __future__ import annotations
import os
import pickle
import tempfile
from dataclasses import dataclass, field

import numpy as np
from tqdm.auto import tqdm

from .config import RunConfig
from .data import Dataset, normalize_per_window
from .features import extract_features, FEATURE_NAMES
from .models_classical import build_classical
from .models_cnn import build_cnn, count_params, model_size_mb
from .noise import add_awgn
from .metrics import (
    classification_metrics,
    time_classical_inference,
    time_cnn_inference,
)
from .utils import Timer


@dataclass
class RunResult:
    model: str
    dataset: str
    task: str
    metrics: dict = field(default_factory=dict)
    efficiency: dict = field(default_factory=dict)
    train_time_s: float = 0.0
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "dataset": self.dataset,
            "task": self.task,
            "metrics": self.metrics,
            "efficiency": self.efficiency,
            "train_time_s": self.train_time_s,
            "extras": self.extras,
        }


# ---------------------------------------------------------------------------
# Classical pipeline
# ---------------------------------------------------------------------------
def train_classical(model_name: str, ds: Dataset, cfg: RunConfig,
                    snr_db: float | None = None) -> RunResult:
    from sklearn.preprocessing import StandardScaler

    F_train = extract_features(ds.X_train, ds.fs)
    F_val   = extract_features(ds.X_val,   ds.fs)
    X_test  = ds.X_test if snr_db is None else add_awgn(ds.X_test, snr_db, cfg.seed)
    F_test  = extract_features(X_test, ds.fs)

    scaler  = StandardScaler().fit(F_train)
    F_train = scaler.transform(F_train)
    F_val   = scaler.transform(F_val)
    F_test  = scaler.transform(F_test)

    clf = build_classical(model_name, n_classes=ds.n_classes, seed=cfg.seed)

    with Timer() as t:
        clf.fit(F_train, ds.y_train)
    train_time = t.elapsed

    y_pred  = clf.predict(F_test)
    metrics = classification_metrics(ds.y_test, y_pred, ds.class_names)

    def feature_fn(X, fs):
        return scaler.transform(extract_features(X, fs))

    perf = time_classical_inference(clf, feature_fn, ds.X_test, ds.fs)

    with tempfile.NamedTemporaryFile(delete=False) as f:
        pickle.dump(clf, f)
        tmp_path = f.name
    try:
        size_mb = os.path.getsize(tmp_path) / (1024 ** 2)
    finally:
        os.unlink(tmp_path)

    eff = {
        "model_size_mb":        float(size_mb),
        "feature_ms_per_window": perf["feature_ms"],
        "predict_ms_per_window": perf["predict_ms"],
        "total_ms_per_window":   perf["total_ms"],
        "n_features":            len(FEATURE_NAMES),
    }
    return RunResult(
        model=model_name, dataset=ds.name, task=ds.task,
        metrics=metrics, efficiency=eff, train_time_s=train_time,
        extras={"snr_db": snr_db, "scaler": "StandardScaler"},
    )


# ---------------------------------------------------------------------------
# CNN pipeline
# ---------------------------------------------------------------------------
def train_cnn(model_name: str, ds: Dataset, cfg: RunConfig,
              device: str,
              snr_db: float | None = None,
              show_progress: bool = True) -> RunResult:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    Xtr_n = normalize_per_window(ds.X_train)
    Xva_n = normalize_per_window(ds.X_val)
    X_test_raw = ds.X_test if snr_db is None else add_awgn(ds.X_test, snr_db, cfg.seed)
    Xte_n = normalize_per_window(X_test_raw)

    def to_dl(X, y, shuffle):
        ds_ = TensorDataset(
            torch.tensor(X[:, None, :], dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )
        return DataLoader(ds_, batch_size=cfg.batch_size, shuffle=shuffle,
                          drop_last=shuffle, num_workers=0)

    tr_dl = to_dl(Xtr_n, ds.y_train, True)
    va_dl = to_dl(Xva_n, ds.y_val,   False)
    te_dl = to_dl(Xte_n, ds.y_test,  False)

    model = build_cnn(model_name, ds.n_classes).to(device)
    opt   = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs))
    crit  = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state   = None
    history      = {"train_loss": [], "val_loss": [], "val_acc": []}

    with Timer() as t_train:
        epoch_bar = tqdm(range(1, cfg.epochs + 1),
                         desc=f"  {model_name.upper()} [{ds.name}]",
                         unit="ep", leave=False, disable=not show_progress)
        for ep in epoch_bar:
            model.train()
            tl_sum, n = 0.0, 0
            for xb, yb in tr_dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward()
                opt.step()
                tl_sum += float(loss.item()) * len(xb)
                n      += len(xb)
            sched.step()
            train_loss = tl_sum / max(1, n)

            model.eval()
            vl_sum, vc, vt = 0.0, 0, 0
            with torch.no_grad():
                for xb, yb in va_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    out     = model(xb)
                    vl_sum += float(crit(out, yb).item()) * len(xb)
                    vc     += int((out.argmax(1) == yb).sum())
                    vt     += len(yb)
            val_loss = vl_sum / max(1, vt)
            val_acc  = vc     / max(1, vt)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = {k: v.detach().cpu().clone()
                                for k, v in model.state_dict().items()}

            epoch_bar.set_postfix(
                train=f"{train_loss:.3f}",
                val=f"{val_loss:.3f}",
                acc=f"{val_acc:.3f}",
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    preds = []
    with torch.no_grad():
        for xb, _ in te_dl:
            preds.extend(model(xb.to(device)).argmax(1).cpu().numpy().tolist())
    y_pred  = np.array(preds)
    metrics = classification_metrics(ds.y_test, y_pred, ds.class_names)
    perf    = time_cnn_inference(model, Xte_n, device)

    eff = {
        "params":                int(count_params(model)),
        "model_size_mb":         float(model_size_mb(model)),
        "forward_ms_per_window": perf["forward_ms"],
        "total_ms_per_window":   perf["total_ms"],
    }
    extras = {
        "snr_db":       snr_db,
        "best_val_acc": float(best_val_acc),
        "history":      history,
        "device":       device,
    }
    return RunResult(
        model=model_name, dataset=ds.name, task=ds.task,
        metrics=metrics, efficiency=eff,
        train_time_s=t_train.elapsed, extras=extras,
    )
