"""Dataset loaders for CWRU and PaderBorn with file-level splits.

Each loader returns a `Dataset` object with windowed signals already split into
train / val / test partitions at the *recording-file* level so windows from the
same source signal never appear in two splits at once.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Public dataset object
# ---------------------------------------------------------------------------
@dataclass
class Dataset:
    name: str
    fs: int
    window_size: int
    step: int
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    class_names: list[str]
    task: str = "multiclass"

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def summary(self) -> dict:
        return {
            "name": self.name,
            "fs": self.fs,
            "window_size": self.window_size,
            "step": self.step,
            "task": self.task,
            "n_classes": self.n_classes,
            "class_names": self.class_names,
            "n_train": int(len(self.X_train)),
            "n_val":   int(len(self.X_val)),
            "n_test":  int(len(self.X_test)),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _segment(sig: np.ndarray, win: int, step: int) -> np.ndarray:
    if len(sig) < win:
        return np.zeros((0, win), dtype=np.float32)
    starts = range(0, len(sig) - win + 1, step)
    return np.stack([sig[s:s + win] for s in starts]).astype(np.float32)


def _binarize(class_names: list[str], healthy_label: str) -> Callable[[int], int]:
    healthy_idx = class_names.index(healthy_label)
    return lambda y: 0 if y == healthy_idx else 1


# ---------------------------------------------------------------------------
# CWRU
# ---------------------------------------------------------------------------
# Drive-end (DE) channel @ 12 kHz.
# Split strategy: load 0 + 1 -> train, load 2 -> val, load 3 -> test (file-level).
CWRU_FILES = [
    # (filename, fault, size_in, load_hp)
    ("97.mat",  "Normal", 0,     0), ("98.mat",  "Normal", 0,     1),
    ("99.mat",  "Normal", 0,     2), ("100.mat", "Normal", 0,     3),
    ("105.mat", "IR",     0.007, 0), ("106.mat", "IR",     0.007, 1),
    ("107.mat", "IR",     0.007, 2), ("108.mat", "IR",     0.007, 3),
    ("169.mat", "IR",     0.014, 0), ("170.mat", "IR",     0.014, 1),
    ("171.mat", "IR",     0.014, 2), ("172.mat", "IR",     0.014, 3),
    ("209.mat", "IR",     0.021, 0), ("210.mat", "IR",     0.021, 1),
    ("211.mat", "IR",     0.021, 2), ("212.mat", "IR",     0.021, 3),
    ("118.mat", "B",      0.007, 0), ("119.mat", "B",      0.007, 1),
    ("120.mat", "B",      0.007, 2), ("121.mat", "B",      0.007, 3),
    ("185.mat", "B",      0.014, 0), ("186.mat", "B",      0.014, 1),
    ("187.mat", "B",      0.014, 2), ("188.mat", "B",      0.014, 3),
    ("222.mat", "B",      0.021, 0), ("223.mat", "B",      0.021, 1),
    ("224.mat", "B",      0.021, 2), ("225.mat", "B",      0.021, 3),
    ("130.mat", "OR",     0.007, 0), ("131.mat", "OR",     0.007, 1),
    ("132.mat", "OR",     0.007, 2), ("133.mat", "OR",     0.007, 3),
    ("197.mat", "OR",     0.014, 0), ("198.mat", "OR",     0.014, 1),
    ("199.mat", "OR",     0.014, 2), ("200.mat", "OR",     0.014, 3),
    ("234.mat", "OR",     0.021, 0), ("235.mat", "OR",     0.021, 1),
    ("236.mat", "OR",     0.021, 2), ("237.mat", "OR",     0.021, 3),
]


def _cwru_load_de(path: str) -> np.ndarray:
    mat = sio.loadmat(path)
    for k in mat:
        if k.endswith("_DE_time"):
            return mat[k].flatten().astype(np.float32)
    for k, v in mat.items():
        if not k.startswith("_") and isinstance(v, np.ndarray) and v.size > 10_000:
            return v.flatten().astype(np.float32)
    raise ValueError(f"Could not find DE channel in {path}")


def load_cwru(data_root: str, window: int, step: int, task: str = "multiclass") -> Dataset:
    cwru_dir = os.path.join(data_root, "cwru")
    if not os.path.isdir(cwru_dir):
        # Backward compatibility with the old layout:
        alt = os.path.join(data_root, "..", "cwru_data")
        if os.path.isdir(alt):
            cwru_dir = alt
        else:
            raise FileNotFoundError(
                f"CWRU directory not found. Expected at '{cwru_dir}'. "
                "Run `python scripts/download_cwru.py` first."
            )

    le = LabelEncoder().fit(["B", "IR", "Normal", "OR"])
    splits = {p: {"X": [], "y": []} for p in ("train", "val", "test")}

    for fname, fault, _size, load in CWRU_FILES:
        fpath = os.path.join(cwru_dir, fname)
        if not os.path.exists(fpath):
            continue
        sig = _cwru_load_de(fpath)
        wins = _segment(sig, window, step)
        if len(wins) == 0:
            continue
        if   load == 3: part = "test"
        elif load == 2: part = "val"
        else:           part = "train"
        splits[part]["X"].append(wins)
        splits[part]["y"].extend([le.transform([fault])[0]] * len(wins))

    def stack(p):
        if not splits[p]["X"]:
            return np.zeros((0, window), dtype=np.float32), np.zeros(0, dtype=np.int64)
        return np.vstack(splits[p]["X"]).astype(np.float32), np.array(splits[p]["y"], dtype=np.int64)

    X_tr, y_tr = stack("train"); X_va, y_va = stack("val"); X_te, y_te = stack("test")
    class_names = list(le.classes_)

    if task == "binary":
        bin_fn = _binarize(class_names, "Normal")
        y_tr = np.array([bin_fn(int(y)) for y in y_tr], dtype=np.int64)
        y_va = np.array([bin_fn(int(y)) for y in y_va], dtype=np.int64)
        y_te = np.array([bin_fn(int(y)) for y in y_te], dtype=np.int64)
        class_names = ["Normal", "Faulty"]

    return Dataset(
        name="cwru", fs=12_000, window_size=window, step=step,
        X_train=X_tr, y_train=y_tr,
        X_val=X_va,   y_val=y_va,
        X_test=X_te,  y_test=y_te,
        class_names=class_names, task=task,
    )


# ---------------------------------------------------------------------------
# PaderBorn
# ---------------------------------------------------------------------------
# Vibration channel @ 64 kHz. Each bearing ships as a single RAR archive
# containing one .mat per operating condition x run combination, named
# "<OC>_<bearing>_<run>.mat", e.g. "N15_M07_F10_KA01_1.mat".
# Operating conditions:
#   N15_M07_F10  -> 1500 rpm, 0.7 Nm, 1000 N  (most commonly used)
#   N09_M07_F10  -> 900 rpm,  0.7 Nm, 1000 N
#   N15_M01_F10  -> 1500 rpm, 0.1 Nm, 1000 N
#   N15_M07_F04  -> 1500 rpm, 0.7 Nm, 400 N
PADERBORN_BEARINGS = {
    "Healthy":  {"ids": ["K001", "K002", "K003"], "damage": "none"},
    "OR_Artif": {"ids": ["KA01", "KA03"],         "damage": "artificial"},
    "IR_Artif": {"ids": ["KI01", "KI03"],         "damage": "artificial"},
    "OR_Real":  {"ids": ["KA22", "KA30"],         "damage": "real"},
    "IR_Real":  {"ids": ["KI14", "KI16"],         "damage": "real"},
}
PADERBORN_DEFAULT_OC = "N15_M07_F10"
PADERBORN_RUNS = list(range(1, 6))   # use 5 of 20 runs to keep windows manageable


def _paderborn_load_signal(path: str, bearing_id: str) -> np.ndarray:
    mat = sio.loadmat(path, squeeze_me=False)
    try:
        # Standard PaderBorn struct layout.
        # filename without extension is also the field name.
        base = os.path.splitext(os.path.basename(path))[0]
        return mat[base]["Y"][0][0][0][6][2][0].flatten().astype(np.float32)
    except Exception:
        pass
    try:
        # Older convenience layout used in some redistributions
        return mat[bearing_id]["gs"][0][0].flatten().astype(np.float32)
    except Exception:
        pass
    for k, v in mat.items():
        if k.startswith("_"):
            continue
        if isinstance(v, np.ndarray) and v.size > 50_000:
            return v.flatten().astype(np.float32)
    raise ValueError(f"Could not find vibration signal in {path}")


def _paderborn_collect(data_root: str, damage_kind: str,
                        oc: str = PADERBORN_DEFAULT_OC):
    """Return a list of (signal, label, bearing_id, run) tuples."""
    pb_dir = os.path.join(data_root, "paderborn")
    if not os.path.isdir(pb_dir):
        alt = os.path.join(data_root, "..", "paderborn_data")
        if os.path.isdir(alt):
            pb_dir = alt
        else:
            raise FileNotFoundError(
                f"PaderBorn directory not found. Expected at '{pb_dir}'. "
                "Run `python scripts/download_paderborn.py` first."
            )

    items = []
    for label, info in PADERBORN_BEARINGS.items():
        if info["damage"] not in ("none", damage_kind):
            continue
        if label == "Healthy":
            class3 = "Healthy"
        elif "OR" in label:
            class3 = "OR"
        else:
            class3 = "IR"
        for bid in info["ids"]:
            bear_dir = os.path.join(pb_dir, bid)
            if not os.path.isdir(bear_dir):
                continue
            for run in PADERBORN_RUNS:
                fname = f"{oc}_{bid}_{run}.mat"
                fpath = os.path.join(bear_dir, fname)
                if not os.path.exists(fpath):
                    continue
                try:
                    sig = _paderborn_load_signal(fpath, bid)
                except Exception as e:
                    tqdm_write = print
                    tqdm_write(f"  [PaderBorn] skipping {fpath}: {e}")
                    continue
                items.append((sig, class3, bid, run))
    return items, pb_dir


def load_paderborn(data_root: str, window: int, step: int,
                   damage_kind: str, task: str = "multiclass",
                   split_mode: str = "run") -> Dataset:
    """Load PaderBorn with one of two file-level split strategies.

    split_mode = 'run'           : runs 1-3 train, run 4 val, run 5 test (lenient)
    split_mode = 'cross-bearing' : leave-one-bearing-per-class out (harder).
    """
    items, pb_dir = _paderborn_collect(data_root, damage_kind)
    if not items:
        raise RuntimeError(
            f"No PaderBorn ({damage_kind}) recordings found in '{pb_dir}'. "
            "Run `python scripts/download_paderborn.py` first."
        )

    le = LabelEncoder().fit(["Healthy", "IR", "OR"])
    splits = {p: {"X": [], "y": []} for p in ("train", "val", "test")}

    if split_mode == "run":
        train_runs, val_runs, test_runs = {1, 2, 3}, {4}, {5}

        def assign(label: str, bid: str, run: int) -> str | None:
            if   run in train_runs: return "train"
            elif run in val_runs:   return "val"
            elif run in test_runs:  return "test"
            return None

    elif split_mode == "cross-bearing":
        # Bearings held out for the test partition (one per class).
        # Train/val use the *other* bearings only; val just splits run 5 from
        # the training bearings to allow model selection without test-set leakage.
        if damage_kind == "artificial":
            test_bearings = {"K003", "KA03", "KI03"}
        else:  # 'real'
            test_bearings = {"K003", "KA30", "KI16"}

        def assign(label: str, bid: str, run: int) -> str | None:
            if bid in test_bearings:
                return "test"
            if run == 5:
                return "val"
            return "train"
    else:
        raise ValueError(f"Unknown PaderBorn split_mode '{split_mode}'")

    for sig, label, bid, run in items:
        wins = _segment(sig, window, step)
        if len(wins) == 0:
            continue
        part = assign(label, bid, run)
        if part is None:
            continue
        splits[part]["X"].append(wins)
        splits[part]["y"].extend([le.transform([label])[0]] * len(wins))

    def stack(p):
        if not splits[p]["X"]:
            return np.zeros((0, window), dtype=np.float32), np.zeros(0, dtype=np.int64)
        return np.vstack(splits[p]["X"]).astype(np.float32), np.array(splits[p]["y"], dtype=np.int64)

    X_tr, y_tr = stack("train"); X_va, y_va = stack("val"); X_te, y_te = stack("test")
    class_names = list(le.classes_)

    if task == "binary":
        bin_fn = _binarize(class_names, "Healthy")
        y_tr = np.array([bin_fn(int(y)) for y in y_tr], dtype=np.int64)
        y_va = np.array([bin_fn(int(y)) for y in y_va], dtype=np.int64)
        y_te = np.array([bin_fn(int(y)) for y in y_te], dtype=np.int64)
        class_names = ["Healthy", "Faulty"]

    suffix = "" if split_mode == "run" else "-xb"
    return Dataset(
        name=f"paderborn-{damage_kind}{suffix}",
        fs=64_000, window_size=window, step=step,
        X_train=X_tr, y_train=y_tr,
        X_val=X_va,   y_val=y_va,
        X_test=X_te,  y_test=y_te,
        class_names=class_names, task=task,
    )


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------
def load_dataset(name: str, data_root: str, window: int, step: int,
                 task: str = "multiclass") -> Dataset:
    if name == "cwru":
        return load_cwru(data_root, window, step, task)
    if name == "paderborn-artificial":
        return load_paderborn(data_root, window, step, "artificial", task,
                              split_mode="run")
    if name == "paderborn-real":
        return load_paderborn(data_root, window, step, "real", task,
                              split_mode="run")
    if name == "paderborn-artificial-xb":
        return load_paderborn(data_root, window, step, "artificial", task,
                              split_mode="cross-bearing")
    if name == "paderborn-real-xb":
        return load_paderborn(data_root, window, step, "real", task,
                              split_mode="cross-bearing")
    raise ValueError(f"Unknown dataset '{name}'")


def normalize_per_window(X: np.ndarray) -> np.ndarray:
    if len(X) == 0:
        return X
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-8
    return ((X - mu) / sd).astype(np.float32)
