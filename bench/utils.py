"""Utility helpers: seeding, timing, IO."""

from __future__ import annotations
import json
import os
import random
import sys
import time
import warnings
import numpy as np


def silence_noisy_warnings() -> None:
    """Suppress numerical / convergence warnings that flood the log without
    affecting correctness (overflow / divide-by-zero in sklearn linear models
    when features have heavy-tailed distributions, etc.)."""
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
    warnings.filterwarnings("ignore", category=UserWarning,
                             module="sklearn")


def configure_stdout() -> None:
    """Force line-buffered stdout/stderr so progress bars appear in real time
    even when output is piped to a file or terminal recording.
    """
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


class Timer:
    def __init__(self):
        self.t0 = None
        self.elapsed = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def banner(text: str, width: int = 78, char: str = "=") -> str:
    return f"\n{char * width}\n {text}\n{char * width}"


def device_str(spec: str = "auto") -> str:
    if spec != "auto":
        return spec
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
