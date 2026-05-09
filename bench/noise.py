"""Gaussian-noise injection at a target signal-to-noise ratio (dB)."""

from __future__ import annotations
import numpy as np


def add_awgn(X: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    """Add additive white Gaussian noise to each row of X to reach `snr_db`.

    Power of the signal is computed per-row, so the noise level adapts to each
    window's amplitude.
    """
    if X.size == 0:
        return X
    rng = np.random.default_rng(seed)
    sig_power = np.mean(X ** 2, axis=1, keepdims=True) + 1e-12
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise = rng.normal(0.0, np.sqrt(noise_power), size=X.shape).astype(X.dtype)
    return (X + noise).astype(np.float32)
