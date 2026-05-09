"""Time-domain and frequency-domain feature extraction for vibration windows."""

from __future__ import annotations
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.stats import skew, kurtosis

FEATURE_NAMES = [
    # time domain (11)
    "mean", "std", "rms",
    "peak", "peak2peak",
    "skewness", "kurtosis",
    "crest_factor", "shape_factor", "impulse_factor", "clearance_factor",
    # frequency domain (9)
    "fft_mean", "fft_std", "fft_max",
    "dom_freq", "spec_centroid", "spec_entropy",
    "band1_e", "band2_e", "band3_e",
]


def _safe(x: float) -> float:
    return float(x) if np.isfinite(x) else 0.0


def extract_window_features(w: np.ndarray, fs: int) -> np.ndarray:
    out = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    if len(w) == 0:
        return out

    abs_w = np.abs(w)
    rms   = float(np.sqrt(np.mean(w ** 2)))
    peak  = float(np.max(abs_w))
    mean_abs   = float(np.mean(abs_w)) + 1e-12
    sqrt_mean  = float(np.mean(np.sqrt(abs_w))) ** 2 + 1e-12
    rms_safe   = rms + 1e-12

    out[0]  = float(np.mean(w))
    out[1]  = float(np.std(w))
    out[2]  = rms
    out[3]  = peak
    out[4]  = float(np.ptp(w))
    out[5]  = _safe(skew(w))
    out[6]  = _safe(kurtosis(w))
    out[7]  = peak / rms_safe
    out[8]  = rms  / mean_abs
    out[9]  = peak / mean_abs
    out[10] = peak / sqrt_mean

    mag = np.abs(rfft(w))
    freqs = rfftfreq(len(w), d=1.0 / fs)
    if mag.sum() <= 0:
        return out

    out[11] = float(mag.mean())
    out[12] = float(mag.std())
    out[13] = float(mag.max())

    out[14] = float(freqs[int(np.argmax(mag))])
    p = mag / (mag.sum() + 1e-12)
    out[15] = float(np.sum(freqs * p))
    out[16] = float(-np.sum(p * np.log(p + 1e-12)))

    n = len(mag)
    third = n // 3
    total_e = float(np.sum(mag ** 2)) + 1e-12
    out[17] = float(np.sum(mag[:third] ** 2)) / total_e
    out[18] = float(np.sum(mag[third:2 * third] ** 2)) / total_e
    out[19] = float(np.sum(mag[2 * third:] ** 2)) / total_e

    return out


def extract_features(X: np.ndarray, fs: int,
                     progress: bool = False, desc: str = "features") -> np.ndarray:
    """Loop over windows and stack feature vectors. Optionally show a tqdm bar."""
    if len(X) == 0:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    out = np.zeros((len(X), len(FEATURE_NAMES)), dtype=np.float32)
    iterable = range(len(X))
    if progress and len(X) >= 1000:
        from tqdm.auto import tqdm
        iterable = tqdm(iterable, desc=f"  {desc}", unit="win", leave=False)
    for i in iterable:
        out[i] = extract_window_features(X[i], fs)
    return out
