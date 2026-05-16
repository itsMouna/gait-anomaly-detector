"""
Feature extraction for gait signal windows.

Seven hand-crafted features per window cover both time-domain statistics
and frequency-domain energy, giving classical models a compact but
clinically interpretable representation.
"""

import numpy as np
from scipy.fft import fft

FEATURE_NAMES = [
    "mean_amplitude",
    "std_amplitude",
    "mean_energy",
    "max_amplitude",
    "min_amplitude",
    "mean_fft_magnitude",
    "std_fft_magnitude",
]


def magnitude(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Euclidean magnitude of triaxial acceleration: sqrt(x²+y²+z²)."""
    return np.sqrt(x**2 + y**2 + z**2)


def _raw_features(windows: np.ndarray) -> np.ndarray:
    """Return un-normalised (n, 7) feature matrix."""
    features = []
    for w in windows:
        fft_vals = np.abs(fft(w))[:len(w) // 2]
        features.append([
            np.mean(w),
            np.std(w),
            np.mean(w ** 2),
            np.max(w),
            np.min(w),
            np.mean(fft_vals),
            np.std(fft_vals),
        ])
    return np.array(features)


def extract_features(windows: np.ndarray,
                     fit_scaler: np.ndarray = None) -> np.ndarray:
    """
    Extract and normalise a 7-dimensional feature vector per window.

    Parameters
    ----------
    windows     : np.ndarray, shape (n, window_len)
    fit_scaler  : np.ndarray, shape (m, 7), optional
        Pre-computed feature matrix whose mean/std will be used for
        normalisation.  Pass the training feature matrix here when
        transforming test data so there is no data leakage.
        If None, normalise using the current batch's own statistics.

    Returns
    -------
    X : np.ndarray, shape (n, 7) — normalised feature matrix
    """
    X   = _raw_features(windows)
    ref = fit_scaler if fit_scaler is not None else X
    # ref must be (m, 7) — guard against accidentally passing raw windows
    if ref.ndim != 2 or ref.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"fit_scaler must be a (m, {len(FEATURE_NAMES)}) feature matrix, "
            f"got shape {ref.shape}. "
            "Pass extract_features(train_windows) as fit_scaler, "
            "not the raw train_windows array."
        )
    return (X - np.mean(ref, axis=0)) / (np.std(ref, axis=0) + 1e-8)
