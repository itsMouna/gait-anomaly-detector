"""
Data loading for the UCI HAR Dataset.

Provides both a simple random split (load_walking_only) and a
subject-aware split (load_by_subject) for cross-subject generalisation
experiments.
"""

import numpy as np
from config import X_PATH, Y_PATH, Z_PATH, LABEL_PATH, SUBJECT_PATH


# ── Raw loaders ───────────────────────────────────────────────────────

def load_signals():
    """Load raw body acceleration signals (X, Y, Z axes)."""
    x = np.loadtxt(X_PATH)
    y = np.loadtxt(Y_PATH)
    z = np.loadtxt(Z_PATH)
    return x, y, z


def load_labels():
    """Load activity labels (1 = WALKING … 6 = LAYING)."""
    return np.loadtxt(LABEL_PATH).astype(int)


def load_subjects():
    """Load per-window subject IDs (1–30)."""
    return np.loadtxt(SUBJECT_PATH).astype(int)


# ── Walking-only views ────────────────────────────────────────────────

def load_walking_only():
    """Return (x, y, z) windows for WALKING windows only (label == 1)."""
    x, y, z = load_signals()
    labels   = load_labels()
    mask     = labels == 1
    return x[mask], y[mask], z[mask]


def load_walking_with_subjects():
    """Return (x, y, z, subject_ids) for WALKING windows only."""
    x, y, z = load_signals()
    labels   = load_labels()
    subjects = load_subjects()
    mask     = labels == 1
    return x[mask], y[mask], z[mask], subjects[mask]


# ── Cross-subject split ───────────────────────────────────────────────

def load_by_subject(train_subjects: list, test_subjects: list):
    """
    Split WALKING windows by subject ID.

    Parameters
    ----------
    train_subjects : list[int]
        Subject IDs whose data goes into the training set.
    test_subjects  : list[int]
        Subject IDs whose data forms the held-out test set.

    Returns
    -------
    train : tuple (x, y, z)
    test  : tuple (x, y, z)
    """
    x, y, z, subjects = load_walking_with_subjects()

    train_mask = np.isin(subjects, train_subjects)
    test_mask  = np.isin(subjects, test_subjects)

    train = (x[train_mask], y[train_mask], z[train_mask])
    test  = (x[test_mask],  y[test_mask],  z[test_mask])

    return train, test
