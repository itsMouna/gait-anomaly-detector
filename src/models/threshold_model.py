import numpy as np


class ThresholdModel:
    """
    Simple statistical anomaly detector.

    Flags a window as anomalous if its std or energy feature
    exceeds mean + k * std of the training distribution.
    """

    def __init__(self, k: float = 2.0):
        self.k = k
        self.std_th = None
        self.energy_th = None

    def fit(self, X: np.ndarray):
        """Compute thresholds from training features."""
        self.std_th = np.mean(X[:, 1]) + self.k * np.std(X[:, 1])
        self.energy_th = np.mean(X[:, 2]) + self.k * np.std(X[:, 2])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return 1 for anomaly, 0 for normal."""
        std = X[:, 1]
        energy = X[:, 2]
        return ((std > self.std_th) | (energy > self.energy_th)).astype(int)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return a continuous anomaly score (max of normalised features)."""
        std_score = X[:, 1] / (self.std_th + 1e-8)
        energy_score = X[:, 2] / (self.energy_th + 1e-8)
        return np.maximum(std_score, energy_score)
