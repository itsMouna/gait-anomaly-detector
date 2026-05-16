import numpy as np


class ThresholdDetector:
    def __init__(self, std_threshold, energy_threshold):
        self.std_threshold = std_threshold
        self.energy_threshold = energy_threshold

    def predict(self, features):
        predictions = []

        for feature_vector in features:
            _, std, energy, _ = feature_vector

            if std > self.std_threshold or energy > self.energy_threshold:
                predictions.append(1)
            else:
                predictions.append(0)

        return np.array(predictions)