import numpy as np


def create_windows(signal, window_size=128, step=64):
    windows = []

    for i in range(0, len(signal) - window_size, step):
        windows.append(signal[i:i + window_size])

    return np.array(windows)