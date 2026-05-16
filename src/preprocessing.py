"""
Anomaly injection with clinically motivated perturbation types.

Each anomaly type maps to a real gait pathology:

  progressive_fatigue    — linear amplitude drift; models muscle fatigue
                           or progressive neurological asymmetry.
                           Ref: Maki (1997) J Gerontol, stride variability
                           as a fatigue marker.

  sensor_noise           — additive Gaussian noise; models IMU placement
                           error, loose attachment, or skin-electrode
                           impedance drift in wearable sensors.
                           Ref: Faber et al. (2018) Sensors.

  compensatory_asymmetry — global amplitude attenuation; models reduced
                           push-off on one limb after orthopaedic injury
                           or stroke-induced hemiparesis.
                           Ref: Patterson et al. (2010) Gait & Posture.

  near_fall_event        — localised high-amplitude spike; models the
                           sudden deceleration impulse recorded during a
                           stumble or near-fall.
                           Ref: Bourke et al. (2007) Med Eng Phys.
"""

import numpy as np
from config import ANOMALY_THRESHOLDS


def inject_anomalies(signal: np.ndarray, rate: float = 0.15,
                     seed: int = 42):
    """
    Inject clinically motivated anomalies into a window array.

    Parameters
    ----------
    signal : np.ndarray, shape (n_windows, window_len)
    rate   : float  — fraction of windows to corrupt
    seed   : int    — RNG seed for reproducibility

    Returns
    -------
    corrupted  : np.ndarray  — copy of signal with injected anomalies
    idx        : np.ndarray  — indices of corrupted windows
    type_labels: np.ndarray  — clinical type string per corrupted window
    """
    rng       = np.random.default_rng(seed)
    corrupted = signal.copy()
    n         = len(signal)
    k         = int(n * rate)
    idx       = rng.choice(n, k, replace=False)

    thresholds  = ANOMALY_THRESHOLDS
    type_labels = []

    for i in idx:
        mode       = rng.random()
        base       = signal[i].copy()
        window_len = len(base)

        if mode < thresholds["progressive_fatigue"]:
            # Gradual drift — fatigue / neurological asymmetry
            corrupted[i] = base + np.linspace(0, 0.8, window_len)
            type_labels.append("progressive_fatigue")

        elif mode < thresholds["sensor_noise"]:
            # Additive Gaussian noise — sensor placement error
            corrupted[i] = base + rng.normal(0, 0.5, window_len)
            type_labels.append("sensor_noise")

        elif mode < thresholds["compensatory_asymmetry"]:
            # Amplitude attenuation — compensatory gait / hemiparesis
            corrupted[i] = base * 0.6
            type_labels.append("compensatory_asymmetry")

        else:
            # Localised spike — near-fall deceleration impulse
            spike_pos    = rng.integers(0, window_len - 5)
            corrupted[i] = base.copy()
            corrupted[i][spike_pos:spike_pos + 5] += 2.0
            type_labels.append("near_fall_event")

    return corrupted, idx, np.array(type_labels)
