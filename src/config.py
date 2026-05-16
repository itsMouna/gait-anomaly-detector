"""
Central configuration for the Gait Anomaly Detector.

All paths, hyperparameters, and clinical anomaly definitions live here.
Override any value by passing --config your_config.yaml to main.py.
"""

from pathlib import Path
import yaml

SRC_DIR  = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent

# ── Data paths ────────────────────────────────────────────────────────
DATA_DIR   = BASE_DIR / "data" / "UCI HAR Dataset"

X_PATH       = DATA_DIR / "train/Inertial Signals/body_acc_x_train.txt"
Y_PATH       = DATA_DIR / "train/Inertial Signals/body_acc_y_train.txt"
Z_PATH       = DATA_DIR / "train/Inertial Signals/body_acc_z_train.txt"
LABEL_PATH   = DATA_DIR / "train/y_train.txt"
SUBJECT_PATH = DATA_DIR / "train/subject_train.txt"

# ── Experiment settings ───────────────────────────────────────────────
ANOMALY_RATE    = 0.15
RANDOM_SEED     = 42
LSTM_EPOCHS     = 15
LSTM_BATCH_SIZE = 32

TRAIN_SUBJECTS = list(range(1, 22))   # subjects 1–21
TEST_SUBJECTS  = list(range(22, 31))  # subjects 22–30

# ── Clinical anomaly taxonomy ─────────────────────────────────────────
ANOMALY_TYPES = {
    "progressive_fatigue":    "Progressive fatigue drift",
    "sensor_noise":           "Sensor / placement noise",
    "compensatory_asymmetry": "Compensatory gait asymmetry",
    "near_fall_event":        "Near-fall impulsive event",
}

ANOMALY_THRESHOLDS = {
    "progressive_fatigue":    0.25,
    "sensor_noise":           0.50,
    "compensatory_asymmetry": 0.75,
    "near_fall_event":        1.00,
}

# ── Output directories ────────────────────────────────────────────────
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "figures").mkdir(exist_ok=True)
(RESULTS_DIR / "shap").mkdir(exist_ok=True)
(RESULTS_DIR / "generalisation").mkdir(exist_ok=True)


def load_yaml_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
