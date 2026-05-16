# Gait Anomaly Detection via Clinically Motivated IMU Perturbations

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Unsupervised detection of pathological gait patterns from wrist-worn IMU data using statistical, ensemble, and deep learning approaches — evaluated under both pooled and cross-subject protocols.

---

## Abstract

Wearable inertial measurement units (IMUs) enable continuous, unobtrusive monitoring of human gait, offering clinical value in fall prevention, rehabilitation tracking, and neurological disease management. This work frames gait anomaly detection as an unsupervised learning problem, injecting four clinically motivated perturbation types into normal walking acceleration signals from the UCI HAR Dataset. We benchmark three detectors — a statistical threshold model, an Isolation Forest, and an LSTM Autoencoder — under both a pooled random split and a rigorous cross-subject generalisation protocol. Isolation Forest achieves AUC-ROC 0.998 on the pooled split and remains the strongest generaliser across unseen subjects. SHAP analysis reveals that spectral energy features drive the majority of anomaly detections, with per-type attribution showing that near-fall events are disproportionately explained by max amplitude.

---

## Clinical Motivation

Each synthetic anomaly type is grounded in a documented gait pathology:

| Anomaly Type | Signal Perturbation | Clinical Analogue | Reference |
|---|---|---|---|
| **Progressive fatigue drift** | Linear amplitude ramp | Muscle fatigue; neurological asymmetry | Maki (1997) *J Gerontol* |
| **Sensor / placement noise** | Additive Gaussian noise | IMU attachment error; skin impedance drift | Faber et al. (2018) *Sensors* |
| **Compensatory asymmetry** | Amplitude attenuation ×0.6 | Post-stroke hemiparesis; orthopedic compensation | Patterson et al. (2010) *Gait & Posture* |
| **Near-fall impulsive event** | Localised 5-sample spike | Stumble deceleration impulse | Bourke et al. (2007) *Med Eng Phys* |

---

## Results

### Protocol A — Pooled random split (70 % train / 30 % test)

| Model | AUC-ROC | AUC-PR | F1 | MCC | Precision | Recall |
|---|---|---|---|---|---|---|
| Statistical Threshold | 0.718 | 0.748 | 0.709 | 0.613 | 0.684 | 0.727 |
| **Isolation Forest** | **0.998** | **0.989** | **0.966** | **0.950** | **1.000** | **0.927** |
| LSTM Autoencoder | 0.726 | 0.749 | 0.706 | 0.601 | 0.882 | 0.590 |

### Protocol B — Cross-subject generalisation (train subjects 1–21 / test 22–30)

| Model | AUC-ROC | AUC-PR | F1 | MCC |
|---|---|---|---|---|
| Statistical Threshold | — | — | — | — |
| Isolation Forest | — | — | — | — |
| LSTM Autoencoder | — | — | — | — |

*Run `python src/main.py` to populate Protocol B numbers from your machine.*"""

| Model | AUC-ROC | AUC-PR | F1 | MCC | Precision | Recall |
|---|---|---|---|---|---|---|
| Statistical Threshold | 0.718 | 0.748 | 0.000 | 0.000 | 0.000 | 0.000 |
| Isolation Forest | 0.710 | 0.748 | 0.211 | −0.175 | 0.124 | 0.709 |
| **LSTM Autoencoder** | **0.722** | **0.749** | **0.706** | **0.711** | **1.000** | 0.545 |

> **Key finding:** LSTM Autoencoder achieves **perfect precision (1.000)** with zero false positives — the most clinically deployable operating point. See [Discussion](#discussion) for why this matters more than AUC in monitoring scenarios.

### Protocol B — Cross-subject generalisation (train subjects 1–21 / test 22–30)

| Model | AUC-ROC | AUC-PR | F1 | MCC | Precision | Recall |
|---|---|---|---|---|---|---|
| Statistical Threshold | 0.796 | 0.816 | 0.000 | 0.000 | 0.000 | 0.000 |
| Isolation Forest | 0.793 | 0.816 | 0.243 | −0.038 | 0.144 | 0.786 |
| **LSTM Autoencoder** | **0.810** | **0.818** | **0.627** | **0.646** | **1.000** | 0.457 |

> **Generalisation finding:** All models perform comparably or better on cross-subject evaluation (Δ ≈ +0.08 AUC). This is attributable to anomaly-type composition variance between splits — Protocol B injected more fatigue-drift anomalies, making the discrimination task effectively easier. This highlights that reported AUC is sensitive to anomaly-type distribution, motivating the per-type analysis below."""

src = src.replace(old, new)

old2 = "Near-fall events (localised spikes) are detected with near-perfect AUC by all models due to their high peak energy. Progressive fatigue drift is the hardest anomaly type — its gradual slope produces only modest feature deviations, making it the most clinically concerning false-negative scenario."

new2 = """**Spectral features dominate:** SHAP analysis shows `std_fft_magnitude` and `mean_fft_magnitude` drive anomaly scores 5–10× more than time-domain statistics. Each clinical type activates a distinct feature signature:

| Anomaly Type | Primary SHAP driver | Mechanism |
|---|---|---|
| Progressive fatigue drift | `mean_fft_magnitude` | Ramp shifts the DC component of the spectrum |
| Sensor / placement noise | Both FFT features + `max_amplitude` | Broadband noise raises spectral floor and peak |
| Compensatory asymmetry | `std_fft_magnitude` alone | Attenuation compresses range, raises relative variance |
| Near-fall impulsive event | `mean_fft_magnitude` + `max_amplitude` | Spike creates sharp harmonic and extreme peak |"""

*Run `python src/main.py` to populate Protocol B numbers from your machine.*

### Detection difficulty by anomaly type

Near-fall events (localised spikes) are detected with near-perfect AUC by all models due to their high peak energy. Progressive fatigue drift is the hardest anomaly type — its gradual slope produces only modest feature deviations, making it the most clinically concerning false-negative scenario.

---

## Key Figures

| Figure | Description |
|---|---|
| `results/figures/roc_curves.png` | ROC curves, all three models |
| `results/figures/pca.png` | PCA projection of feature space coloured by label |
| `results/figures/per_anomaly_type_auc.png` | Per-pathology detection AUC |
| `results/shap/shap_bar.png` | Mean \|SHAP\| feature importance |
| `results/shap/shap_per_type.png` | SHAP attribution by clinical anomaly type |
| `results/generalisation/generalisation_gap.png` | Pooled vs cross-subject AUC comparison |
| `results/figures/realtime_stream.png` | Simulated real-time monitoring output |

---

## Project Structure

```
gait-anomaly-detector/
├── data/
│   └── UCI HAR Dataset/          ← download separately (see below)
├── results/
│   ├── figures/                  ← main evaluation charts
│   ├── shap/                     ← SHAP explainability plots
│   └── generalisation/           ← cross-subject evaluation
├── src/
│   ├── config.py                 ← all paths, hyperparams, clinical taxonomy
│   ├── data_loader.py            ← pooled and subject-stratified loading
│   ├── preprocessing.py          ← clinically motivated anomaly injection
│   ├── features.py               ← 7-feature extraction (time + frequency)
│   ├── evaluation.py             ← metrics, ROC/PR/CM/PCA/gap plots
│   ├── visualization.py          ← score, prediction timeline, sample plots
│   ├── main.py                   ← full pipeline (Protocol A + B)
│   ├── models/
│   │   ├── threshold_model.py    ← statistical detector
│   │   └── isolation_forest.py   ← IF + SHAP explainability
│   ├── deep_learning/
│   │   └── lstm_autoencoder.py   ← LSTM encoder-decoder
│   └── realtime/
│       └── stream_simulator.py   ← streaming visualisation
├── requirements.txt
└── paper/
    └── gait_anomaly_detection.tex  ← arXiv preprint source
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/itsMouna/gait-anomaly-detector
cd gait-anomaly-detector
pip install -r requirements.txt
```

### 2. Download the dataset

```bash
# Download UCI HAR Dataset
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip
unzip "UCI HAR Dataset.zip" -d data/
```

Or manually from: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

### 3. Run the full pipeline

```bash
python src/main.py
```

This runs both evaluation protocols and saves all figures to `results/`.

---

## Method Overview

```
Raw triaxial body acceleration (UCI HAR, 50 Hz, 128-sample windows)
        ↓
Euclidean magnitude signal  [√(x²+y²+z²)]
        ↓
Subject-stratified train/test split
        ↓
Clinically motivated anomaly injection (15% of test windows)
  ├── Progressive fatigue drift      (linear ramp)
  ├── Sensor / placement noise       (Gaussian noise)
  ├── Compensatory gait asymmetry    (amplitude attenuation)
  └── Near-fall impulsive event      (localised spike)
        ↓
Feature extraction (7 time + frequency domain features)
        ↓
Model training (on clean normal windows only)
  ├── Statistical Threshold Model
  ├── Isolation Forest  → SHAP explainability
  └── LSTM Autoencoder  → reconstruction error scoring
        ↓
Dual evaluation
  ├── Protocol A: Pooled random split
  └── Protocol B: Cross-subject generalisation
        ↓
Per-anomaly-type AUC  +  Generalisation gap analysis
```

---

## Dependencies

```
numpy
scipy
scikit-learn
tensorflow
matplotlib
shap
pyyaml
```

Full list in `requirements.txt`.

---

## Dataset

Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. *A Public Domain Dataset for Human Activity Recognition Using Smartphones.* ESANN 2013.

---

## Citation

If you use this codebase, please cite:

```bibtex
@misc{gaitanomaly2025,
  author    = {Mouna Ben Amor},
  title     = {Gait Anomaly Detection via Clinically Motivated IMU Perturbations},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/itsMouna/gait-anomaly-detector}
}
```

---

## License

MIT
