"""
Gait Anomaly Detection — Full Pipeline
=======================================

Two evaluation protocols:
  A) Pooled split      — random 70/30 train/test (fast baseline)
  B) Cross-subject     — train on subjects 1–21, test on 22–30
                         (tests real-world generalisation)

Three detectors compared:
  1. Statistical Threshold Model
  2. Isolation Forest  (+ SHAP explainability)
  3. LSTM Autoencoder  (deep learning)

Four clinically motivated anomaly types injected:
  - Progressive fatigue drift
  - Sensor / placement noise
  - Compensatory gait asymmetry
  - Near-fall impulsive event

Run from the project root:
    python src/main.py
"""

import os
import sys
import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from config import (
    RESULTS_DIR, TRAIN_SUBJECTS, TEST_SUBJECTS,
    ANOMALY_RATE, RANDOM_SEED, LSTM_EPOCHS, LSTM_BATCH_SIZE,
    ANOMALY_TYPES,
)
from data_loader import load_walking_only, load_by_subject
from preprocessing import inject_anomalies
from features import magnitude, extract_features

from models.threshold_model import ThresholdModel
from models.isolation_forest import IsolationForestModel
from deep_learning.lstm_autoencoder import LSTMAutoencoder
from realtime.stream_simulator import stream_signal

from evaluation import (
    evaluate, print_results_table,
    plot_roc_curves, plot_pr_curves,
    plot_confusion_matrices, plot_pca,
    plot_generalisation_gap,
    plot_per_anomaly_type_auc,
)
from visualization import (
    plot_anomaly_scores, plot_sample_windows, plot_predictions,
)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def run_models(train_signal, test_signal, label_prefix="",
               run_lstm=True):
    """
    Inject anomalies, extract features, train & evaluate all models.

    Returns
    -------
    results     : dict  {model_name: metric_dict}
    all_scores  : dict  {model_name: score_array}
    all_preds   : dict  {model_name: pred_array}
    y_test      : np.ndarray binary labels
    corrupted   : np.ndarray corrupted test windows
    anomaly_idx : np.ndarray indices of injected anomalies
    type_labels : np.ndarray clinical type per injected anomaly
    """
    # 1. Inject clinically motivated anomalies
    corrupted, anomaly_idx, type_labels = inject_anomalies(
        test_signal, rate=ANOMALY_RATE, seed=RANDOM_SEED
    )
    y_test = np.zeros(len(corrupted), dtype=int)
    y_test[anomaly_idx] = 1

    n_anom = len(anomaly_idx)
    print(f"      {n_anom} anomalies injected ({n_anom/len(y_test)*100:.1f}%)")
    for atype, label in ANOMALY_TYPES.items():
        n = (type_labels == atype).sum()
        print(f"        • {label}: {n}")

    # 2. Feature extraction — test normalised with train statistics
    X_train_feat = extract_features(train_signal)
    X_test_feat = extract_features(corrupted,
                                   fit_scaler=extract_features(train_signal))

    # 3. Threshold model
    print(f"\n  [{label_prefix}] Threshold Model")
    tm = ThresholdModel(k=2.0)
    tm.fit(X_train_feat)
    pred_t  = tm.predict(X_test_feat)
    score_t = tm.score(X_test_feat)
    res_t   = evaluate(y_test, pred_t, score_t, "Threshold Model")

    # 4. Isolation Forest
    print(f"\n  [{label_prefix}] Isolation Forest")
    iso = IsolationForestModel(contamination=ANOMALY_RATE)
    iso.fit(X_train_feat)
    pred_i  = iso.predict(X_test_feat)
    score_i = iso.score(X_test_feat)
    res_i   = evaluate(y_test, pred_i, score_i, "Isolation Forest")

    # 5. LSTM Autoencoder
    pred_dl = np.zeros_like(y_test)
    score_dl = np.zeros(len(y_test))
    res_dl   = {"f1": 0, "mcc": 0, "precision": 0, "recall": 0,
                "auc_roc": 0, "auc_pr": 0,
                "tn": 0, "fp": 0, "fn": 0, "tp": 0}

    if run_lstm:
        print(f"\n  [{label_prefix}] LSTM Autoencoder")
        X_tr_dl  = train_signal.reshape(-1, train_signal.shape[1], 1)
        X_te_dl  = corrupted.reshape(-1, corrupted.shape[1], 1)
        lstm_ae  = LSTMAutoencoder(timesteps=128, features=1)
        lstm_ae.fit(X_tr_dl, epochs=LSTM_EPOCHS,
                    batch_size=LSTM_BATCH_SIZE)
        score_dl = lstm_ae.score(X_te_dl)
        thr_dl   = np.mean(score_dl) + 2 * np.std(score_dl)
        pred_dl  = (score_dl > thr_dl).astype(int)
        res_dl   = evaluate(y_test, pred_dl, score_dl, "LSTM Autoencoder")

    results    = {"Threshold": res_t,
                  "Isolation Forest": res_i,
                  "LSTM Autoencoder": res_dl}
    all_scores = {"Threshold": score_t,
                  "Isolation Forest": score_i,
                  "LSTM Autoencoder": score_dl}
    all_preds  = {"Threshold": pred_t,
                  "Isolation Forest": pred_i,
                  "LSTM Autoencoder": pred_dl}

    return (results, all_scores, all_preds,
            y_test, corrupted, anomaly_idx, type_labels, iso, X_test_feat)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 62)
    print("  Gait Anomaly Detection — Research Pipeline")
    print("=" * 62)

    # ── PROTOCOL A: Pooled random split ──────────────────────────────
    print("\n\n══ PROTOCOL A: Pooled random 70/30 split ══")

    print("\n[A1] Loading WALKING signals...")
    x, y, z  = load_walking_only()
    signal   = magnitude(x, y, z)
    split    = int(0.7 * len(signal))
    train_p  = signal[:split]
    test_p   = signal[split:]
    print(f"     {len(signal)} windows | train {len(train_p)} | test {len(test_p)}")

    print("\n[A2] Injecting anomalies & training models...")
    (res_pool, scores_pool, preds_pool,
     y_pool, corrupt_pool, idx_pool, types_pool,
     iso_pool, X_feat_pool) = run_models(
        train_p, test_p, label_prefix="A", run_lstm=True
    )

    print("\n[A3] Saving pooled-split figures...")

    plot_sample_windows(
        corrupt_pool, y_pool, n=4,
        save_path=str(RESULTS_DIR / "figures/sample_windows.png"),
    )
    plot_pca(
        X_feat_pool, y_pool,
        save_path=str(RESULTS_DIR / "figures/pca.png"),
    )
    plot_roc_curves(
        y_pool, scores_pool,
        title="ROC Curves — Pooled split",
        save_path=str(RESULTS_DIR / "figures/roc_curves.png"),
    )
    plot_pr_curves(
        y_pool, scores_pool,
        title="Precision-Recall — Pooled split",
        save_path=str(RESULTS_DIR / "figures/pr_curves.png"),
    )
    plot_confusion_matrices(
        y_pool, preds_pool,
        save_path=str(RESULTS_DIR / "figures/confusion_matrices.png"),
    )
    plot_anomaly_scores(
        scores_pool["Isolation Forest"], y_pool,
        label="Isolation Forest",
        save_path=str(RESULTS_DIR / "figures/scores_isoforest.png"),
    )
    plot_anomaly_scores(
        scores_pool["LSTM Autoencoder"], y_pool,
        threshold=np.mean(scores_pool["LSTM Autoencoder"])
                  + 2 * np.std(scores_pool["LSTM Autoencoder"]),
        label="LSTM Autoencoder",
        save_path=str(RESULTS_DIR / "figures/scores_lstm.png"),
    )
    plot_predictions(
        preds_pool["Isolation Forest"], y_pool,
        label="Isolation Forest",
        save_path=str(RESULTS_DIR / "figures/preds_isoforest.png"),
    )
    plot_predictions(
        preds_pool["LSTM Autoencoder"], y_pool,
        label="LSTM Autoencoder",
        save_path=str(RESULTS_DIR / "figures/preds_lstm.png"),
    )

    # Per-anomaly-type AUC
    plot_per_anomaly_type_auc(
        y_pool, idx_pool, types_pool, scores_pool,
        save_path=str(RESULTS_DIR / "figures/per_anomaly_type_auc.png"),
    )

    # Stream simulation
    stream_signal(
        corrupt_pool[0],
        anomaly_scores=scores_pool["LSTM Autoencoder"],
        save_path=str(RESULTS_DIR / "figures/realtime_stream.png"),
    )

    # ── SHAP explainability (Isolation Forest on pooled split) ────────
    print("\n[A4] Computing SHAP values for Isolation Forest...")

    iso_pool.plot_shap_bar(
        X_feat_pool,
        save_path=str(RESULTS_DIR / "shap/shap_bar.png"),
    )
    iso_pool.plot_shap_summary(
        X_feat_pool,
        save_path=str(RESULTS_DIR / "shap/shap_beeswarm.png"),
    )
    iso_pool.plot_shap_per_anomaly_type(
        X_feat_pool, types_pool, idx_pool,
        save_path=str(RESULTS_DIR / "shap/shap_per_type.png"),
    )
    print("      SHAP plots saved.")

    print_results_table(res_pool)

    # ── PROTOCOL B: Cross-subject split ──────────────────────────────
    print("\n\n══ PROTOCOL B: Cross-subject generalisation split ══")
    print(f"   Train subjects: {TRAIN_SUBJECTS}")
    print(f"   Test  subjects: {TEST_SUBJECTS}")

    print("\n[B1] Loading subject-stratified signals...")
    try:
        (x_tr, y_tr, z_tr), (x_te, y_te, z_te) = load_by_subject(
            TRAIN_SUBJECTS, TEST_SUBJECTS
        )
        train_cs = magnitude(x_tr, y_tr, z_tr)
        test_cs  = magnitude(x_te, y_te, z_te)
        print(f"     Train: {len(train_cs)} windows | "
              f"Test: {len(test_cs)} windows")

        if len(test_cs) < 20:
            raise ValueError("Too few test windows for cross-subject eval.")

        print("\n[B2] Injecting anomalies & training models...")
        (res_cross, scores_cross, preds_cross,
         y_cross, corrupt_cross, idx_cross, types_cross,
         iso_cross, X_feat_cross) = run_models(
            train_cs, test_cs, label_prefix="B", run_lstm=True
        )

        print("\n[B3] Saving cross-subject figures...")
        plot_roc_curves(
            y_cross, scores_cross,
            title="ROC Curves — Cross-subject split",
            save_path=str(
                RESULTS_DIR / "generalisation/roc_cross_subject.png"),
        )
        plot_confusion_matrices(
            y_cross, preds_cross,
            save_path=str(
                RESULTS_DIR / "generalisation/cm_cross_subject.png"),
        )
        plot_generalisation_gap(
            res_pool, res_cross,
            save_path=str(
                RESULTS_DIR / "generalisation/generalisation_gap.png"),
        )

        print_results_table(res_cross)

    except Exception as e:
        print(f"  [B] Cross-subject eval skipped: {e}")
        res_cross = None

    # ── Final summary ─────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"  All results saved to:  {RESULTS_DIR}")
    print("  figures/          — main evaluation charts")
    print("  shap/             — SHAP explainability plots")
    print("  generalisation/   — cross-subject plots")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
