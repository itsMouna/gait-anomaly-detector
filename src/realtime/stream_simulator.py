import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def stream_signal(signal: np.ndarray, anomaly_scores: np.ndarray = None,
                  save_path: str = None):
    """
    Simulate a real-time gait stream by rendering the full signal
    with anomaly scores overlaid, saved to a static image.

    In a real deployment this would push frames to a live dashboard;
    here we produce a clean summary chart suitable for reports.
    """
    n = len(signal)
    t = np.arange(n)

    if anomaly_scores is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    else:
        fig, ax1 = plt.subplots(figsize=(14, 3))
        ax2 = None

    # --- Signal panel ---
    ax1.plot(t, signal, lw=0.7, color="#1565C0", alpha=0.85)
    ax1.set_ylabel("Acceleration (g)")
    ax1.set_title("Realtime Gait Stream — Body Acceleration Magnitude")
    ax1.grid(True, alpha=0.3)

    # --- Anomaly score panel ---
    if ax2 is not None and anomaly_scores is not None:
        score_len = len(anomaly_scores)
        t_score = np.arange(score_len)
        threshold = np.mean(anomaly_scores) + 2 * np.std(anomaly_scores)
        ax2.plot(t_score, anomaly_scores, lw=1, color="#E53935", alpha=0.8,
                 label="Reconstruction error")
        ax2.axhline(threshold, color="orange", linestyle="--", lw=1.5,
                    label=f"Threshold ({threshold:.4f})")
        ax2.set_xlabel("Window index")
        ax2.set_ylabel("Anomaly score (MSE)")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved → {save_path}")
    plt.close(fig)
