import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_signal(signal: np.ndarray, title: str = "Signal",
                save_path: str = None):
    """Plot a 1-D signal."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(signal, lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved → {save_path}")
    plt.close(fig)


def plot_anomaly_scores(scores: np.ndarray, y_true: np.ndarray,
                        threshold: float = None, label: str = "Model",
                        save_path: str = None):
    """
    Plot anomaly scores over windows with true-label colouring.
    Anomalous windows shown in red, normal in blue.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(scores))
    ax.scatter(x[y_true == 0], scores[y_true == 0],
               c="#2196F3", s=10, alpha=0.6, label="Normal")
    ax.scatter(x[y_true == 1], scores[y_true == 1],
               c="#F44336", s=10, alpha=0.8, label="Anomaly")
    if threshold is not None:
        ax.axhline(threshold, color="orange", linestyle="--", lw=1.5,
                   label=f"Threshold ({threshold:.4f})")
    ax.set_title(f"Anomaly Scores — {label}")
    ax.set_xlabel("Window index")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved → {save_path}")
    plt.close(fig)


def plot_sample_windows(signal: np.ndarray, y_true: np.ndarray,
                        n: int = 4, save_path: str = None):
    """
    Show n normal and n anomalous sample windows side by side.
    """
    normal_idx = np.where(y_true == 0)[0][:n]
    anomaly_idx = np.where(y_true == 1)[0][:n]

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 5), sharey=True)
    for col, idx in enumerate(normal_idx):
        axes[0, col].plot(signal[idx], color="#2196F3", lw=0.8)
        axes[0, col].set_title(f"Normal #{idx}", fontsize=9)
        axes[0, col].grid(True, alpha=0.3)

    for col, idx in enumerate(anomaly_idx):
        axes[1, col].plot(signal[idx], color="#F44336", lw=0.8)
        axes[1, col].set_title(f"Anomaly #{idx}", fontsize=9)
        axes[1, col].grid(True, alpha=0.3)

    axes[0, 0].set_ylabel("Normal")
    axes[1, 0].set_ylabel("Anomaly")
    fig.suptitle("Sample Windows — Normal vs Anomaly", fontsize=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved → {save_path}")
    plt.close(fig)


def plot_predictions(preds: np.ndarray, y_true: np.ndarray = None,
                     label: str = "Predictions", save_path: str = None):
    """
    Plot binary prediction sequence.
    If y_true is provided, colour correct/incorrect differently.
    """
    fig, ax = plt.subplots(figsize=(12, 2.5))
    if y_true is not None:
        correct = preds == y_true
        ax.bar(np.where(correct)[0],  np.ones(correct.sum()),  color="#4CAF50",
               width=1, label="Correct")
        ax.bar(np.where(~correct)[0], np.ones((~correct).sum()), color="#F44336",
               width=1, label="Wrong")
        ax.legend()
    else:
        ax.bar(np.arange(len(preds)), preds, width=1, color="#2196F3")
    ax.set_title(f"Prediction Timeline — {label}")
    ax.set_xlabel("Window index")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Normal", "Anomaly"])
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved → {save_path}")
    plt.close(fig)
