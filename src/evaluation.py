"""
Evaluation metrics and publication-quality plots.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    f1_score, matthews_corrcoef,
)
from sklearn.decomposition import PCA


# ── Per-model scalar metrics ──────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_score: np.ndarray = None) -> dict:
    """Return a dict of all scalar metrics for one model."""
    cm        = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics   = {
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "mcc":       matthews_corrcoef(y_true, y_pred),
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "recall":    tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
    }
    if y_score is not None:
        fpr, tpr, _   = roc_curve(y_true, y_score)
        metrics["auc_roc"] = auc(fpr, tpr)
        metrics["auc_pr"]  = average_precision_score(y_true, y_score)
    return metrics


def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             y_score: np.ndarray = None, label: str = "Model") -> dict:
    """Print a full report and return metric dict."""
    m = compute_metrics(y_true, y_pred, y_score)
    print(f"\n{'='*52}")
    print(f"  {label}")
    print(f"{'='*52}")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred,
                                target_names=["Normal", "Anomaly"],
                                zero_division=0))
    print(f"  MCC  : {m['mcc']:.4f}")
    if "auc_roc" in m:
        print(f"  AUC-ROC : {m['auc_roc']:.4f}   AUC-PR : {m['auc_pr']:.4f}")
    return m


def print_results_table(results: dict):
    """Print a LaTeX-ready Markdown results table."""
    print("\n" + "="*70)
    print("  RESULTS SUMMARY")
    print("="*70)
    header = f"{'Model':<22} {'AUC-ROC':>8} {'AUC-PR':>8} "
    header += f"{'F1':>7} {'MCC':>7} {'Prec':>7} {'Recall':>7}"
    print(header)
    print("-"*70)
    for label, m in results.items():
        row  = f"{label:<22}"
        row += f" {m.get('auc_roc', 0):>8.3f}"
        row += f" {m.get('auc_pr',  0):>8.3f}"
        row += f" {m['f1']:>7.3f}"
        row += f" {m['mcc']:>7.3f}"
        row += f" {m['precision']:>7.3f}"
        row += f" {m['recall']:>7.3f}"
        print(row)
    print("="*70)


# ── Comparison plots ──────────────────────────────────────────────────

def plot_roc_curves(y_true: np.ndarray, scores: dict,
                    title: str = "ROC Curves — Anomaly Detection",
                    save_path: str = None):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.50)")
    for label, score in scores.items():
        fpr, tpr, _ = roc_curve(y_true, score)
        ax.plot(fpr, tpr, lw=2,
                label=f"{label} (AUC = {auc(fpr,tpr):.3f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150); print(f"  Saved → {save_path}")
    plt.close(fig)


def plot_pr_curves(y_true: np.ndarray, scores: dict,
                   title: str = "Precision-Recall Curves",
                   save_path: str = None):
    fig, ax = plt.subplots(figsize=(7, 6))
    baseline = y_true.mean()
    ax.axhline(baseline, color="k", linestyle="--", lw=1,
               label=f"Random (AP = {baseline:.2f})")
    for label, score in scores.items():
        p, r, _ = precision_recall_curve(y_true, score)
        ap = average_precision_score(y_true, score)
        ax.plot(r, p, lw=2, label=f"{label} (AP = {ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150); print(f"  Saved → {save_path}")
    plt.close(fig)


def plot_confusion_matrices(y_true: np.ndarray, preds: dict,
                             save_path: str = None):
    n    = len(preds)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1: axes = [axes]
    for ax, (label, y_pred) in zip(axes, preds.items()):
        cm     = confusion_matrix(y_true, y_pred)
        im     = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(label)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Normal","Anomaly"])
        ax.set_yticklabels(["Normal","Anomaly"])
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i,j]),
                        ha="center", va="center", fontsize=14,
                        fontweight="bold",
                        color="white" if cm[i,j] > thresh else "black")
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150); print(f"  Saved → {save_path}")
    plt.close(fig)


def plot_pca(X: np.ndarray, y_true: np.ndarray,
             save_path: str = None):
    pca  = PCA(n_components=2)
    X2   = pca.fit_transform(X)
    var  = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, colour, marker in [(0,"#2196F3","o"),(1,"#F44336","x")]:
        mask = y_true == label
        name = "Normal" if label == 0 else "Anomaly"
        ax.scatter(X2[mask,0], X2[mask,1], c=colour, marker=marker,
                   alpha=0.5, s=20, label=name)
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)")
    ax.set_title("PCA — Feature Space (Normal vs Anomaly)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150); print(f"  Saved → {save_path}")
    plt.close(fig)


def plot_generalisation_gap(pool_results: dict, cross_results: dict,
                             save_path: str = None):
    """
    Bar chart comparing AUC-ROC for pooled vs cross-subject splits.
    The gap between bars is the generalisation cost.
    """
    models  = list(pool_results.keys())
    pool_v  = [pool_results[m].get("auc_roc", 0) for m in models]
    cross_v = [cross_results[m].get("auc_roc", 0) for m in models]

    x   = np.arange(len(models))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1  = ax.bar(x - w/2, pool_v,  w, label="Pooled split",
                 color="#1E88E5", edgecolor="none")
    b2  = ax.bar(x + w/2, cross_v, w, label="Cross-subject split",
                 color="#E53935", edgecolor="none")

    # Annotate gap
    for xi, p, c in zip(x, pool_v, cross_v):
        gap = p - c
        ax.annotate(f"Δ{gap:+.3f}",
                    xy=(xi + w/2, c), xytext=(0, 6),
                    textcoords="offset points",
                    ha="center", fontsize=8, color="#555")

    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim(0.4, 1.05)
    ax.set_title("Generalisation gap: pooled vs cross-subject evaluation")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(0.5, color="k", linestyle="--", lw=1)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150); print(f"  Saved → {save_path}")
    plt.close(fig)


def plot_per_anomaly_type_auc(y_true: np.ndarray,
                               anomaly_idx: np.ndarray,
                               type_labels: np.ndarray,
                               scores: dict,
                               save_path: str = None):
    """
    AUC-ROC per anomaly type per model.
    Reveals which pathologies are easy vs hard to detect.
    """
    from config import ANOMALY_TYPES

    types   = list(ANOMALY_TYPES.keys())
    labels  = [ANOMALY_TYPES[t] for t in types]
    n_types = len(types)
    n_model = len(scores)

    model_names = list(scores.keys())
    x           = np.arange(n_types)
    width       = 0.25
    colours     = ["#1E88E5", "#E53935", "#43A047"]

    fig, ax = plt.subplots(figsize=(10, 5))

    for mi, (mname, score) in enumerate(scores.items()):
        aucs = []
        for atype in types:
            # Build binary vector: 1 only for this anomaly type
            y_bin = np.zeros(len(y_true), dtype=int)
            mask  = type_labels == atype
            y_bin[anomaly_idx[mask]] = 1

            if y_bin.sum() == 0:
                aucs.append(0.5)
                continue
            try:
                fpr, tpr, _ = roc_curve(y_bin, score)
                aucs.append(auc(fpr, tpr))
            except Exception:
                aucs.append(0.5)

        offset = (mi - n_model/2 + 0.5) * width
        ax.bar(x + offset, aucs, width,
               label=mname, color=colours[mi % len(colours)],
               edgecolor="none", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=10, ha="right")
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim(0.4, 1.05)
    ax.set_title("Detection performance by clinical anomaly type")
    ax.axhline(0.5, color="k", linestyle="--", lw=1)
    ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150); print(f"  Saved → {save_path}")
    plt.close(fig)
