"""
Isolation Forest anomaly detector with SHAP explainability.

IsolationForest isolates anomalies by recursively partitioning the
feature space with random splits. Anomalies require fewer splits to
isolate, yielding a shorter average path length and a higher anomaly
score. The model is unsupervised — no anomaly labels are needed at
train time.

SHAP (SHapley Additive exPlanations) decomposes each prediction into
per-feature contributions, making the model interpretable to clinicians:
"this window was flagged because mean_energy was unusually high."
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from features import FEATURE_NAMES


class IsolationForestModel:
    """Isolation Forest wrapper with SHAP explainability."""

    def __init__(self, contamination: float = 0.15,
                 random_state: int = 42):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=200,
            random_state=random_state,
        )
        self._shap_explainer = None

    # ── Core interface ────────────────────────────────────────────────

    def fit(self, X: np.ndarray):
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return 1 for anomaly, 0 for normal."""
        return (self.model.predict(X) == -1).astype(int)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Continuous anomaly score (higher = more anomalous)."""
        return -self.model.decision_function(X)

    # ── SHAP explainability ───────────────────────────────────────────

    def _get_explainer(self, X_background: np.ndarray):
        """Lazy-build a SHAP TreeExplainer using training data."""
        if self._shap_explainer is None:
            import shap
            self._shap_explainer = shap.TreeExplainer(self.model)
        return self._shap_explainer

    def compute_shap(self, X: np.ndarray,
                     X_background: np.ndarray = None) -> np.ndarray:
        """
        Compute SHAP values for every sample in X.

        Returns
        -------
        shap_values : np.ndarray, shape (n_samples, n_features)
            Positive value → feature pushes score toward anomaly.
        """
        explainer   = self._get_explainer(X_background)
        shap_values = explainer.shap_values(X)
        # IsolationForest TreeExplainer returns values for the anomaly
        # score directly; negate so positive = anomalous contribution.
        return -shap_values

    def plot_shap_summary(self, X: np.ndarray,
                          save_path: str = None):
        """
        Beeswarm SHAP summary plot — one dot per sample per feature,
        coloured by feature value.  High-magnitude dots that are warm
        (red) indicate features driving anomaly scores upward.
        """
        import shap
        shap_values = self.compute_shap(X)

        fig, ax = plt.subplots(figsize=(9, 5))
        shap.summary_plot(
            shap_values, X,
            feature_names=FEATURE_NAMES,
            show=False,
            plot_size=None,
        )
        ax = plt.gca()
        ax.set_title("SHAP Feature Importance — Isolation Forest",
                     fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved → {save_path}")
        plt.close("all")

    def plot_shap_bar(self, X: np.ndarray, save_path: str = None):
        """
        Mean |SHAP| bar chart — which features matter most on average.
        Suitable for the paper figure.
        """
        import shap
        shap_values = self.compute_shap(X)
        mean_abs    = np.abs(shap_values).mean(axis=0)

        order = np.argsort(mean_abs)[::-1]
        names = [FEATURE_NAMES[i] for i in order]
        vals  = mean_abs[order]

        fig, ax = plt.subplots(figsize=(8, 4))
        colours = ["#E53935" if v > vals.mean() else "#1E88E5"
                   for v in vals]
        bars = ax.barh(names[::-1], vals[::-1], color=colours[::-1],
                       edgecolor="none", height=0.6)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(
            "Feature contributions to anomaly score (Isolation Forest)")
        ax.axvline(vals.mean(), color="#555", linestyle="--",
                   lw=1, label="Mean importance")
        ax.legend(fontsize=9)
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"  Saved → {save_path}")
        plt.close(fig)

    def plot_shap_per_anomaly_type(self, X: np.ndarray,
                                   type_labels: np.ndarray,
                                   anomaly_idx: np.ndarray,
                                   save_path: str = None):
        """
        Mean |SHAP| grouped by clinical anomaly type.
        Shows which features each pathology activates most strongly.
        """
        import shap
        from config import ANOMALY_TYPES

        shap_values = self.compute_shap(X)
        shap_anoms  = shap_values[anomaly_idx]

        types  = list(ANOMALY_TYPES.keys())
        labels = [ANOMALY_TYPES[t] for t in types]
        n_feat = len(FEATURE_NAMES)

        means  = np.zeros((len(types), n_feat))
        for ti, atype in enumerate(types):
            mask = type_labels == atype
            if mask.sum() > 0:
                means[ti] = np.abs(shap_anoms[mask]).mean(axis=0)

        fig, axes = plt.subplots(1, len(types), figsize=(14, 4),
                                 sharey=True)
        colours = ["#E53935", "#FB8C00", "#1E88E5", "#43A047"]

        for ax, vals, label, colour in zip(axes, means, labels, colours):
            ax.barh(FEATURE_NAMES, vals, color=colour,
                    edgecolor="none", height=0.6, alpha=0.85)
            ax.set_title(label, fontsize=9, wrap=True)
            ax.grid(True, axis="x", alpha=0.3)
            ax.set_xlabel("Mean |SHAP|")

        fig.suptitle(
            "SHAP importance by clinical anomaly type", fontsize=11)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved → {save_path}")
        plt.close(fig)
