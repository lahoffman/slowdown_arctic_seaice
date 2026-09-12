"""
baselines.py — skill strip plot for scalar baselines vs the CNN.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.analysis.baselines import BASELINE_FEATURES

#: Readable row labels for the strip plot.
LABELS = {
    "always_positive": "always slowdown", "random_prevalence": "random (prevalence)",
    "year_climatology": "onset-year climatology",
    "logit_sie_anom": "SIE anomaly", "logit_arctic_sst": "Arctic SST index",
    "logit_nino34": "Niño 3.4", "logit_ipo": "IPO", "logit_pacific": "Niño 3.4 + IPO",
    "logit_indices": "Arctic SST + Niño 3.4 + IPO",
    "logit_sie_arctic": "SIE anomaly + Arctic SST", "logit_sie_nino34": "SIE anomaly + Niño 3.4",
    "logit_sie_ipo": "SIE anomaly + IPO", "logit_sie_pacific": "SIE anomaly + Niño 3.4 + IPO",
    "logit_sie_year": "SIE anomaly + onset year", "logit_all_scalars": "all scalar predictors",
    "cnn_median": "CNN (median of 5 seeds)",
}
from . import style as st
from .style import plt


def plot_summary(stacked, out_png=None):
    """Strip plot of per-split test skill for every model; saves if ``out_png`` given, else returns fig."""
    metrics = ["F1", "AUPRC", "AUROC"]
    models = [m for m in stacked.model.values if not m.startswith("cnn_run")]
    order = [m for m in BASELINE_FEATURES if m in models] + \
            [m for m in models if m not in BASELINE_FEATURES]
    ink, muted, accent = st.INK, st.MUTED, st.BLUE

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 7.5), sharey=True)
    for ax, met, lab in zip(axes, metrics, "abc"):
        ax.set_title(f"({lab}) {met}", loc="left", weight="bold")
        for i, m in enumerate(order):
            v = stacked["metric_value"].sel(metric=met, model=m).values
            ax.scatter(v, np.full(v.size, i) + np.random.default_rng(i).uniform(-0.15, 0.15, v.size),
                       s=18, color=accent if m.startswith("cnn") else muted, alpha=0.8, zorder=3)
            ax.plot([np.median(v)] * 2, [i - 0.3, i + 0.3], color=ink, lw=2, zorder=4)
        if met == "F1":
            ax.axvline(float(stacked.attrs.get("always_positive_f1_median", np.nan)),
                       color=ink, ls=":", lw=1, label="always-positive")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([LABELS.get(m, m) for m in order])
        ax.set_xlabel(met + " (test members)")
        st.tidy(ax)
    axes[0].invert_yaxis()
    fig.suptitle("bar = median across 9 train–validate–test splits; dotted = always-positive F1")
    if out_png is None:
        return fig
    st.save(fig, out_png)
