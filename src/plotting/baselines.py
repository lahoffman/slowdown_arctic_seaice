"""
baselines.py — skill strip plot for scalar baselines vs the CNN.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.analysis.baselines import BASELINE_FEATURES
from . import style as st
from .style import plt


def plot_summary(stacked, out_png: Path) -> None:
    """Strip plot of per-split test skill for every model."""
    metrics = ["F1", "AUPRC", "AUROC"]
    models = [m for m in stacked.model.values if not m.startswith("cnn_run")]
    order = [m for m in BASELINE_FEATURES if m in models] + \
            [m for m in models if m not in BASELINE_FEATURES]
    ink, muted, accent = st.INK, st.MUTED, st.BLUE

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.6), sharey=True)
    for ax, met in zip(axes, metrics):
        for i, m in enumerate(order):
            v = stacked["metric_value"].sel(metric=met, model=m).values
            ax.scatter(v, np.full(v.size, i) + np.random.default_rng(i).uniform(-0.15, 0.15, v.size),
                       s=18, color=accent if m.startswith("cnn") else muted, alpha=0.8, zorder=3)
            ax.plot([np.median(v)] * 2, [i - 0.3, i + 0.3], color=ink, lw=2, zorder=4)
        if met == "F1":
            ax.axvline(float(stacked.attrs.get("always_positive_f1_median", np.nan)),
                       color=ink, ls=":", lw=1, label="always-positive")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=9)
        ax.set_xlabel(met + " (test)")
        st.tidy(ax, grid_axis="x")
    axes[0].invert_yaxis()
    fig.suptitle("Baselines vs CNN — per-split test skill (bar = median across 9 splits)", fontsize=11)
    st.save(fig, out_png)
