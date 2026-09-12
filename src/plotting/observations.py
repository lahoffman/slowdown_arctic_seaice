"""
observations.py — CNN predictions on observed SST with climate indices (Fig. 4).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from matplotlib.lines import Line2D

from . import style as st
from .style import plt


def _index_panel(ax, years, vals, ylabel, ref_lines=()):
    ax.fill_between(years, vals, 0, where=vals > 0, color=st.C_POS, alpha=0.35)
    ax.fill_between(years, vals, 0, where=vals <= 0, color=st.C_NEG, alpha=0.35)
    ax.plot(years, vals, color=st.INK, lw=1.2)
    ax.axhline(0, color=st.INK, lw=0.6)
    for r in ref_lines:
        ax.axhline(r, color=st.MUTED, ls="--", lw=0.7)
    ax.set_ylabel(ylabel)


def prediction_stack(obs_years, fraction, obs_slow_years, obs_slow, indices: Dict[str, tuple],
                     single_prob: Optional[np.ndarray] = None, threshold: float = 0.5,
                     frac_threshold: float = 0.15, future=(2015.5, 2025.5),
                     title: str = "") -> plt.Figure:
    """
    Stacked panels: [single-model binary prediction], ensemble vote fraction,
    then one panel per index in ``indices`` ({name: (years, values, ylabel, ref_lines)}).
    """
    n = 1 + (single_prob is not None) + len(indices)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.1 * n + 0.8), sharex=True)
    axes = np.atleast_1d(axes)

    def shade(ax, with_votes=False):
        ax.axvspan(*future, color=st.GRID, alpha=0.45, zorder=0)
        for y, s in zip(obs_slow_years, obs_slow):
            if s and y in obs_years:
                ax.axvspan(y - 0.4, y + 0.4, color=st.C_SLOW, alpha=0.25, zorder=0)
        if with_votes:
            for y, f in zip(obs_years, fraction):
                if f > frac_threshold:
                    ax.axvspan(y - 0.4, y + 0.4, facecolor="none", edgecolor="#8b1a1a",
                               lw=1.2, ls="--", zorder=1)

    k = 0
    if single_prob is not None:
        ax = axes[k]; shade(ax)
        ax.plot(obs_years, (single_prob >= threshold).astype(int), color="#8b1a1a", ls="--",
                lw=1.4, marker="o", ms=3.5)
        ax.set_ylim(-0.15, 1.15); ax.set_yticks([0, 1]); ax.set_yticklabels(["NO", "YES"])
        st.panel_label(ax, "(a)"); k += 1
    ax = axes[k]; shade(ax)
    ax.bar(obs_years, fraction, width=0.7, color="#8b1a1a", alpha=0.75, zorder=2)
    ax.axhline(frac_threshold, color=st.INK, ls="--", lw=0.8, label=f"fraction = {frac_threshold}")
    ax.set_ylabel("fraction of CNNs\npredicting slowdown"); ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="upper left")
    st.panel_label(ax, f"({'ab'[k]})"); k += 1
    for name, (yrs, vals, ylabel, refs) in indices.items():
        ax = axes[k]; shade(ax, with_votes=True)
        _index_panel(ax, yrs, vals, ylabel, refs)
        st.panel_label(ax, f"({'abcdefg'[k]})"); k += 1
    axes[-1].set_xlabel("year")
    handles = [Line2D([], [], color=st.C_SLOW, lw=8, alpha=0.3, label="observed slowdown onset"),
               Line2D([], [], color=st.GRID, lw=8, label="window extends past record"),
               Line2D([], [], color="#8b1a1a", ls="--", label=f"> {frac_threshold:.0%} of CNNs vote slowdown")]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, 0.995))
    for ax in axes:
        st.tidy(ax)
    if title:
        fig.suptitle(title, fontsize=11, y=1.02)
    return fig
