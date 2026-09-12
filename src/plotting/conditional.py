"""
conditional.py — P(event | climate-index phase) bar panels (Fig. 3 / S11).
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from . import style as st
from .style import plt


def _err(dep: Dict, phases) -> tuple:
    p = [dep["probs"][ph] for ph in phases]
    lo = [dep["probs"][ph] - dep["ci_lo"][ph] for ph in phases]
    hi = [dep["ci_hi"][ph] - dep["probs"][ph] for ph in phases]
    return p, [lo, hi]


def phase_bars(ax, summary: Dict, show: Sequence[str] = ("all", "tp"), label: str = "") -> None:
    """
    One index panel from ``phase_stats.phase_summary``.

    ``show`` picks which bar sets to draw: 'all' (all slowdowns, grey) and/or
    'tp' (correctly predicted slowdowns, blue). VE is annotated for each.
    """
    phases, names = summary["phases"], summary["names"]
    x = np.arange(len(phases))
    both = len(show) == 2 and "tp" in summary
    w = 0.35 if both else 0.5
    text = []
    if "all" in show:
        p, err = _err(summary["all"], phases)
        ax.bar(x - (w / 2 if both else 0), p, w, yerr=err, color="#9a9a9a", alpha=0.8,
               edgecolor=st.INK, capsize=3, label="all slowdowns")
        ve = summary["ve_all"]
        text.append(f"VE = {ve['ve']:.3f} [{ve['ci_lo']:.3f}, {ve['ci_hi']:.3f}]"
                    + (" (all)" if both else ""))
    if "tp" in show and "tp" in summary:
        p, err = _err(summary["tp"], phases)
        ax.bar(x + (w / 2 if both else 0), p, w, yerr=err, color=st.BLUE, alpha=0.8,
               edgecolor=st.INK, capsize=3, label="TP slowdowns")
        ve = summary["ve_tp"]
        text.append(f"VE = {ve['ve']:.3f} [{ve['ci_lo']:.3f}, {ve['ci_hi']:.3f}]"
                    + (" (TP)" if both else ""))
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_title(f"{label} {summary['title']}".strip(), loc="left", weight="bold")
    ax.text(0.02, 0.98, "\n".join(text), transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.7, lw=0))
    st.tidy(ax)


def phase_figure(summaries: Sequence[Dict], show=("all", "tp"), ylabel="P(event | phase)"):
    """Row of phase panels, one per index, shared y."""
    fig, axes = plt.subplots(1, len(summaries), figsize=(4.3 * len(summaries), 4.2), sharey=True)
    for ax, s, lab in zip(np.atleast_1d(axes), summaries, "abcdef"):
        phase_bars(ax, s, show, f"({lab})")
    np.atleast_1d(axes)[0].set_ylabel(ylabel)
    if len(show) == 2:
        np.atleast_1d(axes)[-1].legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 0.88))
    return fig
