"""obs_predict.py — Fig. 4 layout with the scalar baselines (and CNN votes) on the observed record (AIES Fig. 8b)."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import style as st
from .observations import _index_panel
from .style import plt

LABELS = {"logit_sie_anom": "SIE anomaly", "logit_ipo": "IPO", "logit_sie_ipo": "SIE + IPO",
          "logit_sie_pacific": "SIE + Niño3.4 + IPO"}
C_CNN = "#8b1a1a"


def plot_obs_predict(years, obs: Dict[str, np.ndarray], probs: Dict[str, np.ndarray], z: np.ndarray,
                     frac: Optional[np.ndarray], forced_method: str, out_png, obs_slow_years=(),
                     threshold: float = 0.5, frac_threshold: float = 0.15, future=(2015.5, 2025.5),
                     cnn_label: str = "CNN") -> None:
    """
    Fig. 4 layout. (a) logistic P(slowdown in t+1…t+10) per baseline (band = 9 split-fits) [+ CNN vote
    fraction as bars]; (b) SIE anomaly; (c) IPO; (d) Niño3.4. Shading: observed slowdown onsets by the v1
    linear-threshold definition (blue) and by the offset z > 1 definition against this forced reference (amber);
    grey = window extends past the record.
    """
    idx = {"sie_anom": (f"Sept SIE anomaly\n[10⁶ km²]", ()), "ipo": ("IPO index [°C]", ()), "nino34": ("Niño 3.4 [°C]", (0.4, -0.4))}
    fig, axes = plt.subplots(1 + len(idx), 1, figsize=(11, 2.3 * (1 + len(idx)) + 1.2), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.7] + [1] * len(idx)))
    z_slow = years[np.isfinite(z) & (z > 1)]
    yes = np.nanmedian(probs["logit_sie_ipo"], 0) >= threshold if "logit_sie_ipo" in probs else None

    def shade(ax, with_yes=False):
        ax.axvspan(*future, color=st.GRID, alpha=0.45, zorder=0)
        for y in obs_slow_years:
            if years[0] <= y <= years[-1]:
                ax.axvspan(y - 0.4, y + 0.4, color=st.C_SLOW, alpha=0.22, zorder=0)
        for y in z_slow:
            ax.axvspan(y - 0.4, y + 0.4, color=st.C_EVENT, alpha=0.30, zorder=0)
        if with_yes and yes is not None:
            for y, f in zip(years, yes):
                if f:
                    ax.axvspan(y - 0.4, y + 0.4, facecolor="none", edgecolor=C_CNN, lw=1.2, ls="--", zorder=1)

    ax = axes[0]; shade(ax)
    if frac is not None:
        ax.bar(years, frac, width=0.7, color=C_CNN, alpha=0.55, zorder=2, label=f"{cnn_label}: fraction voting slowdown")
    for i, (name, P) in enumerate(probs.items()):
        col = st.CATEGORICAL[i % len(st.CATEGORICAL)]
        ax.fill_between(years, np.nanmin(P, 0), np.nanmax(P, 0), color=col, alpha=0.15, lw=0)
        ax.plot(years, np.nanmedian(P, 0), color=col, lw=2, label=f"logistic: {LABELS.get(name, name)}", zorder=3)
    ax.axhline(threshold, color=st.INK, ls="--", lw=0.8)
    ax.set_ylim(0, 1.02); ax.set_ylabel("P(slowdown in\nt+1 … t+10)")
    ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=11)
    st.panel_label(ax, "(a)")
    for k, (key, (ylabel, refs)) in enumerate(idx.items(), start=1):
        ax = axes[k]; shade(ax, with_yes=True)
        _index_panel(ax, years, obs[key], ylabel, refs)
        st.panel_label(ax, f"({'abcdefg'[k]})")
    axes[-1].set_xlabel("onset year t")
    handles = [Patch(color=st.C_SLOW, alpha=0.3, label="observed slowdown, v1 definition (linear trend threshold)"),
               Patch(color=st.C_EVENT, alpha=0.4, label=f"observed slowdown, offset z > 1 vs {forced_method} reference"),
               Patch(color=st.GRID, label="window extends past record"),
               Line2D([], [], color=C_CNN, ls="--", label=f"SIE + IPO logistic ≥ {threshold}")]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), fontsize=11)
    for ax in axes:
        st.tidy(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(out_png, dpi=200); plt.close(fig)
    print(f"  figure → {out_png}")
