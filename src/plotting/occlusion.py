"""occlusion.py — region-occlusion figure (step 5.3): skill lost per region, per configuration."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import xarray as xr

from . import style as st
from .style import plt

REGION_LABELS = {"arctic": "Arctic\n(>65°N)", "north_pacific": "North\nPacific", "tropical_pacific": "tropical\nPacific",
                 "north_atlantic": "North\nAtlantic", "not_arctic": "everything\nexcept Arctic"}
TAG_LABELS = {"rel_base": "SST map", "rel_aux": "SST map + SIE scalar", "rel_openwater": "open-water SST + SIE",
              "rel_lag1": "SST one year earlier + SIE", "rel_concurrent": "decade-mean SST + SIE"}


def plot_occlusion(stacks: Dict[str, xr.Dataset], out_png, metric: str = "AUROC",
                   regions: Sequence[str] = ("arctic", "north_pacific", "tropical_pacific", "north_atlantic", "not_arctic")):
    """Δmetric (occluded − full) per region; one bar group per configuration, points = split × seed."""
    tags = list(stacks)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    width = 0.8 / len(tags)
    for i, tag in enumerate(tags):
        v = stacks[tag]["value"].sel(metric=metric)
        d = (v.sel(region=list(regions)) - v.sel(region="none")).stack(s=("split", "run")).values   # (region, n)
        xs = np.arange(len(regions)) + (i - (len(tags) - 1) / 2) * width
        c = st.CATEGORICAL[i % len(st.CATEGORICAL)]
        ax.bar(xs, np.nanmedian(d, axis=1), width=width * 0.9, color=c, alpha=0.85,
               label=f"{TAG_LABELS.get(tag, tag)} (full {float(v.sel(region='none').median()):.2f})")
        for j in range(len(regions)):
            ax.scatter(np.full(d.shape[1], xs[j]) + np.random.default_rng(j).uniform(-0.3, 0.3, d.shape[1]) * width,
                       d[j], s=6, color=c, alpha=0.35, zorder=3)
    ax.axhline(0, color=st.MUTED, lw=0.8)
    ax.set_xticks(range(len(regions))); ax.set_xticklabels([REGION_LABELS.get(r, r) for r in regions])
    ax.set_ylabel(f"Δ{metric} when region is zeroed")
    ax.legend(frameon=False, loc="lower right", title=f"configuration (full-map test {metric})")
    st.tidy(ax)
    if out_png is None:
        return fig
    st.save(fig, out_png)
