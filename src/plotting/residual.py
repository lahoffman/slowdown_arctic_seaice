"""residual.py — figure for the residual analysis (step 8.1)."""

from __future__ import annotations

import numpy as np
import xarray as xr

from . import maps
from . import style as st
from .style import plt


def plot_residual(ds: xr.Dataset, trend_anom, sie_anom, r_onset, r_conc, lat, lon, out_png,
                  window: int = 10):
    """
    (a) trend anomaly vs SIE anomaly at onset with the pooled fit;
    (b) ΔR² over the SIE-only model per index set, onset vs concurrent (points = splits, bar = median);
    (c, d) correlation of the residual with JJA SST at onset and averaged over the trend decade.
    """
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])

    ax = fig.add_subplot(gs[0, 0])
    x, y = sie_anom.ravel(), trend_anom.ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[ok], y[ok], s=4, color=st.C_ALL, alpha=0.25, rasterized=True)
    b = np.polyfit(x[ok], y[ok], 1); xx = np.array([np.nanmin(x), np.nanmax(x)])
    ax.plot(xx, np.polyval(b, xx), color=st.INK, lw=2)
    ax.axhline(0, color=st.MUTED, lw=0.7)
    ax.set_xlabel("September SIE anomaly at onset [M km²]")
    ax.set_ylabel(f"{window}-yr trend anomaly [M km² yr⁻¹]")
    ax.set_title(f"(a) trend vs ice state, r = {np.corrcoef(x[ok], y[ok])[0, 1]:.2f}", loc="left", weight="bold")
    st.tidy(ax)

    ax = fig.add_subplot(gs[0, 1])
    sets = list(ds["set"].values)
    for i, (timing, c, lab) in enumerate((("onset", st.C_ORIG, "onset year"), ("conc", st.C_SMBB, f"{window}-yr mean (concurrent)"))):
        d = (ds["r2_sie_plus"].sel(timing=timing) - ds["r2_sie"]).values      # (set, split)
        xs = np.arange(len(sets)) + (i - 0.5) * 0.3
        for j in range(len(sets)):
            ax.scatter(np.full(d.shape[1], xs[j]), d[j], s=12, color=c, alpha=0.5)
        ax.bar(xs, np.nanmedian(d, axis=1), width=0.26, color=c, alpha=0.8, label=lab)
    ax.axhline(0, color=st.MUTED, lw=0.7)
    ax.set_xticks(range(len(sets))); ax.set_xticklabels(sets)
    ax.set_ylabel("ΔR² over SIE-only (test members)")
    ax.set_title(f"(b) added R² beyond SIE alone (R² = {float(ds['r2_sie'].median()):.2f})",
                 loc="left", weight="bold")
    ax.legend(frameon=False, loc="upper left"); st.tidy(ax)
    ax.set_ylim(top=ax.get_ylim()[1] + 0.35 * np.ptp(ax.get_ylim()))

    vmax = 0.3
    for pos, r, title in ((gs[1, 0], r_onset, "(c) corr(residual, JJA SST at onset)"),
                          (gs[1, 1], r_conc, f"(d) corr(residual, {window}-yr mean JJA SST)")):
        axm = maps.map_axes(fig, pos)
        maps.global_map(axm, lon, lat, r, st.CMAP_SST, -vmax, vmax, "correlation")
        axm.set_title(title, loc="left", weight="bold")
    fig.tight_layout()
    if out_png is None:
        return fig
    st.save(fig, out_png)
