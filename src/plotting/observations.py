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
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 0.995))
    for ax in axes:
        st.tidy(ax)
    if title:
        fig.suptitle(title, y=1.02)
    return fig


def plot_product_comparison(c: dict, lat, lon, out_png, landmask=None) -> None:
    """ERSST vs OISST on the CESM2 grid: Arctic JJA series, climatology and trend difference maps, coverage."""
    from . import maps
    na, nb = c["names"]; yrs = c["years"]
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.4, 1])
    ax = fig.add_subplot(gs[0, :])
    ax.plot(yrs, c["arctic_a"], color=st.C_ORIG, lw=2, label=na)
    ax.plot(yrs, c["arctic_b"], color=st.C_ALL, lw=2, label=nb)
    ax2 = ax.twinx(); ax2.plot(yrs, c["arctic_b"] - c["arctic_a"], color=st.ORANGE, lw=1.2, ls="--", label=f"{nb} − {na}")
    ax2.axhline(0, color=st.MUTED, lw=0.6); ax2.set_ylabel("difference [°C]", color=st.ORANGE); ax2.spines["top"].set_visible(False)
    ax.set_ylabel("Arctic (>65°N) JJA SST [°C]"); ax.set_xlim(yrs[0], yrs[-1])
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, ncol=3, loc="upper left"); st.tidy(ax)
    ax.set_title("(a) Arctic-mean JJA SST on the CESM2-LE grid", loc="left", weight="bold")

    for k, (field, lab, ttl) in enumerate((
            (c["clim_diff"], f"JJA SST, {nb} − {na}, {c['clim'][0]}–{c['clim'][1]} [°C]", "(b) climatological difference"),
            (c["trend_diff"], f"JJA trend, {nb} − {na} [°C/decade]", f"(c) trend difference {yrs[0]}–{yrs[-1]}"))):
        axm = maps.map_axes(fig, gs[1, k])
        f = np.where(landmask == 1, np.nan, field) if landmask is not None else field
        v = float(np.nanpercentile(np.abs(f), 98))
        maps.global_map(axm, lon, lat, f, st.CMAP_SST, -v, v, lab)
        if maps.HAS_CARTOPY:
            try:
                axm.set_extent([-180, 180, 30, 90], crs=maps.ccrs.PlateCarree())
            except Exception:  # pragma: no cover
                pass
        axm.set_title(ttl, loc="left", weight="bold")

    ax = fig.add_subplot(gs[2, :])
    ax.plot(yrs, c["coverage_a"], color=st.C_ORIG, lw=2, label=f"{na}: fraction of Arctic ocean cells with SST")
    ax.plot(yrs, c["coverage_b"], color=st.C_ALL, lw=2, label=f"{nb}: fraction of Arctic ocean cells with SST")
    if c.get("arctic_ice_b") is not None:
        ax.plot(yrs, c["arctic_ice_b"], color=st.GREEN, lw=1.6, ls=":", label=f"{nb}: Arctic-mean JJA ice fraction")
    ax.set_ylim(0, 1.05); ax.set_xlim(yrs[0], yrs[-1]); ax.set_xlabel("year"); ax.set_ylabel("fraction")
    ax.legend(frameon=False, loc="lower left"); st.tidy(ax)
    ax.set_title("(d) coverage north of 65°N and what sits under the ice", loc="left", weight="bold")
    st.save(fig, out_png)
