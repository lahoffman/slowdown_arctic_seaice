"""
slowdowns.py — diagnostic figures for relative slowdown labels.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import xarray as xr

from src.data.cesm2le.slowdowns_relative import frequency_by_year
from . import style as st
from .style import plt


def plot_relative_labels(ds: xr.Dataset, sie: np.ndarray, years: np.ndarray,
                         out_png, original: xr.Dataset = None, member: int = 7,
                         pool_years: Tuple[int, int] = (1990, 2040)) -> None:
    """
    Three-panel diagnostic: group-mean SIE with one member's slowdowns,
    standardised trend anomaly z with ±n_sigma, and slowdown frequency by
    onset year (original vs relative, per forcing group).
    """
    ink, muted, grid = st.INK, st.MUTED, st.GRID
    c_all, c_cmip6, c_smbb, c_orig = st.C_ALL, st.C_CMIP6, st.C_SMBB, st.C_ORIG
    n_sigma, window = float(ds.attrs["n_sigma"]), int(ds.attrs["window"])
    tyrs = ds["nyr"].values
    lab, z = ds["slowdown"].values, ds["z"].values
    sel = (tyrs >= pool_years[0]) & (tyrs <= pool_years[1])

    fig, axes = plt.subplots(3, 1, figsize=(9, 10.5))

    # (a) SIE: members grey, group means, highlighted member + its slowdown windows
    ax = axes[0]
    ax.plot(years, sie.T, color=grid, lw=0.5, zorder=1)
    if sie.shape[0] == 100:
        ax.plot(years, sie[:50].mean(0), color=c_cmip6, lw=1.8, label="CMIP6-BB mean (0–49)")
        ax.plot(years, sie[50:].mean(0), color=c_smbb, lw=1.8, label="SMBB mean (50–99)")
    ax.plot(years, sie.mean(0), color=ink, lw=1.2, ls="--", label="100-member mean")
    ax.plot(years, sie[member], color=c_all, lw=1.2, label=f"member {member}")
    for j, y0 in enumerate(tyrs):
        if lab[member, j]:
            i0 = int(np.where(years == y0)[0][0])
            yy = sie[member, i0:i0 + window]
            xx = years[i0:i0 + window]
            b = np.polyfit(xx, yy, 1)
            ax.plot(xx, np.polyval(b, xx), color=st.C_EVENT, lw=2.2, zorder=5)
    ax.set_xlim(years[0], min(years[-1], 2060))
    ax.set_ylabel("September SIE [M km²]")
    ax.set_title(f"(a) SIE and relative slowdown windows for member {member} "
                 f"(orange = {window}-yr trends flagged as slowdown)", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, ncol=2)

    # (b) z for all members, highlighted member, ±n_sigma
    ax = axes[1]
    ax.plot(tyrs, z.T, color=grid, lw=0.5, zorder=1)
    ax.plot(tyrs, z[member], color=c_all, lw=1.2, zorder=3)
    ax.scatter(tyrs[lab[member] == 1], z[member][lab[member] == 1], s=22, color=st.C_EVENT, zorder=5)
    ax.axhline(n_sigma, color=ink, ls="--", lw=1); ax.axhline(-n_sigma, color=ink, ls="--", lw=1)
    ax.axhline(0, color=muted, lw=0.8)
    ax.axvspan(pool_years[0], pool_years[1], color=grid, alpha=0.35, zorder=0, label="σ pooling / analysis period")
    ax.set_xlim(tyrs[0], min(tyrs[-1], 2060))
    ax.set_ylabel("trend anomaly z = (trend − group mean) / σ")
    ax.set_title(f"(b) standardised trend anomaly, all members (σ_pool = {float(ds['sigma'][0]):.3f} M km² yr⁻¹, "
                 f"threshold ±{n_sigma:g}σ)", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    # (c) frequency by onset year
    ax = axes[2]
    if original is not None:
        o = original["slowdown"].sel(nyr=slice(tyrs[0], tyrs[-1]))
        ax.plot(o["nyr"].values, o.values.mean(0), color=c_orig, lw=2.2, label="original labels (all members)")
    freq = frequency_by_year(lab, tyrs)
    ax.plot(tyrs, freq["all"], color=c_all, lw=2, label="relative — all")
    if "cmip6" in freq:
        ax.plot(tyrs, freq["cmip6"], color=c_cmip6, lw=1.4, label="relative — CMIP6-BB (0–49)")
        ax.plot(tyrs, freq["smbb"], color=c_smbb, lw=1.4, label="relative — SMBB (50–99)")
    ax.axhline(freq["all"][sel].mean(), color=c_all, ls=":", lw=1)
    ax.axvspan(pool_years[0], pool_years[1], color=grid, alpha=0.35, zorder=0)
    ax.set_xlim(tyrs[0], min(tyrs[-1], 2060)); ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of members flagged slowdown")
    ax.set_xlabel("onset year of trend window")
    ax.set_title("(c) slowdown frequency by onset year — a flat line means the labels carry no forced epoch",
                 fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, ncol=2)

    for ax in axes:
        st.tidy(ax)
    fig.suptitle(f"Relative slowdown labels — window {window} yr, {n_sigma:g}σ, demean={ds.attrs['demean']}",
                 fontsize=11)
    st.save(fig, out_png)


def plot_window_sweep(datasets: Dict[int, xr.Dataset], out_png,
                      pool_years: Tuple[int, int] = (1990, 2040)) -> None:
    """Slowdown frequency by onset year for several window lengths (one line each)."""
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for c, (w, ds) in zip(st.CATEGORICAL, sorted(datasets.items())):
        tyrs = ds["nyr"].values
        ax.plot(tyrs, ds["slowdown"].values.mean(0), color=c, lw=1.6, label=f"{w}-yr window")
    ax.axvspan(pool_years[0], pool_years[1], color=st.GRID, alpha=0.35, zorder=0)
    ax.set_xlim(min(d["nyr"].values[0] for d in datasets.values()), 2060); ax.set_ylim(0, 0.6)
    ax.set_ylabel("fraction of members flagged"); ax.set_xlabel("onset year")
    ax.set_title("Relative slowdown frequency by onset year — window sweep", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, ncol=3)
    st.tidy(ax)
    st.save(fig, out_png)
