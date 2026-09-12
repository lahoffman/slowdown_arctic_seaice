"""
paper.py — manuscript figures, one function per figure.

Each function takes already-loaded data and returns a matplotlib Figure;
loading and saving live in scripts/make_figure.py. Panel helpers shared by
several figures sit at the top.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from matplotlib.lines import Line2D

from . import style as st
from .style import plt

# Slowdown / non-slowdown colours (colour-blind safe; replaces green/red)
C_SLOW, C_NOSLOW = st.BLUE, st.ORANGE
C_MEMBER = "#5e4fa2"
C_THR_OBS, C_THR_MODEL = st.SKY, "#a6cee3"


# =============================================================================
# Panel helpers
# =============================================================================

def panel_label(ax, label, size=13):
    ax.text(0.0, 1.03, label, transform=ax.transAxes, fontsize=size,
            fontweight="bold", va="bottom", ha="left")


def trend_segments(ax, years, series, slopes, mask, window=10, lw=1.2,
                   labels=("trend (slowdown)", "trend (no slowdown)")):
    """Draw each window's fitted trend line, coloured by its slowdown flag."""
    x = np.arange(window)
    seen = {True: False, False: False}
    for j in range(slopes.size):
        if not (np.isfinite(series[j]) and np.isfinite(slopes[j])):
            continue
        slow = bool(mask[j])
        lab = (labels[0] if slow else labels[1]) if not seen[slow] else ""
        seen[slow] = True
        ax.plot(years[j:j + window], slopes[j] * x + series[j],
                color=C_SLOW if slow else C_NOSLOW, lw=lw, label=lab)


def _members_bg(ax, x, arr, label="ensemble members"):
    ax.plot(x, arr.T, lw=0.3, color=st.GRID, zorder=1)
    return Line2D([], [], color=st.GRID, lw=2, label=label)


# =============================================================================
# Figure S1 — slowdown definition (observations + CESM2-LE)
# =============================================================================

def fig_s1(nsidc: dict, sie: np.ndarray, years: np.ndarray, labels: xr.Dataset,
           member: int = 6, window: int = 10, varname: str = "SIE",
           month: str = "SEP", xmax: int = 2100) -> plt.Figure:
    """
    Six-panel definition figure.

    (a,b) NSIDC series with coloured trend segments; trend series with the
          observed mean and μ+σ threshold.
    (c,d) CESM2-LE forced response with its trend classification.
    (e,f) all members with one highlighted member and its slowdowns.

    ``labels`` is either the original slowdown file (threshold_slowdown =
    f × ensemble-mean trend) or a relative-label file (has ``z`` and
    ``reference_trend``); panels (c,d,f) adapt to whichever is passed.
    """
    relative = "z" in labels
    tyrs = labels["nyr"].values.astype(int)
    trends = labels["linear_trends_ens"].values
    slow = labels["slowdown"].values.astype(int)
    nyr = tyrs.size
    ice_lab = f"{month} {varname} [M km²]"
    trend_lab = f"{window}-yr trend [M km² yr⁻¹]"

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    (axa, axb), (axc, axd), (axe, axf) = axes

    # (a) NSIDC series + segments
    axa.plot(nsidc["ice_years"], nsidc["ice"], lw=2.2, color=st.INK, label=f"NSIDC {varname}")
    trend_segments(axa, nsidc["ice_years"], nsidc["ice"], nsidc["trends"],
                   nsidc["slowdown"], window)
    axa.set_ylabel(ice_lab); panel_label(axa, "(a)"); axa.legend(fontsize=8, frameon=False)

    # (b) NSIDC trend series
    ty, tr, sm = nsidc["trend_years"], nsidc["trends"], nsidc["slowdown"]
    axb.plot(ty, tr, lw=1.2, color=st.MUTED)
    axb.scatter(ty[sm == 1], tr[sm == 1], s=22, color=C_SLOW, zorder=3, label="slowdown")
    axb.scatter(ty[sm == 0], tr[sm == 0], s=22, color=C_NOSLOW, zorder=3, label="no slowdown")
    axb.axhline(nsidc["mean_trend"], color=C_THR_OBS, lw=1.5, label="mean trend μ")
    axb.axhline(nsidc["threshold"], color=C_THR_OBS, lw=1.5, ls="--", label="threshold μ+σ")
    axb.set_ylabel(trend_lab); panel_label(axb, "(b)"); axb.legend(fontsize=8, frameon=False)

    # (c,d) forced response
    if relative:
        for name, sl, c in (("CMIP6-BB mean (0–49)", slice(0, 50), st.C_CMIP6),
                            ("SMBB mean (50–99)", slice(50, 100), st.C_SMBB)):
            axc.plot(years, sie[sl].mean(0), lw=1.8, color=c, label=name)
            axd.plot(tyrs, labels["reference_trend"].values[sl][0], lw=1.8, color=c, label=name)
        axc.plot(years, sie.mean(0), lw=1.2, ls="--", color=st.INK, label="100-member mean")
        axd.plot(tyrs, labels["linear_trends_mean"].values, lw=1.2, ls="--", color=st.INK,
                 label="100-member mean")
        axd.axhline(0, color=st.MUTED, lw=0.8)
    else:
        ens_mean = sie.mean(0)
        tm, thr = labels["linear_trends_mean"].values, labels["threshold_slowdown"].values
        mean_mask = (tm > thr).astype(int)
        axc.plot(years, ens_mean, lw=2.2, color=st.INK, label="CESM2-LE ensemble mean")
        trend_segments(axc, years, ens_mean, tm, mean_mask, window)
        axd.plot(tyrs, tm, lw=1.2, color=st.MUTED)
        axd.scatter(tyrs[mean_mask == 1], tm[mean_mask == 1], s=22, color=C_SLOW, zorder=3, label="slowdown")
        axd.scatter(tyrs[mean_mask == 0], tm[mean_mask == 0], s=22, color=C_NOSLOW, zorder=3, label="no slowdown")
        axd.axhline(nsidc["threshold"], color=C_THR_OBS, lw=1.5, ls="--", label="obs threshold")
        axd.plot(tyrs, thr, color=C_THR_MODEL, lw=1.8, label="model threshold f × ens-mean trend")
    axc.set_ylabel(ice_lab); panel_label(axc, "(c)"); axc.legend(fontsize=8, frameon=False)
    axd.set_ylabel(trend_lab); panel_label(axd, "(d)"); axd.legend(fontsize=8, frameon=False)

    # (e) members + highlighted member with slowdown windows
    h = _members_bg(axe, years, sie)
    for j in range(nyr):
        if slow[member, j]:
            i0 = int(np.where(years == tyrs[j])[0][0])
            axe.plot(years[i0:i0 + window], trends[member, j] * np.arange(window) + sie[member, i0],
                     lw=2.2, color=st.C_EVENT, zorder=5)
    axe.plot(years, sie.mean(0), lw=1.5, color=st.INK, label="ensemble mean")
    axe.plot(years, sie[member], lw=1.2, color=C_MEMBER, label=f"member {member}")
    hh, ll = axe.get_legend_handles_labels()
    axe.legend(handles=[h] + hh + [Line2D([], [], color=st.C_EVENT, lw=2.2, label="slowdown windows")],
               fontsize=8, frameon=False)
    axe.set_ylabel(ice_lab); panel_label(axe, "(e)")

    # (f) member trends (or z) with thresholds
    if relative:
        z = labels["z"].values
        h = _members_bg(axf, tyrs, z)
        axf.plot(tyrs, z[member], lw=1.2, color=C_MEMBER, label=f"member {member}")
        axf.scatter(tyrs[slow[member] == 1], z[member][slow[member] == 1], s=24,
                    color=st.C_EVENT, zorder=5, label="slowdown")
        ns = float(labels.attrs.get("n_sigma", 1.0))
        axf.axhline(ns, color=st.INK, ls="--", lw=1, label=f"±{ns:g}σ"); axf.axhline(-ns, color=st.INK, ls="--", lw=1)
        axf.axhline(0, color=st.MUTED, lw=0.8)
        axf.set_ylabel("trend anomaly z [σ]")
    else:
        h = _members_bg(axf, tyrs, trends)
        axf.plot(tyrs, labels["linear_trends_mean"].values, lw=1.5, color=st.INK, label="ensemble mean")
        axf.axhline(nsidc["threshold"], color=C_THR_OBS, lw=1.5, ls="--", label="obs threshold")
        axf.plot(tyrs, labels["threshold_slowdown"].values, color=C_THR_MODEL, lw=1.8, label="model threshold")
        axf.plot(tyrs, trends[member], lw=1.2, color=C_MEMBER, label=f"member {member}")
        axf.scatter(tyrs[slow[member] == 1], trends[member][slow[member] == 1], s=24,
                    color=st.C_EVENT, zorder=5, label="slowdown")
        axf.set_ylabel(trend_lab)
    hh, ll = axf.get_legend_handles_labels()
    axf.legend(handles=[h] + hh, fontsize=8, frameon=False)
    panel_label(axf, "(f)")

    for ax in axes.ravel():
        st.tidy(ax)
    for ax in (axa, axb):
        ax.set_xlim(nsidc["ice_years"][0], nsidc["ice_years"][-1] + 1)
    for ax in (axc, axd, axe, axf):
        ax.set_xlim(years[0], xmax)
    for ax in (axe, axf):
        ax.set_xlabel("year")
    tag = ("relative labels: z = (trend − group-mean trend)/σ > "
           f"{labels.attrs.get('n_sigma', 1):g}" if relative
           else "original labels: trend > f_obs × ensemble-mean trend")
    fig.suptitle(f"Figure S1 — slowdown definition ({tag})", fontsize=11)
    return fig
