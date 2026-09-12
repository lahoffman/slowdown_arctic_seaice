"""
slowdowns.py — sea ice time-series, trend and label-distribution panels.

Panel helpers (trend_segments, members_bg, trend_scatter, distributions)
are shared by Figs. 1, S1 and S2; the relative-label diagnostics at the
bottom are used by 02_cesm2le_slowdowns_relative.py.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

from src.data.cesm2le.slowdowns_relative import frequency_by_year
from . import style as st
from .style import plt


# =============================================================================
# Panel helpers
# =============================================================================

def trend_segments(ax, years, series, slopes, mask, window=10, lw=1.2,
                   labels=("trend (slowdown)", "trend (no slowdown)")):
    """Fitted trend line for each window, coloured by its slowdown flag."""
    x = np.arange(window)
    seen = {True: False, False: False}
    for j in range(slopes.size):
        if not (np.isfinite(series[j]) and np.isfinite(slopes[j])):
            continue
        slow = bool(mask[j])
        lab = (labels[0] if slow else labels[1]) if not seen[slow] else ""
        seen[slow] = True
        ax.plot(years[j:j + window], slopes[j] * x + series[j],
                color=st.C_SLOW if slow else st.C_NOSLOW, lw=lw, label=lab)


def members_bg(ax, x, arr, label="ensemble members"):
    """All members as thin grey lines; returns a legend handle."""
    ax.plot(x, np.asarray(arr).T, lw=0.3, color=st.GRID, zorder=1)
    return Line2D([], [], color=st.GRID, lw=2, label=label)


def trend_scatter(ax, years, trends, mask, line_color=st.MUTED):
    """Trend series with slowdown / non-slowdown points."""
    ax.plot(years, trends, lw=1.2, color=line_color)
    ax.scatter(years[mask == 1], trends[mask == 1], s=22, color=st.C_SLOW, zorder=3, label="slowdown")
    ax.scatter(years[mask == 0], trends[mask == 0], s=22, color=st.C_NOSLOW, zorder=3, label="no slowdown")


def member_windows(ax, years, sie, trends, slow, tyrs, member, window=10, lw=2.2):
    """Highlight one member's series and its slowdown trend windows."""
    for j in range(tyrs.size):
        if slow[member, j]:
            i0 = int(np.where(years == tyrs[j])[0][0])
            ax.plot(years[i0:i0 + window], trends[member, j] * np.arange(window) + sie[member, i0],
                    lw=lw, color=st.C_EVENT, zorder=5)
    ax.plot(years, sie[member], lw=1.2, color=st.C_MEMBER, label=f"member {member}")
    return Line2D([], [], color=st.C_EVENT, lw=lw, label="slowdown windows")


def _fmt(ax, x="%.1f", y="%.2f"):
    ax.xaxis.set_major_formatter(FormatStrFormatter(x))
    ax.yaxis.set_major_formatter(FormatStrFormatter(y))


def label_distributions(axes, slow: np.ndarray, sie_win: np.ndarray, varlabel="SEP SIE",
                        period: str = "", second: Optional[np.ndarray] = None,
                        second_label: str = ""):
    """
    Four panels (Fig. S2 column): events per member, class fraction, SIE PDF,
    SIE-anomaly PDF. ``second`` optionally overlays a second per-member count.
    """
    a, b, c, d = axes
    per = slow.sum(1)
    mx = int(max(per.max(), second.max() if second is not None else 0))
    bins = np.arange(-0.5, mx + 1.5)
    a.hist(per, bins=bins, density=True, color=st.BLUE, alpha=0.75, label=period or None)
    if second is not None:
        a.hist(second, bins=bins, density=True, color=st.ORANGE, alpha=0.4, label=second_label)
        a.legend(frameon=False)
    a.set_xlabel("slowdown events per member"); a.set_ylabel("frequency"); _fmt(a, "%.0f")
    flat = slow.ravel()
    b.bar([0, 1], [1 - flat.mean(), flat.mean()], color="#9a9a9a", width=0.8)
    b.set_xticks([0, 1]); b.set_xticklabels(["no slowdown", "slowdown"])
    b.set_ylabel("fraction of windows"); b.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    for ax, data, xl in ((c, sie_win, f"{varlabel} [M km²]"),
                         (d, sie_win - np.nanmean(sie_win, 0), f"{varlabel} anomaly [M km²]")):
        ax.hist(data[slow == 0], bins=30, density=True, alpha=0.6, color=st.C_NOSLOW, label="no slowdown")
        ax.hist(data[slow == 1], bins=30, density=True, alpha=0.6, color=st.C_SLOW, label="slowdown")
        ax.set_xlabel(xl); ax.set_ylabel("density"); ax.legend(frameon=False); _fmt(ax)
    for ax in axes:
        if period:
            ax.set_title(period, color=st.MUTED, loc="right")
        st.tidy(ax)


def sie_gmt_counts(ax, years, sie_count, gmt_count, both_count):
    """Members with an SIE / GMT / both slowdown per onset year (Fig. S12)."""
    ax.plot(years, sie_count, color=st.SKY, lw=2.2, label="September SIE slowdowns")
    ax.plot(years, gmt_count, color=st.ORANGE, lw=2.2, label="yearly GMT slowdowns")
    ax.plot(years, both_count, color="#9a9a9a", lw=2.2, label="both (SIE & GMT)")
    ax.set_ylabel("number of members with slowdowns"); ax.set_xlabel("onset year")
    ax.set_xlim(years[0], years[-1]); ax.set_ylim(bottom=0); ax.legend(frameon=False)
    st.tidy(ax)


def joint_trend_pdf(ax, gmt_tr, sie_tr):
    """Joint KDE of GMT vs SIE decadal trends with correlation annotation."""
    from scipy.stats import gaussian_kde, pearsonr
    ok = np.isfinite(gmt_tr) & np.isfinite(sie_tr)
    x, y = gmt_tr[ok], sie_tr[ok]
    try:
        xg = np.linspace(*np.percentile(x, [0.5, 99.5]), 150)
        yg = np.linspace(*np.percentile(y, [0.5, 99.5]), 150)
        X, Y = np.meshgrid(xg, yg)
        Z = gaussian_kde(np.vstack([x, y]))(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        cf = ax.contourf(X, Y, Z, levels=20, cmap="Blues")
        plt.colorbar(cf, ax=ax, label="density")
    except np.linalg.LinAlgError:          # degenerate sample → hexbin
        hb = ax.hexbin(x, y, gridsize=40, cmap="Blues", mincnt=1)
        plt.colorbar(hb, ax=ax, label="count")
    ax.axhline(0, color=st.INK, lw=0.5, ls="--"); ax.axvline(0, color=st.INK, lw=0.5, ls="--")
    ax.set_xlabel("GMT decadal trend [K yr⁻¹]"); ax.set_ylabel("SIE decadal trend [M km² yr⁻¹]")
    r = pearsonr(x, y)[0]
    ax.text(0.03, 0.97, f"r = {r:.3f}", transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8, lw=0))


# =============================================================================
# Relative-label diagnostics
# =============================================================================


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
                 f"(orange = {window}-yr trends flagged as slowdown)", loc="left")
    ax.legend(frameon=False, ncol=2)

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
                 f"threshold ±{n_sigma:g}σ)", loc="left")
    ax.legend(frameon=False, loc="upper right")

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
    ax.set_title("(c) slowdown frequency by onset year — a flat line means the labels carry no forced epoch", loc="left")
    ax.legend(frameon=False, ncol=2)

    for ax in axes:
        st.tidy(ax)
    fig.suptitle(f"Relative slowdown labels — window {window} yr, {n_sigma:g}σ, demean={ds.attrs['demean']}")
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
    ax.set_title("Relative slowdown frequency by onset year — window sweep", loc="left")
    ax.legend(frameon=False, ncol=3)
    st.tidy(ax)
    st.save(fig, out_png)


def plot_sigma_mode_comparison(ds_pooled: xr.Dataset, ds_yearly: xr.Dataset, sie: np.ndarray,
                               years: np.ndarray, out_png,
                               pool_years: Tuple[int, int] = (1990, 2040),
                               cap_year: int = 2030) -> None:
    """
    Step 1.2 decision figure — pooled σ vs year-dependent σ for the relative labels.

    (a) trend-anomaly spread across members by onset year: raw σ(t), the 5-yr
        smoothed σ used by ``sigma_mode='yearly'`` and the pooled constant;
        group-mean SIE on the right axis shows where the ice approaches zero.
    (b) slowdown frequency by onset year under each σ mode (all members) and
        the two forcing groups for the yearly mode.
    (c) how many of the 100 labels per onset year differ between the modes.
    The training window and the optional onset cap are shaded / marked.
    """
    tyrs = ds_pooled["nyr"].values
    lab_p, lab_y = ds_pooled["slowdown"].values, ds_yearly["slowdown"].values
    anom = ds_pooled["trend_anom"].values
    n_sigma = float(ds_pooled.attrs["n_sigma"])
    xmax = min(tyrs[-1], 2060)

    fig, axes = plt.subplots(3, 1, figsize=(9, 10.5), sharex=True)

    ax = axes[0]
    ax.plot(tyrs, np.nanstd(anom, axis=0), color=st.MUTED, lw=1, label="σ(t), raw")
    ax.plot(tyrs, ds_yearly["sigma"].values, color=st.ORANGE, lw=2, label="σ(t), 5-yr smoothed (yearly mode)")
    ax.plot(tyrs, ds_pooled["sigma"].values, color=st.BLUE, lw=2, ls="--",
            label=f"σ pooled {pool_years[0]}–{pool_years[1]}")
    ax.set_ylabel("spread of trend anomalies\n[M km² yr⁻¹]")
    ax.set_title("(a) member spread of the decadal-trend anomaly by onset year", loc="left")
    ax2 = ax.twinx()
    isel = (years >= tyrs[0]) & (years <= xmax)
    ax2.plot(years[isel], sie.mean(0)[isel], color=st.INK, lw=1, alpha=0.6,
             label="ensemble-mean Sept SIE (right axis)")
    ax2.set_ylabel("ensemble-mean Sept SIE [M km²]", color=st.INK)
    ax2.spines["top"].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right")

    ax = axes[1]
    fp, fy = frequency_by_year(lab_p, tyrs), frequency_by_year(lab_y, tyrs)
    ax.plot(tyrs, fp["all"], color=st.BLUE, lw=2.2, label="pooled σ — all members")
    ax.plot(tyrs, fy["all"], color=st.ORANGE, lw=2.2, label="yearly σ — all members")
    if "cmip6" in fy:
        ax.plot(tyrs, fy["cmip6"], color=st.C_CMIP6, lw=1, ls=":", label="yearly — CMIP6-BB")
        ax.plot(tyrs, fy["smbb"], color=st.C_SMBB, lw=1, ls=":", label="yearly — SMBB")
    ax.set_ylim(0, 0.6)
    ax.set_ylabel("fraction of members flagged")
    ax.set_title(f"(b) slowdown frequency by onset year (threshold {n_sigma:g}σ)", loc="left")
    ax.legend(frameon=False, ncol=2)

    ax = axes[2]
    diff = (lab_p != lab_y).sum(0)
    ax.bar(tyrs, diff, color=st.MUTED, width=0.8)
    ax.set_ylabel("labels differing\n(of 100 members)")
    ax.set_xlabel("onset year of trend window")
    ax.set_title(f"(c) pooled vs yearly disagreement — {int((lab_p != lab_y)[:, (tyrs >= pool_years[0]) & (tyrs <= pool_years[1])].sum())} "
                 f"of {100 * ((tyrs >= pool_years[0]) & (tyrs <= pool_years[1])).sum()} training-window labels", loc="left")

    for ax in axes:
        ax.axvspan(pool_years[0], pool_years[1], color=st.GRID, alpha=0.35, zorder=0)
        ax.axvline(cap_year, color=st.INK, lw=1, ls="-.")
        ax.set_xlim(tyrs[0], xmax)
        st.tidy(ax)
    axes[2].text(cap_year + 0.5, axes[2].get_ylim()[1] * 0.95, f"optional onset cap {cap_year}",
                 fontsize=plt.rcParams["legend.fontsize"], va="top")
    fig.suptitle(f"Relative labels — pooled vs year-dependent σ (window {int(ds_pooled.attrs['window'])} yr, "
                 f"demean={ds_pooled.attrs['demean']})")
    st.save(fig, out_png)


def sigma_panel(ax, ds: xr.Dataset, pool_years: Tuple[int, int], cap_year: int, xmax: int = 2100) -> None:
    """Spread of member trend anomalies in each onset year vs the single pooled σ used as the threshold."""
    tyrs, anom = ds["nyr"].values.astype(int), ds["trend_anom"].values
    ax.plot(tyrs, np.nanstd(anom, axis=0), color=st.INK, lw=1.6,
            label="std. of member trend anomalies in each onset year")
    ax.plot(tyrs, ds["sigma"].values, color=st.C_ALL, lw=2, ls="--",
            label=f"pooled σ = {float(ds['sigma'][0]):.3f} M km² yr⁻¹ ({pool_years[0]}–{pool_years[1]}) = slowdown threshold")
    ax.axvspan(pool_years[0], pool_years[1], color=st.GRID, alpha=0.5, zorder=0)
    ax.axvline(cap_year, color=st.INK, lw=1, ls="-.")
    ymax = 1.6 * float(np.nanmax(np.nanstd(anom, axis=0)))    # headroom so the legend never sits on the data
    ax.set_xlim(tyrs[0], xmax); ax.set_ylim(0, ymax)
    ax.text(cap_year + 1, 0.02 * ymax, f"onset cap {cap_year}",
            fontsize=plt.rcParams["legend.fontsize"], va="bottom")
    ax.set_ylabel("spread of trend anomalies\n[M km² yr⁻¹]")
    ax.legend(frameon=False, loc="upper right", ncol=1)
    st.tidy(ax)


def base_rate_panel(ax, ds: xr.Dataset, original: xr.Dataset = None,
                    pool_years: Tuple[int, int] = (1990, 2040), cap_year: int = 2030,
                    xmax: int = 2100) -> None:
    """Fraction of members flagged by onset year: relative labels (all / per group) vs the original labels."""
    tyrs, lab = ds["nyr"].values.astype(int), ds["slowdown"].values
    if original is not None:
        o = original["slowdown"].sel(nyr=slice(tyrs[0], xmax))
        ax.plot(o["nyr"].values, o.values.mean(0), color=st.C_ORIG, lw=2.2,
                label="previous definition: threshold ∝ ensemble-mean trend (LB22)")
    freq = frequency_by_year(lab, tyrs)
    ax.plot(tyrs, freq["all"], color=st.C_ALL, lw=2.2, label="relative definition — all members")
    if "cmip6" in freq:
        ax.plot(tyrs, freq["cmip6"], color=st.C_CMIP6, lw=1.4, label="relative — CMIP6-BB (0–49)")
        ax.plot(tyrs, freq["smbb"], color=st.C_SMBB, lw=1.4, label="relative — SMBB (50–99)")
    ax.axvspan(pool_years[0], pool_years[1], color=st.GRID, alpha=0.5, zorder=0)
    ax.axvline(cap_year, color=st.INK, lw=1, ls="-.")
    ax.set_xlim(tyrs[0], xmax); ax.set_ylim(0, 0.9)
    ax.set_ylabel("fraction of members\nflagged slowdown"); ax.set_xlabel("onset year of trend window")
    ax.legend(frameon=False, loc="upper right", ncol=2)
    st.tidy(ax)
