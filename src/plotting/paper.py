"""
paper.py — manuscript figures, one function per figure.

Each function takes already-loaded data and returns a matplotlib Figure;
loading and saving live in scripts/make_figure.py. Panel drawing is
delegated to the topic modules (slowdowns, maps, performance, conditional,
observations); nothing here touches files.

Main text                               Supplement
  fig_1  schematic + NSIDC + member       fig_s1  slowdown definition (6 panels)
  fig_2  TP composite: SST + LRP          fig_s2  label distributions (8 panels)
  fig_3  P(TP | phase), test              fig_s3  baselines vs CNN skill
  fig_4  observations: votes + indices    fig_s4  PR curve / threshold
                                          fig_s5  confusion matrices
                                          fig_s6  metric strip, all CNNs
                                          fig_s7  test-member timeline
                                          fig_s8  SST composites: all vs CNN-filtered
                                          fig_s9/s10/s11  FP / TN / FN composites
                                          fig_s12 P(event | phase), train, all vs TP
                                          fig_s13 SIE vs GMT slowdown counts
Extras: fig_phase_all, fig_regional_relevance, fig_sie_gmt_joint, fig_learning_curve.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import xarray as xr

from . import style as st
from . import slowdowns as sd
from . import maps, performance as perf, conditional, observations as obs
from .style import plt, panel_label


# =============================================================================
# Figure 1 — schematic + observed record + one member
# =============================================================================

def fig_1(nsidc: dict, sie: np.ndarray, years: np.ndarray, labels: xr.Dataset,
          member: int = 6, schematic: Optional[np.ndarray] = None, window: int = 10,
          varname="SIE", month="SEP", xmax=2100) -> plt.Figure:
    """(a) CNN schematic image (optional), (b) NSIDC SIE with trend segments, (c) members."""
    tyrs = labels["nyr"].values.astype(int)
    trends, slow = labels["linear_trends_ens"].values, labels["slowdown"].values.astype(int)
    if schematic is not None:
        fig = plt.figure(figsize=(15, 6.5))
        gs = fig.add_gridspec(2, 2, width_ratios=[2.0, 1.0], wspace=0.15, hspace=0.35)
        ax_a = fig.add_subplot(gs[:, 0]); ax_a.imshow(schematic); ax_a.axis("off")
        panel_label(ax_a, "(a)", y=0.98)
        ax_b, ax_c = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
        lb, lc = "(b)", "(c)"
    else:
        fig, (ax_b, ax_c) = plt.subplots(1, 2, figsize=(13, 4.5))
        lb, lc = "(a)", "(b)"
    ax_b.plot(nsidc["ice_years"], nsidc["ice"], lw=2.2, color=st.INK, label=f"NSIDC {varname}")
    sd.trend_segments(ax_b, nsidc["ice_years"], nsidc["ice"], nsidc["trends"], nsidc["slowdown"], window)
    ax_b.set_ylabel(f"{month} {varname} [M km²]"); ax_b.legend(frameon=False, fontsize=8)
    panel_label(ax_b, lb)
    h = sd.members_bg(ax_c, years, sie)
    ax_c.plot(years, sie.mean(0), lw=1.6, color=st.INK, label="CESM2-LE ensemble mean")
    hw = sd.member_windows(ax_c, years, sie, trends, slow, tyrs, member, window)
    hh, _ = ax_c.get_legend_handles_labels()
    ax_c.legend(handles=[h] + hh + [hw], frameon=False, fontsize=8)
    ax_c.set_ylabel(f"{month} {varname} [M km²]"); ax_c.set_xlim(years[0], xmax)
    panel_label(ax_c, lc)
    for ax in (ax_b, ax_c):
        st.tidy(ax); ax.set_xlabel("year")
    return fig


# =============================================================================
# Figure 2 / S9–S11 — composites
# =============================================================================

def fig_2(comp: Dict, signed: bool = True, smooth: Optional[float] = None,
          nino_boxes: bool = False) -> plt.Figure:
    """SST and LRP composite for one outcome (TP by default). ``comp`` from CompositeAccumulator.result."""
    boxes = None
    if nino_boxes:
        from src.analysis.composites import NINO_BOXES
        boxes = {k: dict(lat=(-5, 5), lon=v) for k, v in NINO_BOXES.items()}
    return maps.composite_pair(comp["lon"], comp["lat"], comp["sst"], comp["lrp"],
                               signed=signed, smooth_lrp=smooth, boxes=boxes)


fig_s9 = fig_s10 = fig_s11 = fig_2   # FP / TN / FN: same layout, different scenario


def fig_s8(comps: Dict[str, Dict]) -> plt.Figure:
    """SST composites: (a) all slowdowns (b) TP (c) all non-slowdowns (d) TN."""
    order = ["ALL_SLOW", "TP", "ALL_NONSLOW", "TN"]
    c0 = comps[order[0]]
    return maps.composite_grid(c0["lon"], c0["lat"], [comps[k]["sst"] for k in order],
                               ["(a)", "(b)", "(c)", "(d)"], ncol=2, kind="sst")


def fig_regional_relevance(comp: Dict) -> plt.Figure:
    """(a) SST (b) positive relevance with region boxes (c) regional mean relevance."""
    from src.analysis.composites import REGIONS
    fig = plt.figure(figsize=(18, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.8], wspace=0.25)
    ax_a, ax_b = maps.map_axes(fig, gs[0]), maps.map_axes(fig, gs[1])
    maps.sst_panel(ax_a, comp["lon"], comp["lat"], comp["sst"], "(a)")
    maps.lrp_panel(ax_b, comp["lon"], comp["lat"], comp["lrp"], "(b)", signed=False)
    colors = {k: c for k, c in zip(REGIONS, st.CATEGORICAL * 3)}
    maps.add_boxes(ax_b, REGIONS, colors)
    ax_c = fig.add_subplot(gs[2])
    maps.regional_relevance_bars(ax_c, comp["regional"])
    ax_c.set_title("(c)", loc="left", weight="bold")
    return fig


# =============================================================================
# Figure 3 / S12 — event dependence on index phase
# =============================================================================

def fig_3(summaries: Sequence[Dict]) -> plt.Figure:
    """P(TP slowdown | phase) for Arctic SST, IPO, Niño3.4 (test data)."""
    return conditional.phase_figure(summaries, show=("tp",), ylabel="P(TP slowdown | phase)")


def fig_s12(summaries: Sequence[Dict]) -> plt.Figure:
    """All slowdowns vs TP slowdowns by phase (training data)."""
    return conditional.phase_figure(summaries, show=("all", "tp"), ylabel="P(event | phase)")


def fig_phase_all(summaries: Sequence[Dict]) -> plt.Figure:
    """P(slowdown | phase) for all slowdowns only — the model-free statement."""
    return conditional.phase_figure(summaries, show=("all",), ylabel="P(slowdown | phase)")


# =============================================================================
# Figure 4 — observations
# =============================================================================

def fig_4(obs_years, fraction, obs_slow_years, obs_slow, indices: Dict, single_prob=None,
          threshold=0.5, frac_threshold=0.15, title="") -> plt.Figure:
    return obs.prediction_stack(obs_years, fraction, obs_slow_years, obs_slow, indices,
                                single_prob, threshold, frac_threshold, title=title)


# =============================================================================
# Figure S1 — slowdown definition
# =============================================================================

def fig_s1(nsidc: dict, sie: np.ndarray, years: np.ndarray, labels: xr.Dataset,
           member: int = 6, window: int = 10, varname="SIE", month="SEP", xmax=2100) -> plt.Figure:
    """
    Six panels: (a,b) NSIDC series and trends with μ, μ+σ; (c,d) CESM2-LE forced
    response and its trend classification; (e,f) members with one highlighted.
    Adapts (c,d,f) to original or relative label files (the latter carry ``z``).
    """
    relative = "z" in labels
    tyrs = labels["nyr"].values.astype(int)
    trends, slow = labels["linear_trends_ens"].values, labels["slowdown"].values.astype(int)
    ice_lab, trend_lab = f"{month} {varname} [M km²]", f"{window}-yr trend [M km² yr⁻¹]"
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    (axa, axb), (axc, axd), (axe, axf) = axes

    axa.plot(nsidc["ice_years"], nsidc["ice"], lw=2.2, color=st.INK, label=f"NSIDC {varname}")
    sd.trend_segments(axa, nsidc["ice_years"], nsidc["ice"], nsidc["trends"], nsidc["slowdown"], window)
    sd.trend_scatter(axb, nsidc["trend_years"], nsidc["trends"], nsidc["slowdown"])
    axb.axhline(nsidc["mean_trend"], color=st.C_THR_OBS, lw=1.5, label="mean trend μ")
    axb.axhline(nsidc["threshold"], color=st.C_THR_OBS, lw=1.5, ls="--", label="threshold μ+σ")

    if relative:
        for name, sl, c in (("CMIP6-BB mean (0–49)", slice(0, 50), st.C_CMIP6),
                            ("SMBB mean (50–99)", slice(50, 100), st.C_SMBB)):
            axc.plot(years, sie[sl].mean(0), lw=1.8, color=c, label=name)
            axd.plot(tyrs, labels["reference_trend"].values[sl][0], lw=1.8, color=c, label=name)
        axc.plot(years, sie.mean(0), lw=1.2, ls="--", color=st.INK, label="100-member mean")
        axd.plot(tyrs, labels["linear_trends_mean"].values, lw=1.2, ls="--", color=st.INK, label="100-member mean")
        axd.axhline(0, color=st.MUTED, lw=0.8)
    else:
        tm, thr = labels["linear_trends_mean"].values, labels["threshold_slowdown"].values
        mm = (tm > thr).astype(int)
        axc.plot(years, sie.mean(0), lw=2.2, color=st.INK, label="CESM2-LE ensemble mean")
        sd.trend_segments(axc, years, sie.mean(0), tm, mm, window)
        sd.trend_scatter(axd, tyrs, tm, mm)
        axd.axhline(nsidc["threshold"], color=st.C_THR_OBS, lw=1.5, ls="--", label="obs threshold")
        axd.plot(tyrs, thr, color=st.C_THR_MODEL, lw=1.8, label="model threshold f × ens-mean trend")

    h = sd.members_bg(axe, years, sie)
    axe.plot(years, sie.mean(0), lw=1.5, color=st.INK, label="ensemble mean")
    hw = sd.member_windows(axe, years, sie, trends, slow, tyrs, member, window)
    hh, _ = axe.get_legend_handles_labels()
    axe.legend(handles=[h] + hh + [hw], fontsize=8, frameon=False)

    if relative:
        z = labels["z"].values
        h = sd.members_bg(axf, tyrs, z)
        axf.plot(tyrs, z[member], lw=1.2, color=st.C_MEMBER, label=f"member {member}")
        axf.scatter(tyrs[slow[member] == 1], z[member][slow[member] == 1], s=24, color=st.C_EVENT,
                    zorder=5, label="slowdown")
        ns = float(labels.attrs.get("n_sigma", 1.0))
        axf.axhline(ns, color=st.INK, ls="--", lw=1, label=f"±{ns:g}σ"); axf.axhline(-ns, color=st.INK, ls="--", lw=1)
        axf.axhline(0, color=st.MUTED, lw=0.8); axf.set_ylabel("trend anomaly z [σ]")
    else:
        h = sd.members_bg(axf, tyrs, trends)
        axf.plot(tyrs, labels["linear_trends_mean"].values, lw=1.5, color=st.INK, label="ensemble mean")
        axf.axhline(nsidc["threshold"], color=st.C_THR_OBS, lw=1.5, ls="--", label="obs threshold")
        axf.plot(tyrs, labels["threshold_slowdown"].values, color=st.C_THR_MODEL, lw=1.8, label="model threshold")
        axf.plot(tyrs, trends[member], lw=1.2, color=st.C_MEMBER, label=f"member {member}")
        axf.scatter(tyrs[slow[member] == 1], trends[member][slow[member] == 1], s=24, color=st.C_EVENT,
                    zorder=5, label="slowdown"); axf.set_ylabel(trend_lab)
    hh, _ = axf.get_legend_handles_labels()
    axf.legend(handles=[h] + hh, fontsize=8, frameon=False)

    for ax, lab in zip((axa, axc, axe), "ace"):
        ax.set_ylabel(ice_lab); panel_label(ax, f"({lab})")
    for ax, lab in zip((axb, axd), "bd"):
        ax.set_ylabel(trend_lab); panel_label(ax, f"({lab})")
    panel_label(axf, "(f)")
    for ax in (axa, axb, axc, axd):
        ax.legend(fontsize=8, frameon=False)
    for ax in axes.ravel():
        st.tidy(ax)
    for ax in (axa, axb):
        ax.set_xlim(nsidc["ice_years"][0], nsidc["ice_years"][-1] + 1)
    for ax in (axc, axd, axe, axf):
        ax.set_xlim(years[0], xmax)
    for ax in (axe, axf):
        ax.set_xlabel("year")
    tag = (f"relative labels: z = (trend − group-mean trend)/σ > {labels.attrs.get('n_sigma', 1):g}"
           if relative else "original labels: trend > f_obs × ensemble-mean trend")
    fig.suptitle(f"Figure S1 — slowdown definition ({tag})", fontsize=11)
    return fig


# =============================================================================
# Figure S2 — label distributions
# =============================================================================

def fig_s2(sie: np.ndarray, years: np.ndarray, labels: xr.Dataset, split_year: int = 2040,
           varlabel="SEP SIE") -> plt.Figure:
    """Left column: all onset years. Right column: onsets before ``split_year``."""
    tyrs = labels["nyr"].values.astype(int)
    slow = labels["slowdown"].values.astype(int)
    idx = np.searchsorted(years, tyrs)
    sie_win = sie[:, idx]
    early = tyrs < split_year
    fig, axes = plt.subplots(4, 2, figsize=(13, 15))
    sd.label_distributions(axes[:, 0], slow, sie_win, varlabel, f"{tyrs[0]}–{tyrs[-1]}")
    sd.label_distributions(axes[:, 1], slow[:, early], sie_win[:, early], varlabel,
                           f"{tyrs[0]}–{split_year - 1}", second=slow[:, ~early].sum(1),
                           second_label=f"{split_year}–{tyrs[-1]}")
    for ax, lab in zip(axes.T.ravel(), "abcdefgh"):
        panel_label(ax, f"({lab})")
    return fig


# =============================================================================
# Figure S3 — scalar baselines vs CNN
# =============================================================================

def fig_s3(stacked: xr.Dataset) -> plt.Figure:
    """Per-split test skill of the logistic/prior baselines and the CNN (07_baselines.py)."""
    from .baselines import plot_summary
    return plot_summary(stacked, out_png=None)


# =============================================================================
# Figures S4–S7 — model performance
# =============================================================================

def fig_s4(y_true, y_score) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    perf.pr_curve(ax, y_true, y_score)
    return fig


def fig_s5(y_true: Dict, y_score: Dict, threshold: float) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, part, lab in zip(axes, ("train", "test"), "ab"):
        perf.confusion_panel(ax, y_true[part], y_score[part], threshold, f"({lab})")
    return fig


def fig_s6(values: Dict[str, np.ndarray]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    perf.metric_strip(ax, values)
    return fig


def fig_s7(y_true, y_pred, years, member_labels) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    perf.member_timeline(ax, y_true, y_pred, years, member_labels)
    return fig


def fig_learning_curve(history: Dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    perf.learning_curve(ax, history)
    return fig


# =============================================================================
# Figure S13 — SIE vs GMT slowdowns
# =============================================================================

def fig_s13(years, sie_count, gmt_count, both_count) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sd.sie_gmt_counts(ax, years, sie_count, gmt_count, both_count)
    return fig


def fig_sie_gmt_joint(gmt_tr, sie_tr) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 6))
    sd.joint_trend_pdf(ax, gmt_tr, sie_tr)
    st.tidy(ax)
    return fig
