"""ledger_grl.py — the two composite figures of the GRL paper: the predictor ledger and the residual maps."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from . import maps
from . import style as st
from .style import plt

INDEX_LABEL = {"arctic": "Arctic SST", "nino34": "Niño 3.4", "ipo": "IPO", "pacific": "Niño 3.4 + IPO", "indices": "all indices"}
REGION_LABEL = {"barents_kara": "Barents–Kara", "nordic_gin": "Nordic / GIN", "labrador_baffin": "Labrador–Baffin",
                "bering_chukchi": "Bering–Chukchi", "subpolar_atl": "subpolar Atlantic", "north_pacific": "North Pacific",
                "trop_pacific": "tropical Pacific", "global": "global mean", "all regions": "all regions"}


def _bars(ax, rows, colors, xlabel, zero=True):
    """Horizontal bars (median) with split dots; rows = [(label, values, color_key)]."""
    y = np.arange(len(rows))
    for i, (lab, vals, ck) in enumerate(rows):
        vals = np.asarray(vals, float); vals = vals[np.isfinite(vals)]
        ax.barh(i, np.nanmedian(vals), color=colors[ck], height=0.7)
        ax.scatter(vals, np.full(vals.size, i), s=12, color=st.INK, alpha=0.6, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
    if zero:
        ax.axvline(0, color=st.INK, lw=0.6)
    ax.set_xlabel(xlabel); st.tidy(ax)


def fig_ledger_grl(d: Dict) -> plt.Figure:
    """
    (a) ice state R² and ΔR² of each ocean-surface candidate, onset (grey) vs decade mean (green);
    (b) ΔR² of OHC at onset / decade mean by region and depth as a heat-map; (c) the GMT ledger for contrast.
    ``d`` is the dict from make_figure.Data.ledger_grl().
    """
    fig = plt.figure(figsize=(17, 7), constrained_layout=True); gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.3, 0.9])
    col = {"state": st.BLUE, "onset": st.MUTED, "conc": st.GREEN, "ref": st.ORANGE}

    # (a) ocean surface
    ax = fig.add_subplot(gs[0])
    rows = [("ice state: SIE", d["r2_sie"], "state")]
    if d.get("r2_sie_siv") is not None:
        rows.append(("ice state: SIE + volume", d["r2_sie_siv"], "state"))
    for k in ("arctic", "nino34", "ipo", "pacific"):
        if k in d["idx_onset"]:
            rows.append((f"+ {INDEX_LABEL[k]}, onset", d["idx_onset"][k], "onset"))
            rows.append((f"+ {INDEX_LABEL[k]}, decade mean", d["idx_conc"][k], "conc"))
    for k, v in d.get("pcs_onset", {}).items():
        rows.append((f"+ SST PCs ({k}), onset", v, "onset"))
        rows.append((f"+ SST PCs ({k}), decade mean", d["pcs_conc"][k], "conc"))
    if d.get("cnn") is not None:
        rows.append(("+ CNN on the SST map", d["cnn"], "onset"))
    _bars(ax, rows, col, "test R² (ice state) and added test R² (ocean surface)")
    st.panel_label(ax, "(a) ocean surface")

    # (b) OHC heat-map: rows regions, columns depth × timing
    ax = fig.add_subplot(gs[1])
    H, cols = d["ohc_matrix"], d["ohc_cols"]; regs = d["ohc_rows"]
    vmax = max(0.03, np.nanmax(np.abs(H)))
    im = ax.imshow(H, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=35, ha="right")
    ax.set_yticks(range(len(regs))); ax.set_yticklabels([REGION_LABEL.get(r, r) for r in regs])
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if np.isfinite(H[i, j]):
                ax.text(j, i, f"{H[i, j]:+.3f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(H[i, j]) > 0.6 * vmax else st.INK)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.06, pad=0.04, shrink=0.8); cb.set_label("added test R² beyond the ice state")
    st.panel_label(ax, "(b) upper-ocean heat content")

    # (c) GMT ledger
    ax = fig.add_subplot(gs[2])
    g = d.get("gmt")
    if g is not None:
        rows = [("GMT state at onset", [g["base"]], "state")]
        for k, v in g["delta"].items():
            rows.append((("+ " + REGION_LABEL.get(k, k)) if not k.startswith("IPO") else "+ IPO (reference)", v,
                         "ref" if k.startswith("IPO") else "onset"))
        _bars(ax, rows, col, "test R² / added test R² (GMT trend)")
    st.panel_label(ax, "(c) GMT")
    return fig


def fig_maps_grl(d: Dict, vmax: float = 0.3) -> plt.Figure:
    """2×2: correlation of the ice-state residual with SST (top) and OHC100 (bottom), at onset (left) and decade mean (right)."""
    fig = plt.figure(figsize=(13, 7), constrained_layout=True); gs = fig.add_gridspec(2, 2)
    panels = [("(a) JJA SST at onset", d["sst_onset"], d["lat"], d["lon"]),
              ("(b) decade-mean JJA SST", d["sst_conc"], d["lat"], d["lon"]),
              ("(c) OHC100 at onset", d.get("ohc_onset"), d.get("olat"), d.get("olon")),
              ("(d) decade-mean OHC100", d.get("ohc_conc"), d.get("olat"), d.get("olon"))]
    for k, (title, field, lat, lon) in enumerate(panels):
        ax = maps.map_axes(fig, gs[k // 2, k % 2])
        if field is not None:
            maps.global_map(ax, lon, lat, field, "RdBu_r", -vmax, vmax, cbar=(k == 3), cbar_label="correlation with the residual trend")
        st.panel_label(ax, title)
    return fig
