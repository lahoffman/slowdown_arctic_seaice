"""ohc.py — figures for the OHC analyses (LB22 test; sea-ice ledger)."""

from __future__ import annotations

import numpy as np

from . import maps
from . import style as st
from .style import plt

PRED_LABEL = {"state": "target state at onset", "ohc_global": "global-mean OHC100",
              "ohc_map_pc20": "OHC100 map (20 PCs)", "ohc_map_pc50": "OHC100 map (50 PCs)",
              "ohc_map_pc20+state": "map (20 PCs) + state", "ohc_map_pc50+state": "map (50 PCs) + state",
              "ohc_map_pc20_ann": "OHC100 map (20 PCs), MLP", "ohc_map_pc50_ann": "OHC100 map (50 PCs), MLP"}


def plot_lb22_test(med, offsets, out_png) -> None:
    """Two rows (GMT, SIE) × two metrics (R², AUROC): bars per predictor, one colour per window offset."""
    targets = ["GMT", "SIE"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey="col")
    for i, tgt in enumerate(targets):
        sub = med[med.target == tgt]
        preds = [p for p in PRED_LABEL if p in set(sub.predictor)] + sorted(set(sub.predictor) - set(PRED_LABEL))
        x = np.arange(len(preds)); w = 0.8 / len(offsets)
        for j, metric in enumerate(("r2", "auroc")):
            ax = axes[i, j]
            for k, off in enumerate(offsets):
                vals = [float(sub[(sub.offset == off) & (sub.predictor == p)][metric].iloc[0]) if len(sub[(sub.offset == off) & (sub.predictor == p)]) else np.nan for p in preds]
                ax.bar(x + (k - (len(offsets) - 1) / 2) * w, vals, width=w * 0.92, color=st.CATEGORICAL[k],
                       label=f"window starts t+{off}")
            if metric == "auroc":
                ax.axhline(0.5, color=st.MUTED, lw=0.8, ls="--")
            ax.set_xticks(x); ax.set_xticklabels([PRED_LABEL.get(p, p) for p in preds], rotation=25, ha="right")
            ax.set_ylabel("test R² of trend anomaly" if metric == "r2" else "test AUROC (slowdown)")
            st.panel_label(ax, f"({'abcd'[2 * i + j]}) {tgt}: {'continuous' if metric == 'r2' else 'binary'}")
            st.tidy(ax)
    axes[0, 0].legend(frameon=False, loc="upper right")
    fig.suptitle("Labe & Barnes' predictor (OHC100) with a state baseline and an offset window — CESM2-LE cmip6 members",
                 y=1.0, fontsize=13)
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig); print(f"  figure → {out_png}")


def plot_sie_ledger(delta: dict, corr_map, lat, lon, out_png, base_r2: float) -> None:
    """(a) added test R² over SIE(+volume) per OHC predictor set; (b) correlation of the residual with OHC at onset."""
    fig = plt.figure(figsize=(14, 5.2)); gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.4])
    ax = fig.add_subplot(gs[0])
    names = list(delta); med = [np.nanmedian(delta[n]) for n in names]
    ax.barh(names, med, color=st.BLUE)
    for i, n in enumerate(names):
        ax.scatter(delta[n], np.full(len(delta[n]), i), s=14, color=st.INK, alpha=0.6, zorder=3)
    ax.axvline(0, color=st.INK, lw=0.6); ax.set_xlabel(f"added test R² beyond the ice state (R² {base_r2:.3f})")
    st.panel_label(ax, "(a) OHC at onset → residual decadal trend"); st.tidy(ax)
    ax2 = maps.map_axes(fig, gs[1])
    im = maps.global_map(ax2, lon, lat, corr_map, "RdBu_r", -0.3, 0.3, cbar_label="correlation")
    st.panel_label(ax2, "(b) corr(residual trend anomaly, OHC anomaly at onset)")
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig); print(f"  figure → {out_png}")
