"""sensitivity.py — heat-maps for the label-definition sweep (step 4.2)."""

from __future__ import annotations

import numpy as np
import xarray as xr

from . import style as st
from .style import plt


def plot_sweep(ds: xr.Dataset, models, out_png, metric: str = "AUROC") -> None:
    """Heat-maps (window × σ): prevalence, then test ``metric`` per model (median over splits), 2 × 2 grid."""
    med = ds["value"].median("split")
    panels = [("prevalence", ds["prevalence"].mean("split"), "Greys", (0, 0.4))] + \
             [(f"{metric}, {m.replace('logit_', '')}", med.sel(metric=metric, model=m), "viridis", (0.5, 0.85))
              for m in models]
    ncol = 2; nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.0 * nrow), squeeze=False)
    axes = axes.ravel()
    W, S = ds["window"].values, ds["n_sigma"].values
    for k, (ax, (title, arr, cmap, (v0, v1))) in enumerate(zip(axes, panels)):
        grid = arr.transpose("window", "n_sigma").values
        ax.imshow(grid, origin="lower", cmap=cmap, vmin=v0, vmax=v1, aspect="auto")
        for i in range(len(W)):
            for j in range(len(S)):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                        color="white" if (grid[i, j] - v0) / (v1 - v0) < 0.55 else "black")
        ax.set_xticks(range(len(S))); ax.set_xticklabels([f"{s:g}σ" for s in S])
        ax.set_yticks(range(len(W))); ax.set_yticklabels([f"{w} yr" for w in W])
        ax.set_title(f"({'abcdefgh'[k]}) {title}", loc="left", weight="bold")
        ax.tick_params(length=0)
        if k % ncol == 0:
            ax.set_ylabel("trend window")
        if k >= len(panels) - ncol:
            ax.set_xlabel("threshold")
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.tight_layout()
    if out_png is None:
        return fig
    st.save(fig, out_png)


def plot_by_group(ds: xr.Dataset, out_png) -> None:
    """Per-model AUROC and F1 for CMIP6-BB vs SMBB test members (points = splits, bar = median)."""
    models = list(ds["model"].values)
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.6 * len(models) + 2), sharey=True)
    colors = {"all": st.C_ALL, "cmip6": st.C_CMIP6, "smbb": st.C_SMBB}
    for ax, metric in zip(axes, ("AUROC", "F1")):
        for i, m in enumerate(models):
            for k, g in enumerate(ds["group"].values):
                v = ds["value"].sel(metric=metric, model=m, group=g).values
                y = i + (k - 1) * 0.25
                ax.scatter(v, np.full(v.size, y), s=14, color=colors[g], alpha=0.5)
                ax.plot([np.nanmedian(v)] * 2, [y - 0.1, y + 0.1], color=colors[g], lw=3,
                        label=g if i == 0 else None)
        ax.set_yticks(range(len(models))); ax.set_yticklabels([m.replace("logit_", "") for m in models])
        ax.set_xlabel(f"{metric} (test members)"); ax.set_title(metric, loc="left", weight="bold"); st.tidy(ax)
    axes[0].legend(frameon=False, loc="lower right"); axes[0].invert_yaxis()
    fig.suptitle("skill by forcing group — same fits, test members split by group")
    st.save(fig, out_png)


def plot_events(table: dict, cnn_rows: list, out_png) -> None:
    """(a) event-duration histogram, (b) events per onset year, (c) event hit rate vs sample F1 per model."""
    n = 3 if cnn_rows else 2
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    d = table["duration"]
    axes[0].hist(d, bins=np.arange(0.5, d.max() + 1.5), color=st.C_ALL, edgecolor="white")
    axes[0].set_xlabel("event duration [consecutive onset years]"); axes[0].set_ylabel("events")
    axes[0].set_title(f"(a) {d.size} events from {d.sum()} positive member-years", loc="left", weight="bold")
    yrs, cnt = np.unique(table["onset"], return_counts=True)
    axes[1].bar(yrs, cnt, color=st.C_ALL, width=0.8)
    axes[1].set_xlabel("event onset year"); axes[1].set_ylabel("events (all members)")
    axes[1].set_title("(b) event onsets by year", loc="left", weight="bold")
    if cnn_rows:
        ax = axes[2]
        ax.scatter([r["f1_sample"] for r in cnn_rows], [r["hit_rate"] for r in cnn_rows], s=18, color=st.C_ALL,
                   label="hit rate")
        ax.scatter([r["f1_sample"] for r in cnn_rows], [r["false_alarm_ratio"] for r in cnn_rows], s=18,
                   color=st.C_NOSLOW, label="false-alarm ratio")
        ax.set_xlabel("sample-level F1"); ax.set_ylabel("event-level rate"); ax.set_ylim(0, 1)
        ax.legend(frameon=False); ax.set_title("(c) event vs sample skill, one point per CNN", loc="left", weight="bold")
    for ax in axes:
        st.tidy(ax)
    st.save(fig, out_png)
