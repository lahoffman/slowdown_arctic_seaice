"""xai_compare.py — composites from several attribution methods beside the occlusion result."""

from __future__ import annotations

import numpy as np

from . import maps
from . import style as st
from .style import plt

LABELS = {"lrp_z": "LRP-z", "lrp_epsilon": "LRP-ε", "lrp_a2b1": "LRP-α2β1", "deep_taylor": "DeepTaylor",
          "integrated_gradients": "Integrated gradients", "smoothgrad": "SmoothGrad",
          "input_x_gradient": "Input × gradient", "gradient": "Gradient", "shap_deep": "SHAP (Deep)"}
REG = {"arctic": "Arctic", "north_pacific": "N. Pacific", "tropical_pacific": "trop. Pacific", "north_atlantic": "N. Atlantic"}


def plot_xai_compare(comps: dict, row: dict, area: dict, lat, lon, out_png):
    """Top: TP-composite relevance per method (normalised). Bottom: region share of |relevance| per method vs area share and occlusion ΔAUROC."""
    ms = list(comps); n = len(ms); ncol = min(4, n); nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(4.6 * ncol, 2.6 * nrow + 4.2))
    gs = fig.add_gridspec(nrow + 1, ncol, height_ratios=[1] * nrow + [1.5])
    for i, m in enumerate(ms):
        ax = maps.map_axes(fig, gs[i // ncol, i % ncol])
        maps.global_map(ax, lon, lat, comps[m]["tp"], st.CMAP_LRP_SIGNED, -1, 1, cbar=False)
        ax.set_title(f"({'abcdefghij'[i]}) {LABELS.get(m, m)}", loc="left", weight="bold")
    ax = fig.add_subplot(gs[nrow, :])
    regs = list(area); x = np.arange(len(regs)); w = 0.8 / (n + 1)
    for i, m in enumerate(ms):
        ax.bar(x + (i - n / 2) * w, [row["share"][m][r] for r in regs], width=w * 0.95,
               color=st.CATEGORICAL[i % len(st.CATEGORICAL)], label=LABELS.get(m, m))
    ax.scatter(x, [area[r] for r in regs], marker="_", s=600, color=st.INK, zorder=4, label="area share (uniform relevance)")
    ax.set_xticks(x); ax.set_xticklabels([REG.get(r, r) for r in regs]); ax.set_ylabel("share of |relevance| (TP composite)")
    if row.get("occ"):
        ax2 = ax.twinx()
        ax2.scatter(x, [row["occ"].get(r, np.nan) for r in regs], marker="D", s=70, color=st.C_EVENT, zorder=5,
                    label="occlusion ΔAUROC")
        ax2.axhline(0, color=st.MUTED, lw=0.7); ax2.set_ylabel("ΔAUROC when region is zeroed", color=st.C_EVENT)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, frameon=False, ncol=4, fontsize=plt.rcParams["legend.fontsize"] - 1,
                  loc="upper center", bbox_to_anchor=(0.5, -0.18))
    else:
        ax.legend(frameon=False, ncol=4, fontsize=plt.rcParams["legend.fontsize"] - 1, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.set_title(f"({'abcdefghij'[n]}) where each method places relevance vs what occlusion says the model uses "
                 f"(split {row['split']}, seed {row['run']})", loc="left", weight="bold")
    st.tidy(ax)
    fig.tight_layout()
    if out_png is None:
        return fig
    st.save(fig, out_png)
