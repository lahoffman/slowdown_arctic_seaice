"""regression.py — figure for the CNN regression test (step 8.7)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from . import style as st
from .style import plt


def plot_regression(ds: xr.Dataset, pred_dir: Path, out_png):
    """(a) test R² per model: CNN vs OLS references, by split; (b) CNN prediction vs truth, best split."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    splits = sorted(set(ds["split"].values.tolist()))
    for i, k in enumerate(splits):
        m = ds["split"].values == k
        ax.scatter(np.full(m.sum(), i), ds["r2_cnn"].values[m], color=st.C_ALL, s=28, zorder=3,
                   label="CNN (each seed)" if i == 0 else None)
        ax.hlines(ds["r2_sie"].values[m][0], i - 0.3, i + 0.3, color=st.INK, lw=2.2, label="OLS: SIE anomaly" if i == 0 else None)
        ax.hlines(ds["r2_sie_pacific"].values[m][0], i - 0.3, i + 0.3, color=st.C_ORIG, lw=2.2, ls="--",
                  label="OLS: SIE + Niño 3.4 + IPO" if i == 0 else None)
    ax.axhline(0, color=st.MUTED, lw=0.7)
    ax.set_xticks(range(len(splits))); ax.set_xticklabels([f"split {k}" for k in splits])
    ax.set_ylabel("test R² of trend anomaly"); ax.legend(frameon=False, loc="lower right")
    ax.set_title("(a) map + scalar CNN vs linear ice-state references", loc="left", weight="bold"); st.tidy(ax)

    ax = axes[1]
    best = ds.isel(model=int(np.argmax(ds["r2_cnn"].values)))
    f = np.load(Path(pred_dir) / f"pred_split{int(best['split'])}_run{int(best['run'])}.npz")
    ax.scatter(f["y_true"], f["y_pred"], s=8, color=st.C_ALL, alpha=0.5, label=f"CNN, R² {float(best['r2_cnn']):.2f}")
    b = np.polyfit(f["sie"], f["y_true"], 1)
    ax.scatter(f["y_true"], np.polyval(b, f["sie"]), s=8, color=st.INK, alpha=0.35, label=f"OLS on SIE, R² {float(best['r2_sie']):.2f}")
    lim = np.nanpercentile(np.abs(f["y_true"]), 99.5)
    ax.plot([-lim, lim], [-lim, lim], color=st.MUTED, lw=0.8); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("true trend anomaly [M km² yr⁻¹]"); ax.set_ylabel("predicted [M km² yr⁻¹]")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title(f"(b) best model, split {int(best['split'])} seed {int(best['run'])}, test members", loc="left", weight="bold"); st.tidy(ax)
    fig.tight_layout()
    if out_png is None:
        return fig
    st.save(fig, out_png)
