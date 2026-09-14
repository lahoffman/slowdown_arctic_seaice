"""interannual.py — figure for the interannual / onset-year sanity checks (step 8.6)."""

from __future__ import annotations

import numpy as np

from . import maps
from . import style as st
from .style import plt


def plot_interannual(member_r: dict, running: dict, run_years, lag_r2: dict, off_r2: dict,
                     corr_map, lat, lon, out_png):
    """
    (a) per-member corr(SIE anomaly, JJA index); (b) 30-yr running correlation, member spread;
    (c) test R² of SIE(t) ~ SIE(t−1) [+ indices]; (d) R² of the decadal trend anomaly on SIE(t)
    when the window starts at t, t+1, t+2; (e) corr(SIE anomaly, concurrent JJA SST).
    """
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1])
    cols = {"nino34": st.C_ORIG, "ipo": st.C_SMBB, "arctic": st.C_CMIP6}
    names = {"nino34": "Niño 3.4", "ipo": "IPO", "arctic": "Arctic SST"}

    ax = fig.add_subplot(gs[0, 0])
    for k, r in member_r.items():
        ax.hist(r, bins=np.linspace(-0.6, 0.6, 25), alpha=0.55, color=cols[k], label=f"{names[k]} (median {np.median(r):+.2f})")
    ax.axvline(0, color=st.MUTED, lw=0.8); ax.set_xlabel("corr(Sept SIE anomaly, JJA index), per member")
    ax.set_ylabel("members"); ax.legend(frameon=False, fontsize=plt.rcParams["legend.fontsize"] - 1)
    ax.set_title("(a) interannual link", loc="left", weight="bold"); st.tidy(ax)

    ax = fig.add_subplot(gs[0, 1])
    for k, r in running.items():
        lo, hi = np.percentile(r, [10, 90], axis=0)
        ax.fill_between(run_years, lo, hi, color=cols[k], alpha=0.2)
        ax.plot(run_years, np.median(r, 0), color=cols[k], lw=2, label=names[k])
    ax.axhline(0, color=st.MUTED, lw=0.8); ax.set_xlabel("centre year of 30-yr window"); ax.set_ylabel("correlation")
    ax.legend(frameon=False); ax.set_title("(b) 30-yr running corr.", loc="left", weight="bold"); st.tidy(ax)

    ax = fig.add_subplot(gs[0, 2])
    keys = list(lag_r2); base = np.nanmedian(lag_r2["persistence"])
    for i, k in enumerate(keys):
        v = lag_r2[k]; c = st.C_ALL if k == "persistence" else cols.get(k.split("+")[1], st.INK)
        ax.bar(i, np.nanmedian(v), color=c, alpha=0.85); ax.scatter(np.full(v.size, i), v, s=10, color=st.INK, alpha=0.5)
    ax.axhline(base, color=st.MUTED, lw=0.8, ls="--")
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(["SIE(t−1)"] + ["+ " + k.split("+")[1] for k in keys[1:]], rotation=30, ha="right")
    ax.set_ylabel("test R² of SIE(t)"); ax.set_title("(c) year-ahead SIE", loc="left", weight="bold"); st.tidy(ax)

    ax = fig.add_subplot(gs[1, 0])
    offs = list(off_r2)
    for i, o in enumerate(offs):
        v = off_r2[o]
        ax.bar(i, np.nanmedian(v), color=st.C_ALL, alpha=0.85); ax.scatter(np.full(v.size, i), v, s=10, color=st.INK, alpha=0.5)
    ax.set_xticks(range(len(offs))); ax.set_xticklabels([f"t{'+' + str(o) if o else ''} … t+{o + 9}" for o in offs])
    ax.set_xlabel("trend window"); ax.set_ylabel("test R² of trend anomaly on SIE(t)")
    ax.set_title("(d) onset year in window?", loc="left", weight="bold"); st.tidy(ax)

    axm = maps.map_axes(fig, gs[1, 1:])
    maps.global_map(axm, lon, lat, corr_map, st.CMAP_SST, -0.4, 0.4, "correlation")
    axm.set_title("(e) corr(Sept SIE anomaly, concurrent JJA SST)", loc="left", weight="bold")
    fig.tight_layout()
    if out_png is None:
        return fig
    st.save(fig, out_png)
