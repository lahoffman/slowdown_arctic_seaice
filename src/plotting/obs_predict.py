"""obs_predict.py — observed scalars and the baseline probabilities they imply (AIES Fig. 8b)."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from . import style as st
from .style import plt

LABELS = {"logit_sie_anom": "logistic: SIE anomaly", "logit_ipo": "logistic: IPO",
          "logit_sie_ipo": "logistic: SIE + IPO", "logit_sie_pacific": "logistic: SIE + Niño3.4 + IPO"}


def plot_obs_predict(years, obs: Dict[str, np.ndarray], probs: Dict[str, np.ndarray], z: np.ndarray,
                     frac: Optional[np.ndarray], forced_method: str, out_png, base_rate: float = 0.17,
                     cnn_label: str = "CNN votes") -> None:
    """(a) observed SIE anomaly, (b) JJA IPO and Niño3.4, (c) P(slowdown in t+1…t+10) per baseline (+CNN) with observed z."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 9.5), sharex=True, gridspec_kw=dict(height_ratios=[1, 1, 1.6]))
    a, b, c = axes
    a.bar(years, obs["sie_anom"], color=np.where(obs["sie_anom"] > 0, st.C_POS, st.C_NEG), alpha=0.8, width=0.8)
    a.axhline(0, color=st.INK, lw=0.6); a.set_ylabel("Sept SIE anomaly\n[10⁶ km²]")
    st.panel_label(a, f"(a) observed September SIE minus forced reference ({forced_method})")

    b.plot(years, obs["ipo"], color=st.C_EVENT, lw=2, label="IPO (filtered), JJA")
    b.plot(years, obs["nino34"], color=st.SKY, lw=1.4, label="Niño3.4, JJA")
    b.axhline(0, color=st.INK, lw=0.6); b.set_ylabel("index [°C]")
    b.legend(loc="upper left", ncol=2, frameon=False)
    st.panel_label(b, "(b) observed JJA Pacific indices (ERSSTv5)")

    for i, (name, P) in enumerate(probs.items()):
        col = st.CATEGORICAL[i % len(st.CATEGORICAL)]
        c.fill_between(years, np.nanmin(P, 0), np.nanmax(P, 0), color=col, alpha=0.15, lw=0)
        c.plot(years, np.nanmedian(P, 0), color=col, lw=2, label=LABELS.get(name, name))
    if frac is not None:
        c.plot(years, frac, color=st.INK, lw=2, ls="--", label=cnn_label)
    c.axhline(base_rate, color=st.MUTED, ls=":", lw=1, label=f"base rate {base_rate:.2f}")
    ok = np.isfinite(z)
    c2 = c.twinx()
    c2.scatter(years[ok], z[ok], marker="D", s=45, color=np.where(z[ok] > 1, st.C_EVENT, st.MUTED), zorder=5,
               edgecolor=st.INK, lw=0.5)
    c2.axhline(1, color=st.C_EVENT, lw=0.8, ls="--"); c2.set_ylabel("observed offset-window trend z", color=st.C_EVENT)
    c2.set_ylim(-3, 3)
    c.set_ylim(0, 1); c.set_ylabel("P(slowdown in t+1…t+10)"); c.set_xlabel("onset year t")
    c.legend(loc="upper left", ncol=2, frameon=False, fontsize=11)
    st.panel_label(c, "(c) baseline probabilities from CESM2-LE fits (band: 9 split-fits); diamonds: observed outcome (z > 1 = slowdown)")
    for ax in axes:
        ax.grid(False)
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)
    print(f"  figure → {out_png}")
