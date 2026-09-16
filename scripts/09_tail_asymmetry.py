#!/usr/bin/env python3
"""
09_tail_asymmetry.py — slowdowns and rapid ice loss events as the two tails of one regression.

Reads the pooled state fit from 09_residual_analysis.py (results/residual/w<w>_off<o>/residual_pooled.nc):
trend anomaly, its fit on the ice state, and the residual. Reports, per tail (|z| > n_sigma), how well the
state-based prediction separates the tail from the rest (AUROC), how far predicted values reach into the
tail, and the skewness of the residual.

Outputs: results/residual/w<w>_off<o>/tail_asymmetry.md, FIGURES_DIR/diagnostics/tail_asymmetry_w<w>_off<o>.png

Usage:
  python scripts/09_tail_asymmetry.py                       # w10_off1
  python scripts/09_tail_asymmetry.py --key w10_off1 --n-sigma 1.0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import stats
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--key", default="w10_off1", help="results/residual/<key>")
    p.add_argument("--n-sigma", type=float, default=1.0)
    p.add_argument("--no-fig", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    d = paths.RESULTS_DIR / "residual" / a.key
    with xr.open_dataset(d / "residual_pooled.nc") as ds:
        y, fit, res = ds["trend_anom"].values.ravel(), ds["fit_state"].values.ravel(), ds["residual"].values.ravel()
        state = ds.attrs.get("state", "sie_anom"); r2 = float(ds.attrs.get("r2_pooled", np.nan))
    ok = np.isfinite(y) & np.isfinite(fit)
    y, fit, res = y[ok], fit[ok], res[ok]
    sig = y.std(); z, zf = y / sig, fit / sig
    tails = {"slowdown (z > +%g)" % a.n_sigma: z > a.n_sigma, "rapid loss (z < −%g)" % a.n_sigma: z < -a.n_sigma}
    lines = [f"# Tails of the decadal trend anomaly ({a.key}); state = {state}, pooled R² {r2:.3f}\n",
             f"{y.size} member-years; σ(trend anomaly) = {sig:.3f} Mkm² yr⁻¹; residual skewness {stats.skew(res):+.2f}, "
             f"trend skewness {stats.skew(y):+.2f}.\n",
             "| tail | n | prevalence | AUROC of state fit | median predicted z in tail | share of tail with predicted |z| > 0.5 |",
             "|---|---|---|---|---|---|"]
    rows = {}
    for name, m in tails.items():
        sgn = 1 if "slow" in name else -1
        au = roc_auc_score(m.astype(int), sgn * zf)
        rows[name] = dict(n=int(m.sum()), prev=m.mean(), auroc=au, med=float(np.median(sgn * zf[m])),
                          share=float(np.mean(sgn * zf[m] > 0.5)))
        lines.append(f"| {name} | {m.sum()} | {m.mean():.3f} | {au:.3f} | {sgn * np.median(sgn * zf[m]):+.2f} | {rows[name]['share']:.2f} |")
    lines.append("\nAUROC uses the continuous state-based prediction as the score for each tail separately; "
                 "0.5 = the state does not distinguish that tail from the rest.")
    md = "\n".join(lines) + "\n"; (d / "tail_asymmetry.md").write_text(md); print(md)

    if a.no_fig:
        return
    from src.plotting import style as st
    from src.plotting.style import plt
    st.paper_rc()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    bins = np.linspace(-4, 4, 41)
    ax.hist(z, bins, color=st.MUTED, alpha=0.6, label="trend anomaly")
    ax.hist(res / sig, bins, histtype="step", color=st.BLUE, lw=1.8, label=f"residual after {state}")
    for v in (-a.n_sigma, a.n_sigma):
        ax.axvline(v, color=st.INK, lw=0.8, ls="--")
    ax.set_xlabel("decadal trend anomaly / σ"); ax.set_ylabel("member-years"); ax.legend(frameon=False)
    st.panel_label(ax, "(a) both tails in one regression"); st.tidy(ax)
    ax = axes[1]
    names = list(rows); au = [rows[k]["auroc"] for k in names]
    ax.bar(names, au, color=[st.C_SLOW, st.C_NOSLOW]); ax.axhline(0.5, color=st.INK, lw=0.8, ls="--")
    ax.set_ylim(0.4, 0.9); ax.set_ylabel("AUROC of the state-based prediction")
    st.panel_label(ax, "(b) how well the ice state picks each tail"); st.tidy(ax)
    out = paths.FIGURES_DIR / "diagnostics" / f"tail_asymmetry_{a.key}.png"
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig); print(f"  figure → {out}")


if __name__ == "__main__":
    main()
