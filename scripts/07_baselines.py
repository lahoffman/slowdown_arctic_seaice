"""
07_baselines.py — scalar baselines for the slowdown CNN (revision plan §1.1).

For each of the 9 TVT splits, fit on training members and score on test
members:

  always_positive     predict slowdown everywhere (F1 = 2p/(1+p))
  random_prevalence   guess positives at the training prevalence
  year_climatology    P(slowdown | onset year) from training members
  logit_sie_anom      logistic regression on Sept SIE anomaly at onset
  logit_arctic_sst    logistic regression on the Arctic JJA SST index
  logit_indices       Arctic SST + Niño3.4 + IPO
  logit_sie_arctic    SIE anomaly + Arctic SST
  logit_sie_year      SIE anomaly + year climatology (no ocean variability)
  logit_all_scalars   everything above
  cnn_run{r}          cached CNN predictions (06_cnn_predict_cesm2le.py), if present

Outputs (under RESULTS_DIR/baselines/):
  baselines_split{k}.nc     metric × model per split
  baselines_all_splits.nc   stacked + across-split median
  baselines_summary.md      markdown table for the manuscript
  baselines_coefs.json      logistic coefficients per split
  FIGURES_DIR/baselines_skill.png

Usage:
  python scripts/07_baselines.py
  python scripts/07_baselines.py --n-boot 200 --no-cnn --no-fig
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths
from src.analysis import baselines as bl

START_YEAR, END_YEAR = 1990, 2040
N_SPLITS, N_BLOCKS = 9, 10
CNN_PRED_DIR = paths.RESULTS_DIR / "predictions" / "cesm2le"
OUT_DIR = paths.RESULTS_DIR / "baselines"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start-year", type=int, default=START_YEAR)
    p.add_argument("--end-year", type=int, default=END_YEAR)
    p.add_argument("--n-boot", type=int, default=1000,
                   help="bootstrap / random-classifier draws (default 1000)")
    p.add_argument("--no-cnn", action="store_true",
                   help="skip cached CNN predictions even if present")
    p.add_argument("--no-fig", action="store_true", help="skip the summary figure")
    p.add_argument("--variable", default="sie", choices=["sie", "sia"])
    p.add_argument("--month", default="SEP")
    return p.parse_args()


def plot_summary(stacked, out_png: Path) -> None:
    """Strip plot of per-split test skill for every model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["F1", "AUPRC", "AUROC"]
    models = [m for m in stacked.model.values if not m.startswith("cnn_run")]
    order = [m for m in bl.BASELINE_FEATURES if m in models] + \
            [m for m in models if m not in bl.BASELINE_FEATURES]
    ink, muted, accent = "#1f2933", "#8a949e", "#0072B2"

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.6), sharey=True)
    for ax, met in zip(axes, metrics):
        for i, m in enumerate(order):
            v = stacked["metric_value"].sel(metric=met, model=m).values
            ax.scatter(v, np.full(v.size, i) + np.random.default_rng(i).uniform(-0.15, 0.15, v.size),
                       s=18, color=accent if m.startswith("cnn") else muted, alpha=0.8, zorder=3)
            ax.plot([np.median(v)] * 2, [i - 0.3, i + 0.3], color=ink, lw=2, zorder=4)
        if met == "F1":
            ax.axvline(float(stacked.attrs.get("always_positive_f1_median", np.nan)),
                       color=ink, ls=":", lw=1, label="always-positive")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=9)
        ax.set_xlabel(met + " (test)")
        ax.grid(axis="x", color="#e5e8eb", lw=0.8, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle("Baselines vs CNN — per-split test skill (bar = median across 9 splits)", fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    print(f"  figure → {out_png}")


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("07  —  Scalar baselines for slowdown classification")
    print(f"  years {args.start_year}–{args.end_year}   splits {N_SPLITS}   n_boot {args.n_boot}\n")

    # 1. labels + scalar fields on the (nens, nyear) grid -----------------------
    labels, years = bl.load_labels(paths.cesm2le_slowdown_file(args.variable, args.month),
                                   args.start_year, args.end_year)
    sie, sie_anom = bl.load_sie_anomaly(paths.CESM2LE_AICE_DIR / "metrics", years,
                                        month=args.month, variable=args.variable)
    fields = {"sie": sie, "sie_anom": sie_anom}
    fields.update(bl.load_climate_indices_jja(paths.CESM2LE_CLIMATE_INDICES_DIR,
                                              args.start_year, args.end_year))
    print(f"  labels {labels.shape}  prevalence {labels.mean():.3f}  "
          f"fields: {sorted(fields)}")
    print(f"  slowdown frequency by onset year (all members):")
    freq = labels.mean(0)
    for y0 in range(args.start_year, args.end_year + 1, 10):
        sel = (years >= y0) & (years < y0 + 10)
        print(f"    {y0}–{min(y0 + 9, args.end_year)}: {freq[sel].mean():.2f}")

    # 2. per-split fits --------------------------------------------------------
    datasets, coefs = [], {}
    for k, test_block, val_block, train_blocks in bl.iter_split_blocks(N_SPLITS, N_BLOCKS):
        cnn = None if args.no_cnn else bl.load_cnn_test_predictions(CNN_PRED_DIR, k)
        if cnn is not None and not cnn:
            cnn = None
        ds, extra = bl.run_split(fields, labels, years, k, test_block, val_block,
                                 train_blocks, n_boot=args.n_boot, cnn_preds=cnn)
        ds.to_netcdf(OUT_DIR / f"baselines_split{k}.nc")
        datasets.append(ds)
        coefs[f"split{k}"] = extra["coefs"]
        f1 = ds["metric_value"].sel(metric="F1")
        print(f"  split {k}: F1  always+ {float(f1.sel(model='always_positive')):.3f}"
              f"  year {float(f1.sel(model='year_climatology')):.3f}"
              f"  sie {float(f1.sel(model='logit_sie_anom')):.3f}"
              f"  arctic {float(f1.sel(model='logit_arctic_sst')):.3f}"
              + (f"  cnn {float(f1.sel(model='cnn_median')):.3f}" if "cnn_median" in ds.model.values else ""))

    # 3. summary ----------------------------------------------------------------
    stacked = bl.summarise(datasets)
    stacked.attrs["always_positive_f1_median"] = float(
        np.median([d.attrs["always_positive_f1_analytic"] for d in datasets]))
    stacked.attrs["random_f1_p95_median"] = float(
        np.median([d.attrs["random_f1_p95"] for d in datasets]))
    stacked.to_netcdf(OUT_DIR / "baselines_all_splits.nc")
    md = bl.summary_markdown(stacked)
    header = (f"# Baseline skill (test members, median across {N_SPLITS} splits)\n\n"
              f"Always-positive F1 (analytic, median) = {stacked.attrs['always_positive_f1_median']:.3f}; "
              f"random-at-prevalence F1 95th pct = {stacked.attrs['random_f1_p95_median']:.3f}.\n\n")
    (OUT_DIR / "baselines_summary.md").write_text(header + md + "\n")
    (OUT_DIR / "baselines_coefs.json").write_text(json.dumps(coefs, indent=2))
    print("\n" + header + md)
    print(f"\n  outputs → {OUT_DIR}")

    if not args.no_fig:
        plot_summary(stacked, paths.FIGURES_DIR / "baselines_skill.png")


if __name__ == "__main__":
    main()
