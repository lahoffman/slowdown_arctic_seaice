"""
07_baselines.py — scalar baselines for the slowdown CNN (revision plan §1.1).

For each of the 9 TVT splits, fit on training members and score on test
members:

  always_positive     predict slowdown everywhere (F1 = 2p/(1+p))
  random_prevalence   guess positives at the training prevalence
  year_climatology    P(slowdown | onset year) from training members
  logit_sie_anom      logistic regression on Sept SIE anomaly at onset
  logit_arctic_sst    Arctic JJA SST index alone;  logit_nino34, logit_ipo likewise
  logit_pacific       Niño3.4 + IPO;  logit_indices = Arctic SST + Niño3.4 + IPO
  logit_sie_*         SIE anomaly + one index (arctic / nino34 / ipo / pacific / year):
                      what each adds on top of the ice state
  logit_all_scalars   everything above
  cnn_run{r}          cached CNN predictions (06_cnn_predict_cesm2le.py), if present

Also regresses the cached CNN test probabilities on year-climatology and
SIE anomaly to quantify how much of the CNN output those two explain
(cnn_attribution.json).

Outputs (under RESULTS_DIR/baselines[/<tag>]/):
  baselines_split{k}.nc     metric × model per split
  baselines_all_splits.nc   stacked + across-split median
  baselines_summary.md      markdown table for the manuscript
  baselines_coefs.json      logistic coefficients per split
  FIGURES_DIR/baselines_skill[_<tag>].png

Usage:
  python scripts/07_baselines.py                                   # original labels
  python scripts/07_baselines.py --labels-file <relative-label .nc> --tag rel_w10_s1
  python scripts/07_baselines.py --n-boot 200 --no-cnn --no-fig
  # retrained configuration (step 1.5): score its cached predictions on the same axes
  python scripts/07_baselines.py --labels-file <relative .nc> --demean group --cnn-tag rel_aux --tag rel_aux
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
from src.plotting.baselines import plot_summary
from src.plotting import style as st
from src.data.cesm2le.slowdowns_relative import frequency_table

START_YEAR, END_YEAR = 1990, 2040
N_SPLITS, N_BLOCKS = 9, 10
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
    p.add_argument("--label-var", default="slowdown", choices=["slowdown", "riles"],
                   help="which binary label in the file to score (riles = rapid-ice-loss tail, z < −nσ)")
    p.add_argument("--labels-file", type=Path, default=None,
                   help="slowdown label NetCDF (default: original 02_cesm2le_slowdowns output)")
    p.add_argument("--tag", default=None,
                   help="output subdirectory / figure suffix (default: none)")
    p.add_argument("--cnn-tag", default=None,
                   help="tag of the CNN configuration whose cached predictions "
                        "(06_cnn_predict_cesm2le.py --tag) are scored (default: original)")
    p.add_argument("--demean", default="all", choices=["all", "group"],
                   help="forced response removed from the SIE anomaly predictor (default all)")
    return p.parse_args()


def main():
    st.paper_rc()
    args = parse_args()
    out_dir = OUT_DIR / args.tag if args.tag else OUT_DIR
    CNN_PRED_DIR = paths.cesm2le_predictions_dir(args.cnn_tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    label_file = args.labels_file or paths.cesm2le_slowdown_file(args.variable, args.month)
    print("07  —  Scalar baselines for slowdown classification")
    print(f"  years {args.start_year}–{args.end_year}   splits {N_SPLITS}   n_boot {args.n_boot}")
    print(f"  labels: {label_file}")
    print(f"  CNN predictions: {CNN_PRED_DIR}  (sie_anom demean={args.demean})\n")

    # 1. labels + scalar fields on the (nens, nyear) grid -----------------------
    labels, years = bl.load_labels(label_file, args.start_year, args.end_year, var=args.label_var)
    sie, sie_anom = bl.load_sie_anomaly(paths.CESM2LE_AICE_DIR / "metrics", years,
                                        month=args.month, variable=args.variable,
                                        demean=args.demean)
    fields = {"sie": sie, "sie_anom": sie_anom}
    fields.update(bl.load_climate_indices_jja(paths.CESM2LE_CLIMATE_INDICES_DIR,
                                              args.start_year, args.end_year))
    siv_anom = bl.load_siv_anomaly(paths.CESM2LE_AICE_DIR / "metrics", years, month=args.month, demean=args.demean)
    if siv_anom is not None:
        fields["siv_anom"] = siv_anom
    print(f"  labels {labels.shape}  prevalence {labels.mean():.3f}  "
          f"fields: {sorted(fields)}")
    print("  slowdown frequency by decade and forcing group (cmip6 = members 0-49, smbb = 50-99):")
    print(frequency_table(labels, years))
    print()

    # 2. per-split fits --------------------------------------------------------
    datasets, coefs, attrib = [], {}, {}
    for k, test_block, val_block, train_blocks in bl.iter_split_blocks(N_SPLITS, N_BLOCKS):
        cnn = None if args.no_cnn else bl.load_cnn_test_predictions(CNN_PRED_DIR, k)
        if cnn is not None and not cnn:
            cnn = None
        ds, extra = bl.run_split(fields, labels, years, k, test_block, val_block,
                                 train_blocks, n_boot=args.n_boot, cnn_preds=cnn)
        ds.to_netcdf(out_dir / f"baselines_split{k}.nc")
        datasets.append(ds)
        coefs[f"split{k}"] = extra["coefs"]
        if "cnn_attribution" in extra:
            attrib[f"split{k}"] = extra["cnn_attribution"]
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
    stacked.to_netcdf(out_dir / "baselines_all_splits.nc")
    md = bl.summary_markdown(stacked)
    header = (f"# Baseline skill (test members, median across {N_SPLITS} splits)\n\n"
              f"Labels: `{label_file.name}`. Always-positive F1 (analytic, median) = {stacked.attrs['always_positive_f1_median']:.3f}; "
              f"random-at-prevalence F1 95th pct = {stacked.attrs['random_f1_p95_median']:.3f}.\n\n")
    (out_dir / "baselines_summary.md").write_text(header + md + "\n")
    (out_dir / "baselines_coefs.json").write_text(json.dumps(coefs, indent=2))
    print("\n" + header + md)
    if attrib:
        (out_dir / "cnn_attribution.json").write_text(json.dumps(attrib, indent=2))
        print("\n" + bl.attribution_markdown(attrib))
    print(f"\n  outputs → {out_dir}")

    if not args.no_fig:
        suffix = f"_{args.tag}" if args.tag else ""
        plot_summary(stacked, paths.FIGURES_DIR / f"baselines_skill{suffix}.png")


if __name__ == "__main__":
    main()
