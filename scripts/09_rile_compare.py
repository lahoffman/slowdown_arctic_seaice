#!/usr/bin/env python3
"""
09_rile_compare.py — slowdowns vs rapid ice loss events (RILEs), scalar baselines only (step 8.11).

Reads the 07_baselines.py outputs for the tags slow_w{w}_off1 / rile_w{w}_off1 (window w in
years, offset labels: slowdown = trend anomaly > +1σ, RILE = < −1σ) and draws one figure:
test AUROC and AUPRC of every scalar baseline, both tails side by side, one column per window.

Run first (minutes, CPU):
  for w in 3 5 10; do
    L=$SLOWDOWN_DATA_ROOT/cesm2le/slowdowns/cesm2le_sie_slowdown_relative_SEP_w${w}_s1_group_off1_1990-2100.nc
    for v in slowdown riles; do
      python scripts/07_baselines.py --labels-file $L --label-var $v --demean group --start-year 1990 --end-year 2029 \
             --tag ${v/slowdown/slow}_w${w}_off1 --no-cnn --no-fig
    done
  done
Then:  python scripts/09_rile_compare.py --windows 3 5 10
Outputs: results/sensitivity/rile_vs_slowdown.md, FIGURES_DIR/diagnostics/rile_vs_slowdown.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths

MODELS = ["logit_sie_anom", "logit_siv", "logit_sie_siv", "logit_arctic_sst", "logit_nino34", "logit_ipo",
          "logit_pacific", "logit_indices", "logit_sie_ipo", "logit_sie_pacific", "logit_all_scalars"]
NAMES = {"logit_sie_anom": "SIE", "logit_siv": "volume", "logit_sie_siv": "SIE + volume", "logit_arctic_sst": "Arctic SST",
         "logit_nino34": "Niño 3.4", "logit_ipo": "IPO", "logit_pacific": "Niño 3.4 + IPO", "logit_indices": "3 indices",
         "logit_sie_ipo": "SIE + IPO", "logit_sie_pacific": "SIE + Pacific", "logit_all_scalars": "all scalars"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--windows", type=int, nargs="+", default=[3, 5, 10]); p.add_argument("--suffix", default="off1")
    p.add_argument("--no-fig", action="store_true")
    a = p.parse_args()
    data = {}
    for w in a.windows:
        for tail, tag in (("slowdown", f"slow_w{w}_{a.suffix}"), ("rile", f"rile_w{w}_{a.suffix}")):
            f = paths.RESULTS_DIR / "baselines" / tag / "baselines_all_splits.nc"
            if not f.exists():
                print(f"  [skip] {f} — run 07_baselines.py --tag {tag}"); continue
            data[(w, tail)] = xr.open_dataset(f).load()
    if not data:
        sys.exit("nothing to plot")
    lines = ["# Slowdowns vs RILEs — scalar baselines, test AUROC (median over 9 splits), offset labels\n",
             "| window | tail | prevalence | " + " | ".join(NAMES[m] for m in MODELS) + " |", "|---|---|---|" + "---|" * len(MODELS)]
    for (w, tail), ds in sorted(data.items()):
        models = [m for m in MODELS if m in ds.model.values]
        prev = float(ds["metric_median"].sel(metric="AUPRC", model="random_prevalence")) if "random_prevalence" in ds.model.values else np.nan
        lines.append(f"| {w} | {tail} | {prev:.3f} | " + " | ".join(
            f"{float(ds['metric_median'].sel(metric='AUROC', model=m)):.3f}" if m in models else "—" for m in MODELS) + " |")
    md = "\n".join(lines) + "\n"
    out = paths.RESULTS_DIR / "sensitivity"; out.mkdir(parents=True, exist_ok=True)
    (out / "rile_vs_slowdown.md").write_text(md); print(md)
    if not a.no_fig:
        from src.plotting import sensitivity as plot, style as st
        st.paper_rc()
        plot.plot_rile_vs_slowdown(data, a.windows, MODELS, NAMES, paths.FIGURES_DIR / "diagnostics" / "rile_vs_slowdown.png")


if __name__ == "__main__":
    main()
