"""
09_baselines_by_group.py — does skill differ between the two forcing groups? (step 4.4)

Same fits as 07_baselines.py (training on all 80 members), but test members
are scored separately for CMIP6-BB (0–49) and SMBB (50–99). If the group-wise
demeaning has removed the biomass-burning artefact, both columns agree.
Cached CNN predictions of a tag are scored the same way when --cnn-tag is given.

Outputs: results/baselines[/<cnn-tag>]/by_group.nc, by_group_summary.md,
         FIGURES_DIR/diagnostics/baselines_by_group[_<tag>].png

Usage:
  python scripts/09_baselines_by_group.py --cnn-tag rel_base
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths
from src.analysis import baselines as bl, sweep

MODELS = ["logit_sie_anom", "logit_arctic_sst", "logit_indices", "logit_sie_pacific"]
GROUPS = {"all": lambda m: np.ones_like(m, bool), "cmip6": lambda m: m < 50, "smbb": lambda m: m >= 50}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels-file", type=Path, default=paths.CESM2LE_SLOWDOWNS_DIR /
                   "cesm2le_sie_slowdown_relative_SEP_w10_s1_group_1990-2100.nc")
    p.add_argument("--cnn-tag", default=None)
    p.add_argument("--start-year", type=int, default=1990); p.add_argument("--end-year", type=int, default=2030)
    p.add_argument("--no-fig", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    out_dir = paths.RESULTS_DIR / "baselines" / a.cnn_tag if a.cnn_tag else paths.RESULTS_DIR / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    labels, years = bl.load_labels(a.labels_file, a.start_year, a.end_year)
    _, sie_anom = bl.load_sie_anomaly(paths.CESM2LE_AICE_DIR / "metrics", years, demean="group")
    fields = {"sie_anom": sie_anom, **bl.load_climate_indices_jja(paths.CESM2LE_CLIMATE_INDICES_DIR,
                                                                   a.start_year, a.end_year)}
    cnn = None
    if a.cnn_tag:
        cnn = {k: bl.load_cnn_test_predictions(paths.cesm2le_predictions_dir(a.cnn_tag), k) for k in range(9)}
    print(f"09  —  baselines by forcing group  (cnn tag {a.cnn_tag or 'none'})")
    parts = []
    for g, f in GROUPS.items():
        sc = sweep.score_models(fields, labels, years, MODELS, member_filter=f, cnn_preds=cnn)
        parts.append(sc.expand_dims(group=[g]))
    ds = xr.concat(parts, dim="group"); ds.to_netcdf(out_dir / "by_group.nc")
    med = ds["value"].median("split")
    models = list(ds["model"].values)
    lines = ["| model | " + " | ".join(f"{g} AUROC" for g in GROUPS) + " | " +
             " | ".join(f"{g} F1" for g in GROUPS) + " |", "|---|" + "---|" * (2 * len(GROUPS))]
    for m in models:
        lines.append(f"| {m} | " + " | ".join(f"{float(med.sel(group=g, metric='AUROC', model=m)):.3f}" for g in GROUPS)
                     + " | " + " | ".join(f"{float(med.sel(group=g, metric='F1', model=m)):.3f}" for g in GROUPS) + " |")
    prev = " / ".join(f"{g} {float(ds['prevalence'].sel(group=g).mean()):.3f}" for g in GROUPS)
    md = f"# Skill by forcing group (test members; median over 9 splits)\n\nprevalence: {prev}\n\n" + "\n".join(lines) + "\n"
    (out_dir / "by_group_summary.md").write_text(md); print("\n" + md)
    if not a.no_fig:
        from src.plotting import sensitivity as plot, style as st
        st.paper_rc()
        suffix = f"_{a.cnn_tag}" if a.cnn_tag else ""
        plot.plot_by_group(ds, paths.FIGURES_DIR / "diagnostics" / f"baselines_by_group{suffix}.png")


if __name__ == "__main__":
    main()
