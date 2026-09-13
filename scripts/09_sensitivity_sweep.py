"""
09_sensitivity_sweep.py — label-definition sensitivity (step 4.2), baselines only.

Relative labels for every trend window × σ threshold, onsets 1990–2030, scored
with the scalar baselines on the 9 block splits. No CNN. Shows whether the
"SIE anomaly beats the indices" result depends on the 10-yr / 1σ choice.

Outputs: results/sensitivity/sweep.nc, sweep_summary.md,
         FIGURES_DIR/diagnostics/sensitivity_sweep.png

Usage:
  python scripts/09_sensitivity_sweep.py
  python scripts/09_sensitivity_sweep.py --windows 8 10 12 15 --sigmas 0.5 1 1.5
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
from src.data.cesm2le import slowdowns_relative as rel
from src.data.cesm2le.slowdowns import load_sie_monthly_files

MODELS = ["logit_sie_anom", "logit_indices", "logit_sie_pacific", "logit_all_scalars"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--windows", type=int, nargs="+", default=[8, 10, 12, 15])
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    p.add_argument("--start-year", type=int, default=1990); p.add_argument("--end-year", type=int, default=2030)
    p.add_argument("--no-fig", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    out_dir = paths.RESULTS_DIR / "sensitivity"; out_dir.mkdir(parents=True, exist_ok=True)
    years = np.arange(a.start_year, a.end_year + 1)
    sie_full, yrs_full = load_sie_monthly_files(str(paths.CESM2LE_AICE_DIR / "metrics"), "SEP",
                                                variable="sie", start_year=1990, end_year=2100)
    _, sie_anom = bl.load_sie_anomaly(paths.CESM2LE_AICE_DIR / "metrics", years, demean="group")
    fields = {"sie_anom": sie_anom, **bl.load_climate_indices_jja(paths.CESM2LE_CLIMATE_INDICES_DIR,
                                                                   a.start_year, a.end_year)}
    print(f"09  —  sensitivity sweep  windows {a.windows}  σ {a.sigmas}  onsets {a.start_year}–{a.end_year}")
    res = []
    for w in a.windows:
        for s in a.sigmas:
            ds = rel.build_relative_dataset(sie_full, yrs_full, window=w, start_year=1990, n_sigma=s,
                                            demean="group", pool_years=(1990, 2040), sigma_mode="pooled")
            lab = ds["slowdown"].sel(nyr=slice(a.start_year, a.end_year)).values.astype(int)
            sc = sweep.score_models(fields, lab, years, MODELS)
            sc = sc.expand_dims(window=[w], n_sigma=[s]); res.append(sc)
            med = sc["value"].median("split")
            print(f"  w={w:2d} σ={s:3.1f}  prev {float(sc['prevalence'].mean()):.3f}  AUROC: " +
                  "  ".join(f"{m.replace('logit_', '')} {float(med.sel(metric='AUROC', model=m).squeeze()):.3f}" for m in MODELS), flush=True)
    full = xr.combine_by_coords(res); full.to_netcdf(out_dir / "sweep.nc")
    med = full["value"].median("split")
    lines = ["| window | σ | prevalence | " + " | ".join(m.replace("logit_", "") + " AUROC" for m in MODELS) + " | Δ(sie_pacific − sie_anom) |",
             "|---|---|---|" + "---|" * (len(MODELS) + 1)]
    for w in a.windows:
        for s in a.sigmas:
            v = med.sel(window=w, n_sigma=s, metric="AUROC")
            lines.append(f"| {w} | {s:g} | {float(full['prevalence'].sel(window=w, n_sigma=s).mean()):.3f} | " +
                         " | ".join(f"{float(v.sel(model=m)):.3f}" for m in MODELS) +
                         f" | {float(v.sel(model='logit_sie_pacific') - v.sel(model='logit_sie_anom')):+.3f} |")
    md = "# Label sensitivity (baselines only; test AUROC, median over 9 splits)\n\n" + "\n".join(lines) + "\n"
    (out_dir / "sweep_summary.md").write_text(md); print("\n" + md)
    if not a.no_fig:
        from src.plotting import sensitivity as plot, style as st
        st.paper_rc()
        plot.plot_sweep(full, MODELS, paths.FIGURES_DIR / "diagnostics" / "sensitivity_sweep.png")


if __name__ == "__main__":
    main()
