#!/usr/bin/env python3
"""
10_obs_baseline_predict.py — what the scalar baselines say about the observed record (AIES §8, Fig. 8b).

Logistic baselines (SIE anomaly, IPO, SIE+IPO, SIE+Niño3.4+IPO) fitted on CESM2-LE
with the given labels (offset labels by default), applied to observed September SIE
(NSIDC, forced part removed as in the CNN pipeline) and ERSSTv5 JJA indices, year by
year. Where the following decade is observed, the observed offset-window trend z is
computed for verification. CNN vote fractions for --tag are overlaid when present.

Outputs: results/obs_predict/<key>/obs_predict_<forced>.nc, summary_<forced>.md,
         FIGURES_DIR/diagnostics/obs_predict_<key>_<forced>.png      (key = w<window>_off<offset>)

Usage:
  python scripts/10_obs_baseline_predict.py --labels-file $LBL1 --forced ensmean group_smbb linear
  python scripts/10_obs_baseline_predict.py --labels-file $LBL1 --forced ensmean --tag off1 --product oisst
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths
from src.analysis import baselines as bl
from src.analysis import obs_predict as op


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels-file", type=Path, required=True)
    p.add_argument("--forced", nargs="+", default=["ensmean"], help="forced references: ensmean group_cmip6 group_smbb linear")
    p.add_argument("--start-year", type=int, default=1990); p.add_argument("--end-year", type=int, default=2029,
                   help="last onset year used to FIT (must match the label file's cap)")
    p.add_argument("--obs-start", type=int, default=1990); p.add_argument("--obs-end", type=int, default=2025)
    p.add_argument("--demean", default="group", choices=["all", "group"])
    p.add_argument("--tag", default=None, help="CNN tag whose observational predictions to overlay (if present)")
    p.add_argument("--product", default="ersst", choices=["ersst", "oisst"], help="product for the CNN overlay")
    p.add_argument("--no-fig", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    with xr.open_dataset(a.labels_file) as ds:
        window, offset = int(ds.attrs.get("window", 10)), int(ds.attrs.get("trend_offset", 0))
    key = f"w{window}_off{offset}"
    out_dir = paths.RESULTS_DIR / "obs_predict" / key; out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = paths.CESM2LE_AICE_DIR / "metrics"
    nsidc = paths.nsidc_sie_slowdown_events(9)

    # model fits
    labels, years = bl.load_labels(a.labels_file, a.start_year, a.end_year)
    _, sie_anom = bl.load_sie_anomaly(metrics_dir, years, demean=a.demean)
    fields = {"sie_anom": sie_anom, **bl.load_climate_indices_jja(paths.CESM2LE_CLIMATE_INDICES_DIR, a.start_year, a.end_year)}
    fits = op.fit_predictors(fields, labels, years)
    print(f"labels {key}: {labels.shape}, prevalence {labels.mean():.3f}; fitted {list(fits)}")

    obs_years = np.arange(a.obs_start, a.obs_end + 1)
    slow_years = op.observed_slowdown_years(nsidc)
    for fm in a.forced:
        print(f"\n== forced reference: {fm}")
        obs = op.observed_scalars(obs_years, fm, nsidc, metrics_dir, paths.ERSST_IPO, paths.ERSST_NINO34)
        probs = op.predict_observed(fits, obs)
        z = op.observed_offset_z(obs_years, fm, nsidc, metrics_dir, a.labels_file, window, offset)
        frac = op.cnn_vote_fraction(paths.obs_predictions_dir(a.product, fm, a.tag), obs_years) if a.tag else None
        if a.tag and frac is None:
            print(f"  [note] no CNN predictions for tag {a.tag} / {a.product} / {fm} — run 03_obs_test + 06_cnn_predict_obs")
        md = op.summary_markdown(obs_years, obs, probs, z, frac, fm)
        (out_dir / f"summary_{fm}.md").write_text(md); print(md)
        ds = xr.Dataset({f"p_{k}": (("fit", "year"), v) for k, v in probs.items()}
                        | {k: ("year", v) for k, v in obs.items()} | {"obs_z": ("year", z)},
                        coords={"year": obs_years, "fit": np.arange(9)})
        if frac is not None:
            ds["cnn_vote_fraction"] = ("year", frac)
        ds.attrs.update(forced_method=fm, labels_file=str(a.labels_file), window=window, trend_offset=offset)
        ds.to_netcdf(out_dir / f"obs_predict_{fm}.nc")
        if not a.no_fig:
            from src.plotting import obs_predict as plot, style as st
            st.paper_rc()
            plot.plot_obs_predict(obs_years, obs, probs, z, frac, fm,
                                  paths.FIGURES_DIR / "diagnostics" / f"obs_predict_{key}_{fm}.png",
                                  obs_slow_years=slow_years, cnn_label=f"CNN {a.tag} ({a.product})")


if __name__ == "__main__":
    main()
