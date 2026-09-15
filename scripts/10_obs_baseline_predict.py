#!/usr/bin/env python3
"""
10_obs_baseline_predict.py — what the scalar baselines say about the observed record (AIES §8, Fig. 8b).

Logistic baselines (SIE anomaly, IPO, SIE+IPO, SIE+Niño3.4+IPO) fitted on CESM2-LE
with the given labels (offset labels by default), applied to observed September SIE
and JJA Pacific indices year by year; probabilities are calibrated to the training
prevalence (a balanced fit's 0.5 = the base rate). Two references, kept separate:
  --forced     what is removed from the observed SIE to make the *predictor* anomaly the
               model was trained on (ensmean group_cmip6 group_smbb linear quadratic)
  --label-ref  observation-only fit (linear | quadratic) against which the observed decade
               is classified as a slowdown (z > 1); σ from the model (--label-sigma model,
               the paper's definition) or from the observed trend anomalies (obs, as in v1).
CNN vote fractions for --tag are overlaid when present.

Outputs: results/obs_predict/<key>/<product>/obs_predict_<forced>[_sigmodel].nc, summary_<forced>[_sigmodel].md,
         FIGURES_DIR/diagnostics/obs_predict_<key>_<product>_<forced>[_sigmodel].png   (key = w<window>_off<offset>)

Usage:
  python scripts/10_obs_baseline_predict.py --labels-file $LBL1 --forced linear quadratic group_cmip6            # ERSST, obs σ
  python scripts/10_obs_baseline_predict.py --labels-file $LBL1 --forced linear --product oisst --tag off1        # OISST + CNN votes
  python scripts/10_obs_baseline_predict.py --labels-file $LBL1 --forced linear --label-sigma model               # SI: the model's bar
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
    p.add_argument("--forced", nargs="+", default=["ensmean"], help="predictor-anomaly references: ensmean group_cmip6 group_smbb linear quadratic")
    p.add_argument("--label-ref", default="linear", choices=["linear", "quadratic"], help="observed reference for classifying the observed decade")
    p.add_argument("--label-sigma", default="obs", choices=["obs", "model"],
                   help="σ for the observed z: observed trend-anomaly sd (default; the observed bar) or the model's pooled σ (SI comparison)")
    p.add_argument("--no-calibration", action="store_true", help="plot the raw balanced-fit probabilities (0.5 = base rate) instead")
    p.add_argument("--start-year", type=int, default=1990); p.add_argument("--end-year", type=int, default=2029,
                   help="last onset year used to FIT (must match the label file's cap)")
    p.add_argument("--obs-start", type=int, default=1990); p.add_argument("--obs-end", type=int, default=2025)
    p.add_argument("--demean", default="group", choices=["all", "group"])
    p.add_argument("--tag", default=None, help="CNN tag whose observational predictions to overlay (if present)")
    p.add_argument("--product", default="ersst", choices=["ersst", "oisst"], help="SST product for the indices and the CNN overlay")
    p.add_argument("--no-fig", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    with xr.open_dataset(a.labels_file) as ds:
        window, offset = int(ds.attrs.get("window", 10)), int(ds.attrs.get("trend_offset", 0))
    key = f"w{window}_off{offset}"
    out_dir = paths.RESULTS_DIR / "obs_predict" / key / a.product; out_dir.mkdir(parents=True, exist_ok=True)
    ipo_file, nino_file = paths.OBS_INDEX_FILES[a.product]
    for f in (ipo_file, nino_file):
        if not f.exists():
            sys.exit(f"{f} missing — run: python scripts/02_ersst_climate_indices.py --product {a.product} --index nino34 ipo")
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
    base_rate = float(labels.mean())
    z, sigma = op.observed_offset_z(obs_years, nsidc, a.labels_file, window, offset, a.label_ref, a.label_sigma)
    label_note = f"{a.label_ref} observed reference, σ = {sigma:.3f} Mkm² yr⁻¹ from {a.label_sigma}"
    print(f"observed slowdowns (z > 1, {label_note}): {obs_years[np.isfinite(z) & (z > 1)].tolist()}")
    for fm in a.forced:
        print(f"\n== forced reference: {fm}")
        obs = op.observed_scalars(obs_years, fm, nsidc, metrics_dir, ipo_file, nino_file, index_t0=None)
        probs = op.predict_observed(fits, obs, calibrated=not a.no_calibration)
        frac = op.cnn_vote_fraction(paths.obs_predictions_dir(a.product, fm, a.tag), obs_years) if a.tag else None
        if a.tag and frac is None:
            print(f"  [note] no CNN predictions for tag {a.tag} / {a.product} / {fm} — run 03_obs_test + 06_cnn_predict_obs")
        md = op.summary_markdown(obs_years, obs, probs, z, frac, fm, base_rate=base_rate, label_note=label_note)
        sfx = "" if a.label_sigma == "obs" else "_sigmodel"
        (out_dir / f"summary_{fm}{sfx}.md").write_text(md); print(md)
        ds = xr.Dataset({f"p_{k}": (("fit", "year"), v) for k, v in probs.items()}
                        | {k: ("year", v) for k, v in obs.items()} | {"obs_z": ("year", z)},
                        coords={"year": obs_years, "fit": np.arange(9)})
        if frac is not None:
            ds["cnn_vote_fraction"] = ("year", frac)
        ds.attrs.update(forced_method=fm, product=a.product, labels_file=str(a.labels_file), window=window, trend_offset=offset,
                        label_ref=a.label_ref, label_sigma=a.label_sigma, sigma=sigma, base_rate=base_rate,
                        calibrated=int(not a.no_calibration))
        ds.to_netcdf(out_dir / f"obs_predict_{fm}{sfx}.nc")
        if not a.no_fig:
            from src.plotting import obs_predict as plot, style as st
            st.paper_rc()
            plot.plot_obs_predict(obs_years, obs, probs, z, frac, fm,
                                  paths.FIGURES_DIR / "diagnostics" / f"obs_predict_{key}_{a.product}_{fm}{sfx}.png",
                                  obs_slow_years=slow_years, cnn_label=f"CNN {a.tag} ({a.product})",
                                  base_rate=base_rate if not a.no_calibration else 0.5,
                                  label_note=f"{a.label_ref} obs. reference, σ from {a.label_sigma}",
                                  product=a.product)


if __name__ == "__main__":
    main()
