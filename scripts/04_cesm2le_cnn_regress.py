#!/usr/bin/env python3
"""
04_cesm2le_cnn_regress.py — CNN regression on the continuous trend anomaly (step 8.7).

Same inputs as a classification tag's split files (SST maps [+ SIE scalar]), but the
target is the 10-yr group-relative trend anomaly (from the relative label file) instead
of the 1σ label, standardised with training statistics. Each model is scored as test R²
next to an OLS on the SIE anomaly alone and on SIE + Pacific indices, fitted on the same
training samples — so the question "does the map add information beyond the ice state?"
is answered without any threshold.

Outputs: results/regression/<tag>/cnn_regress_split{k}_run{r}.h5, regress_scores.nc,
         regress_summary.md, FIGURES_DIR/diagnostics/regression_<tag>.png

Usage:
  python scripts/04_cesm2le_cnn_regress.py --tag rel_aux --splits 2 5 7 --n-runs 2   # subset test
  python scripts/04_cesm2le_cnn_regress.py --tag rel_aux                              # all 9 × 5
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths
from src.analysis import baselines as bl
from src.analysis.residual import _ols, _r2
from src.cnn.splits import load_tvt_split, block_tvt_split
from src.cnn.model import build_cnn
from src.cnn.train import set_seed, train_model, model_inputs, n_aux_inputs

TRAIN = dict(learning_rate=1e-4, num_epochs=50, batch_size=120, patience=10, task="regression")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True, help="classification tag whose split files supply the inputs")
    p.add_argument("--labels-file", type=Path, default=None,
                   help="relative-labels file supplying trend_anom; default: the file the tag's splits were built from")
    p.add_argument("--splits", type=int, nargs="+", default=list(range(9)))
    p.add_argument("--n-runs", type=int, default=5)
    p.add_argument("--epochs", type=int, default=TRAIN["num_epochs"])
    p.add_argument("--skip-existing", action="store_true"); p.add_argument("--no-fig", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    out_dir = paths.RESULTS_DIR / "regression" / a.tag; out_dir.mkdir(parents=True, exist_ok=True)
    sp0 = load_tvt_split(paths.tvt_split_path(0, a.tag))
    y0, y1 = (int(v) for v in sp0["attrs"]["target_years"].split("-"))
    years = np.arange(y0, y1 + 1)
    if a.labels_file is None:
        a.labels_file = Path(sp0["attrs"]["labels_file"])
    print(f"target: trend_anom from {a.labels_file}")
    with xr.open_dataset(a.labels_file) as ds:
        trend = ds["trend_anom"].sel(nyr=slice(y0, y1)).values.astype(np.float32)          # (nens, nyear)
    _, sie_anom = bl.load_sie_anomaly(paths.CESM2LE_AICE_DIR / "metrics", years, demean="group")
    idx = bl.load_climate_indices_jja(paths.CESM2LE_CLIMATE_INDICES_DIR, y0, y1)
    scalars = {"sie": sie_anom, **idx}
    print(f"04r  —  CNN regression on the trend anomaly  tag {a.tag}  years {y0}–{y1}  splits {a.splits}  runs {a.n_runs}")

    rows = []
    for k, te_b, va_b, tr_b in bl.iter_split_blocks(9, 10):
        if k not in a.splits:
            continue
        sp = load_tvt_split(paths.tvt_split_path(k, a.tag))
        x_tr, x_va, x_te = (model_inputs(sp, part) for part in ("tr", "va", "te"))
        n_aux = n_aux_inputs(sp) if isinstance(x_tr, list) else 0
        maps_tr = x_tr[0] if n_aux else x_tr
        y_tr, y_va, y_te = block_tvt_split(trend, tr_b, va_b, te_b)
        S = {n: block_tvt_split(v.astype(np.float32), tr_b, va_b, te_b) for n, v in scalars.items()}
        mu, sd = float(np.nanmean(y_tr)), float(np.nanstd(y_tr))
        z = lambda y: (y - mu) / sd
        # linear references on the same samples
        r2_sie = _r2(y_te, _ols(S["sie"][0][:, None], y_tr, S["sie"][2][:, None])[1])
        Xp_tr = np.column_stack([S[n][0] for n in ("sie", "nino34", "ipo")]); Xp_te = np.column_stack([S[n][2] for n in ("sie", "nino34", "ipo")])
        r2_pac = _r2(y_te, _ols(Xp_tr, y_tr, Xp_te)[1])
        print(f"\nsplit {k}: n_tr {y_tr.size}  aux {n_aux}  OLS R² SIE {r2_sie:.3f}  SIE+Pacific {r2_pac:.3f}")
        for r in range(a.n_runs):
            f = out_dir / f"cnn_regress_split{k}_run{r}.h5"
            set_seed(42 + r)
            if a.skip_existing and f.exists():
                from tensorflow import keras
                model = keras.models.load_model(f, compile=False); hist = None
            else:
                aux_init = None
                if n_aux:                                           # warm start at OLS(z(y) ~ aux), step 8.8
                    A = np.column_stack([np.ones(len(x_tr[1])), x_tr[1]])
                    beta = np.linalg.lstsq(A, z(y_tr), rcond=None)[0]
                    aux_init = (beta[1:], float(beta[0]))
                model = build_cnn(maps_tr.shape[1], maps_tr.shape[2], maps_tr.shape[3], n_aux=n_aux,
                                  task="regression", aux_init=aux_init)
                model, h = train_model(model, x_tr, z(y_tr), x_va, z(y_va), config={**TRAIN, "num_epochs": a.epochs})
                model.save(str(f)); hist = {kk: [float(v) for v in vv] for kk, vv in h.history.items()}
                (out_dir / f"history_split{k}_run{r}.json").write_text(json.dumps(hist))
            pred = model.predict(x_te, verbose=0).ravel() * sd + mu
            r2 = _r2(y_te, pred)
            rows.append(dict(split=k, run=r, r2_cnn=r2, r2_sie=r2_sie, r2_sie_pacific=r2_pac,
                             epochs=len(hist["loss"]) if hist else -1))
            np.savez(out_dir / f"pred_split{k}_run{r}.npz", y_true=y_te, y_pred=pred, sie=S["sie"][2])
            print(f"  run {r}: CNN test R² {r2:.3f}  (Δ vs SIE {r2 - r2_sie:+.3f})  epochs {rows[-1]['epochs']}", flush=True)

    ds = xr.Dataset({kk: ("model", np.array([row[kk] for row in rows])) for kk in rows[0]})
    ds.to_netcdf(out_dir / "regress_scores.nc")
    med = lambda kk: float(np.median(ds[kk]))
    md = (f"# CNN regression on the trend anomaly — tag `{a.tag}` ({len(rows)} models, splits {sorted(set(ds['split'].values.tolist()))})\n\n"
          f"Test R² (median): **CNN {med('r2_cnn'):.3f}**, OLS on SIE anomaly {med('r2_sie'):.3f}, "
          f"OLS on SIE + Niño 3.4 + IPO {med('r2_sie_pacific'):.3f}.\n\n"
          f"CNN − SIE-only: {float(np.median(ds['r2_cnn'] - ds['r2_sie'])):+.3f} (range "
          f"{float((ds['r2_cnn'] - ds['r2_sie']).min()):+.3f} … {float((ds['r2_cnn'] - ds['r2_sie']).max()):+.3f}).\n")
    (out_dir / "regress_summary.md").write_text(md); print("\n" + md)
    if not a.no_fig:
        from src.plotting import regression as plot, style as st
        st.paper_rc()
        plot.plot_regression(ds, out_dir, paths.FIGURES_DIR / "diagnostics" / f"regression_{a.tag}.png")


if __name__ == "__main__":
    main()
