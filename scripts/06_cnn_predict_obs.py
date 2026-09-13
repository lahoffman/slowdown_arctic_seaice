"""
06_cnn_predict_obs.py — run a retrained CNN configuration on observations.

For each split × seed: standardise the observed residual (and aux scalars)
with that split's training statistics, fill land with the sentinel, predict.
Threshold for the binary vote = the split's training precision–recall
intersection (cached in predictions/cesm2le/<tag>), falling back to 0.5.

Output: results/predictions/<product>/forced_<method>/<tag>/cnn_prediction_<product>_M{k}_{r}.nc

Usage:
  python scripts/06_cnn_predict_obs.py --product oisst --forced-method group_smbb --tag rel_aux
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths
from src.cnn.train import load_model

N_SPLITS, N_SEEDS, BASE_SEED, LAND_FILL = 9, 5, 42, -10.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--product", choices=list(paths.OBS_PRODUCTS), default="ersst")
    p.add_argument("--forced-method", default="ensmean")
    p.add_argument("--tag", default=None)
    return p.parse_args()


def split_stats(k, tag):
    """Training mean/std of the maps and aux columns for split k (scalars only; no big arrays read)."""
    with xr.open_dataset(paths.tvt_split_path(k, tag)) as ds:
        mu, sd = float(ds["mu_train"]), float(ds["sigma_train"])
        a_mu = [float(v) for v in str(ds.attrs.get("aux_mu_train", "")).split(",") if v]
        a_sd = [float(v) for v in str(ds.attrs.get("aux_sigma_train", "")).split(",") if v]
    return mu, sd, np.array(a_mu), np.array(a_sd)


def train_threshold(k, r, tag):
    f = paths.cesm2le_predictions_dir(tag) / f"cnn_prediction_cesm2le_M{k}_{r}.nc"
    if f.exists():
        with xr.open_dataset(f) as ds:
            return float(ds["threshold"])
    return 0.5


def main():
    a = parse_args()
    inp = paths.obs_input_file(a.product, a.forced_method, a.tag)
    if not inp.exists():
        raise FileNotFoundError(f"{inp} — run 03_obs_test.py first")
    out_dir = paths.obs_predictions_dir(a.product, a.forced_method, a.tag); out_dir.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(inp) as ds:
        resid, years = ds["sst_residual"].values, ds["year"].values
        aux = ds["aux"].values if "aux" in ds else None
        landmask = None
    with xr.open_dataset(paths.LANDMASK_FILE) as ds:
        landmask = ds["landmask"].values
    print(f"06  —  {a.product} / forced {a.forced_method} / tag {a.tag or 'orig'}: {resid.shape[0]} years "
          f"{years[0]}–{years[-1]}" + ("  + aux" if aux is not None else ""))
    n = 0
    for k in range(N_SPLITS):
        mu, sd, a_mu, a_sd = split_stats(k, a.tag)
        x = np.where(landmask == 1, LAND_FILL, (resid - mu) / sd)[..., None].astype(np.float32)
        inputs = [x, ((aux - a_mu) / a_sd).astype(np.float32)] if aux is not None and a_mu.size else x
        for r in range(N_SEEDS):
            if not paths.model_path(k, r, a.tag).exists():
                continue
            model = load_model(paths.models_dir(a.tag), k, r)
            prob = model.predict(inputs, verbose=0).ravel()
            thr = train_threshold(k, r, a.tag)
            ds = xr.Dataset({"y_prob": ("year", prob.astype(np.float32)),
                             "y_pred": ("year", (prob >= thr).astype(np.int8)),
                             "threshold": ((), np.float32(thr))},
                            coords={"year": years},
                            attrs=dict(split_idx=k, run_idx=r, seed=BASE_SEED + r, product=a.product,
                                       forced_method=a.forced_method, tag=a.tag or "", input_file=str(inp),
                                       threshold_rule="training precision-recall intersection (0.5 if uncached)"))
            ds.to_netcdf(out_dir / f"cnn_prediction_{a.product}_M{k}_{r}.nc")
            n += 1
        print(f"  split {k}: done", flush=True)
    print(f"  {n} prediction files → {out_dir}")


if __name__ == "__main__":
    main()
