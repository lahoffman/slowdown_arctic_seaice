"""
03_obs_test.py — observational CNN inputs for a retrained configuration (steps 6.1–6.3).

Reads the configuration from the tag's split file (sst_lag, sst_window, aux, openwater),
builds the matching observed input from either SST product, with the forced
signal removed by the chosen reference, and saves it *unstandardised*;
06_cnn_predict_obs.py standardises with each split's training statistics.

Forced references: ensmean (100-member mean, as in v1), group_cmip6 /
group_smbb (one forcing group's mean — the retrained CNNs were trained on
group-demeaned SST), linear (per-pixel trend). Open-water masking uses the
OISST ice field for both products.

Usage:
  python scripts/03_obs_test.py --product oisst --forced-method group_smbb --tag rel_aux
  python scripts/03_obs_test.py --product ersst --forced-method ensmean --tag rel_base --end-year 2024
"""

import argparse
import sys
from pathlib import Path

import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths
from src.data.observations import obs_input as oi


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--product", choices=list(paths.OBS_PRODUCTS), default="ersst")
    p.add_argument("--forced-method", choices=oi.FORCED_METHODS, default="ensmean")
    p.add_argument("--tag", default=None, help="CNN configuration whose inputs to mimic (None = original)")
    p.add_argument("--start-year", type=int, default=None,
                   help="first onset year (default 1990, or 1990 + sst_lag for lagged configurations)")
    p.add_argument("--end-year", type=int, default=2025, help="last onset year with a JJA in the product")
    return p.parse_args()


def config_from_split(tag):
    """sst_lag / aux / openwater / ice_threshold recorded by 03_cesm2le_tvt_splits.py."""
    if tag is None:
        return dict(sst_lag=0, sst_window=1, aux="none", openwater=False, ice_threshold=0.15)
    with xr.open_dataset(paths.tvt_split_path(0, tag)) as ds:
        a = ds.attrs
    return dict(sst_lag=int(a.get("sst_lag", 0)), sst_window=int(a.get("sst_window", 1)), aux=str(a.get("aux", "none")),
                openwater=bool(int(a.get("openwater", 0))), ice_threshold=float(a.get("ice_threshold", 0.15) or 0.15))


def main():
    a = parse_args()
    cfg = config_from_split(a.tag)
    if a.start_year is None:
        a.start_year = 1990 + cfg["sst_lag"]
    if cfg["sst_window"] > 1:          # concurrent configuration: the last onset needs SST through t + window − 1
        a.end_year = min(a.end_year, 2025 - cfg["sst_window"] + 1 + cfg["sst_lag"])
    out = paths.obs_input_file(a.product, a.forced_method, a.tag)
    print("03  —  observational CNN input")
    print(f"  product {a.product}   forced {a.forced_method}   tag {a.tag or 'orig'}   config {cfg}")
    res = oi.prepare_obs_input(
        product_path=paths.OBS_PRODUCTS[a.product], landmask_path=paths.LANDMASK_FILE,
        forced_method=a.forced_method, ensmean_path=paths.CESM2LE_ENSMEAN_JJA,
        groupmean_path=paths.CESM2LE_GROUPMEAN_JJA, start_year=a.start_year, end_year=a.end_year,
        sst_lag=cfg["sst_lag"], sst_window=cfg["sst_window"], aux=cfg["aux"], openwater=cfg["openwater"], ice_threshold=cfg["ice_threshold"],
        ice_product_path=paths.OISST_REGRIDDED, nsidc_events_file=paths.nsidc_sie_slowdown_events(9),
        cesm_metrics_dir=paths.CESM2LE_AICE_DIR / "metrics")
    oi.save_obs_input(res, out, dict(product=a.product, forced_method=a.forced_method, tag=a.tag or "",
                                     **{k: (int(v) if isinstance(v, bool) else v) for k, v in cfg.items()}))
    print(f"  residual {res['residual'].shape}  aux {None if res['aux'] is None else res['aux'].shape}"
          f"  icemask {None if res['icemask'] is None else int(res['icemask'].sum())} cells")
    print(f"  saved → {out}")


if __name__ == "__main__":
    main()
