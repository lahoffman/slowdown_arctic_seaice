"""
obs_input.py — observational CNN inputs for any product / any trained configuration.

Generalises ``ersst.test_cnn`` to (a) either SST product on the CESM2 grid
(ERSSTv5 or OISST v2.1), (b) the forced references used by the retrained
CNNs (100-member mean, or one forcing group's mean), and (c) the extra
inputs of the tagged configurations: the observed September SIE anomaly
(``aux``), a one-year lag (``sst_lag``) and the open-water mask
(``openwater``). Standardisation with the training statistics of each split
happens at prediction time (``scripts/06_cnn_predict_obs.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import netCDF4 as nc
import numpy as np
import xarray as xr

from src.data.observations.ersst.test_cnn import (
    _correct_forced_pixelwise, apply_land_ocean_mask, compute_jja_mean, remove_linear_trend)
from src.data.observations.ersst.climate_indices import _correct_forced_mean_and_trend, _detrend_linear
from src.data.cesm2le.forced import load_ensmean_jja_sst, load_groupmean_jja_sst
from src.data.cesm2le.slowdowns import load_sie_monthly_files

FORCED_METHODS = ("ensmean", "group_cmip6", "group_smbb", "linear")
GROUP_SLICES = {"cmip6": slice(0, 50), "smbb": slice(50, 100)}


# ---------------------------------------------------------------------------
# products
# ---------------------------------------------------------------------------

def load_monthly_product(path: Path) -> Dict:
    """Regridded monthly product → dict(sst (nt, nx, ny), years, months, ice or None)."""
    with xr.open_dataset(path) as ds:
        sst = ds["sst_obs"].values.astype(np.float64)
        if "time" in ds.coords:
            t = ds["time"].dt
            years, months = t.year.values, t.month.values
        else:                                              # ERSST: infer from date_range attr
            y0, m0 = map(int, ds.attrs["date_range"].split(" to ")[0].split("-"))
            idx = np.arange(sst.shape[0]) + (m0 - 1)
            years, months = y0 + idx // 12, idx % 12 + 1
        ice = ds["ice_obs"].values.astype(np.float32) if "ice_obs" in ds else None
    return dict(sst=sst, years=years, months=months, ice=ice)


def jja_by_year(prod: Dict, years: np.ndarray, field: str = "sst") -> np.ndarray:
    """JJA mean of ``field`` for each requested calendar year, (nyear, nx, ny)."""
    arr = prod[field]
    out = np.full((years.size,) + arr.shape[1:], np.nan, np.float64)
    for k, y in enumerate(years):
        sel = (prod["years"] == y) & np.isin(prod["months"], (6, 7, 8))
        if sel.sum() == 3:
            out[k] = np.nanmean(arr[sel], axis=0)
    return out


# ---------------------------------------------------------------------------
# forced references
# ---------------------------------------------------------------------------

def forced_sst_field(method: str, years: np.ndarray, ensmean_path: Path, groupmean_path: Path) -> np.ndarray:
    """Model forced JJA SST (nyear, nx, ny) for ``method`` ('ensmean' | 'group_cmip6' | 'group_smbb')."""
    if method == "ensmean":
        field, fyears = load_ensmean_jja_sst(ensmean_path)
    else:
        gm, names, fyears = load_groupmean_jja_sst(groupmean_path)
        field = gm[names.index(method.split("_", 1)[1])]
    sel = np.isin(fyears, years)
    if sel.sum() != years.size:
        raise ValueError(f"forced field covers {fyears[0]}–{fyears[-1]}, need {years[0]}–{years[-1]}")
    return field[sel]


def remove_forced(sst_jja: np.ndarray, years: np.ndarray, method: str,
                  ensmean_path: Path, groupmean_path: Path) -> np.ndarray:
    """Observed JJA SST minus the forced signal, per ``method`` (see FORCED_METHODS)."""
    if method == "linear":
        return remove_linear_trend(sst_jja, int(years[0]), int(years[-1]))
    forced = forced_sst_field(method, years, ensmean_path, groupmean_path)
    return _correct_forced_pixelwise(sst_jja, forced)


def observed_sie_anomaly(nsidc_events_file: Path, years: np.ndarray, method: str,
                         cesm_metrics_dir: Path, month: str = "SEP") -> np.ndarray:
    """
    Observed September SIE anomaly at each onset year, with the forced part removed
    the same way as for the model: model group/ensemble-mean SIE, mean- and
    trend-corrected to observations, or a linear trend for ``method='linear'``.
    """
    with xr.open_dataset(nsidc_events_file) as ds:
        obs_years = ds["yearice"].values.astype(int)
        obs_sie = ds["seaice"].values.astype(float)
    idx = np.searchsorted(obs_years, years)
    if not np.array_equal(obs_years[np.clip(idx, 0, obs_years.size - 1)], years):
        raise ValueError(f"NSIDC SIE covers {obs_years[0]}–{obs_years[-1]}, need {years[0]}–{years[-1]}")
    sie = obs_sie[idx]
    if method == "linear":
        return _detrend_linear(sie)
    model_sie, model_years = load_sie_monthly_files(str(cesm_metrics_dir), month, variable="sie",
                                                    start_year=1990, end_year=2100)
    sl = GROUP_SLICES[method.split("_", 1)[1]] if method.startswith("group_") else slice(0, 100)
    forced = model_sie[sl].mean(0)[np.isin(model_years, years)]
    _, _, resid = _correct_forced_mean_and_trend(sie, forced, np.arange(years.size, dtype=float))
    return resid


# ---------------------------------------------------------------------------
# end-to-end
# ---------------------------------------------------------------------------

def prepare_obs_input(product_path: Path, landmask_path: Path, forced_method: str,
                      ensmean_path: Path, groupmean_path: Path, start_year: int, end_year: int,
                      sst_lag: int = 0, sst_window: int = 1, aux: str = "none", openwater: bool = False,
                      ice_threshold: float = 0.15, ice_product_path: Optional[Path] = None,
                      nsidc_events_file: Optional[Path] = None, cesm_metrics_dir: Optional[Path] = None,
                      mask_north: Optional[float] = None) -> Dict:
    """
    Observed inputs for one CNN configuration, *unstandardised*.

    Returns dict(residual (nyear, nx, ny): forced-removed JJA SST of year t − sst_lag
    (averaged over ``sst_window`` seasons from there for the concurrent configuration),
    aux (nyear, naux) or None, icemask (nyear, nx, ny) or None, target_years, sst_years).
    """
    if forced_method not in FORCED_METHODS:
        raise ValueError(f"forced_method must be one of {FORCED_METHODS}")
    with nc.Dataset(landmask_path) as ds:
        landmask = np.array(ds["landmask"][:])
    target_years = np.arange(start_year, end_year + 1)
    sst_years = target_years - sst_lag
    all_sst_years = np.arange(sst_years[0], sst_years[-1] + sst_window)     # every season needed

    prod = load_monthly_product(product_path)
    sst_jja = apply_land_ocean_mask(jja_by_year(prod, all_sst_years), landmask)
    ok = ~np.isnan(sst_jja).all(axis=(1, 2))
    if not ok.all():
        raise ValueError(f"product lacks JJA data for years {all_sst_years[~ok]}")
    residual = remove_forced(sst_jja, all_sst_years, forced_method, ensmean_path, groupmean_path)
    if sst_window > 1:
        residual = np.stack([np.nanmean(residual[k:k + sst_window], axis=0) for k in range(sst_years.size)])
    if mask_north is not None:                                 # extra-Arctic configuration: same rows zeroed as in 03
        from configs import paths as _paths
        with nc.Dataset(_paths.CESM2LE_GRID_FILE) as g:
            lat = np.array(g["lat"][:])
        residual[:, lat >= mask_north, :] = 0.0

    aux_arr = None
    if aux == "sie_anom":
        aux_arr = observed_sie_anomaly(nsidc_events_file, target_years, forced_method,
                                       cesm_metrics_dir)[:, None]

    icemask = None
    if openwater:
        ice_src = load_monthly_product(ice_product_path or product_path)
        if ice_src["ice"] is None:
            raise ValueError("open-water masking needs a product with an ice field (OISST); "
                             "pass ice_product_path")
        icemask = jja_by_year(ice_src, sst_years, "ice") > ice_threshold
        residual = np.where(icemask, 0.0, residual)

    return dict(residual=residual, aux=aux_arr, icemask=icemask, target_years=target_years,
                sst_years=sst_years, sst_window=sst_window, landmask=landmask, forced_method=forced_method,
                mask_north=mask_north)


def save_obs_input(res: Dict, out: Path, attrs: Dict) -> None:
    ds = xr.Dataset({"sst_residual": (("year", "nx", "ny"), res["residual"].astype(np.float32))},
                    coords={"year": res["target_years"], "sst_year": ("year", res["sst_years"])})
    if res["aux"] is not None:
        ds["aux"] = (("year", "naux"), res["aux"].astype(np.float32))
    if res["icemask"] is not None:
        ds["icemask"] = (("year", "nx", "ny"), res["icemask"].astype(np.int8))
    ds.attrs.update(attrs)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out, encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars if ds[v].ndim > 1})
