"""
ohc.py — upper-ocean heat content from the CESM2-LE Zarr store on AWS, without downloading TEMP.

Opens the ocean-monthly TEMP stores lazily (s3fs + zarr + dask), integrates the top levels with the POP
layer thicknesses, takes annual means, regrids the POP T-grid to the CAM grid (nearest neighbour, as for
aice) and returns (nyear, nx, ny) per member. Raw TEMP never touches disk.

OHC_0-H = rho * c_p * sum_k (T_k - T_ref) dz_k over levels whose bottom z_w_bot <= H   [J m^-2]
POP gx1v7: 10-m layers to 160 m, then thickening; 0-100 m = 10 levels, 0-300 m ~ 23, 0-700 m = 33.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

RHO, CP = 1026.0, 3990.0                 # kg m-3, J kg-1 K-1
T_REF_C = -1.8                           # freezing reference; cancels in anomalies
CATALOG = "https://ncar-cesm2-lens.s3.amazonaws.com/catalogs/aws-cesm2-le.json"
GROUP_OF_FORCING = {"cmip6": "first50", "smbb": "last50"}


def catalog_temp_entries(catalog_url: str = CATALOG) -> pd.DataFrame:
    """Rows of the intake-esm catalog for ocean monthly TEMP (empty if the store lacks it)."""
    import intake
    cat = intake.open_esm_datastore(catalog_url)
    df = cat.df
    sel = (df["variable"] == "TEMP") & df["frequency"].str.contains("month", case=False)
    if "component" in df:
        sel &= df["component"] == "ocn"
    return df[sel].reset_index(drop=True)


def open_store(path: str) -> xr.Dataset:
    """Lazy xarray view of one Zarr store on the anonymous bucket."""
    import s3fs
    fs = s3fs.S3FileSystem(anon=True)
    return xr.open_zarr(fs.get_mapper(path), consolidated=True)


def layer_geometry(z_t_cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(dz, z_w_bot) in cm from the level centres: POP centres are layer midpoints, so bottoms follow recursively."""
    z_w_bot = np.empty_like(z_t_cm, dtype=float); top = 0.0
    for k, zc in enumerate(z_t_cm):
        z_w_bot[k] = 2 * zc - top; top = z_w_bot[k]
    dz = np.diff(np.concatenate([[0.0], z_w_bot]))
    return dz, z_w_bot


def levels_to(depth_m: float, z_w_bot_cm: np.ndarray) -> int:
    """Number of top levels whose bottom lies at or above ``depth_m`` (POP stores z in cm)."""
    return int((z_w_bot_cm / 100.0 <= depth_m + 1e-6).sum())


def ohc_column(temp_c: xr.DataArray, dz_cm: np.ndarray, n_levels: int) -> xr.DataArray:
    """Vertical integral of the top ``n_levels`` (J m^-2), lazy. temp_c dims (..., z_t, nlat, nlon)."""
    dz_m = xr.DataArray(dz_cm[:n_levels] / 100.0, dims=("z_t",))
    t = temp_c.isel(z_t=slice(0, n_levels))
    return (RHO * CP * (t - T_REF_C) * dz_m).sum("z_t", skipna=False)


def annual_mean(da: xr.DataArray, years: np.ndarray) -> xr.DataArray:
    """Calendar-year means for the requested years (cftime-safe)."""
    yr = da["time"].dt.year
    da = da.assign_coords(year=("time", yr.values))
    out = da.groupby("year").mean("time")
    return out.reindex(year=years)                       # NaN for years the store does not cover


def pop_to_cam_indices(tlat: np.ndarray, tlon: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Nearest POP T-cell for every CAM cell (flat indices into the POP grid), computed once."""
    from scipy.spatial import cKDTree
    src = np.column_stack([tlat.ravel(), np.mod(tlon.ravel(), 360)])
    ok = np.isfinite(src).all(1)
    tree = cKDTree(src[ok])
    lon2, lat2 = np.meshgrid(np.mod(lon, 360), lat)
    _, idx = tree.query(np.column_stack([lat2.ravel(), lon2.ravel()]))
    return np.flatnonzero(ok)[idx]


def regrid(field: np.ndarray, idx: np.ndarray, shape: Tuple[int, int], landmask: Optional[np.ndarray] = None) -> np.ndarray:
    """Apply the nearest-neighbour map to (..., nlat, nlon) → (..., nx, ny); CAM land → NaN."""
    flat = field.reshape(field.shape[:-2] + (-1,))[..., idx]
    out = flat.reshape(field.shape[:-2] + shape)
    if landmask is not None:
        out = np.where(landmask == 1, np.nan, out)
    return out


def member_series(ds_hist: xr.Dataset, ds_ssp: xr.Dataset, member: str, n_levels: Dict[int, int], dz_cm: np.ndarray,
                  years: np.ndarray) -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """
    MONTHLY OHC (nmonth, nlat, nlon) for one member and every requested depth, from ONE pass over the
    historical + SSP stores (chunks carry all 60 levels, so all integrals come from the same read).
    n_levels: {depth_m: number of top levels}. Returns ({depth: array}, year_of_month, month_of_month).
    Annual / seasonal means are taken later by the analysis scripts (``seasonal_mean``).
    """
    parts = {d: [] for d in n_levels}
    for ds in (ds_hist, ds_ssp):
        if ds is None:
            continue
        t = ds["TEMP"].sel(member_id=member)
        for d, n in n_levels.items():
            parts[d].append(ohc_column(t, dz_cm, n))
    stacked = xr.concat([xr.concat(parts[d], dim="time") if len(parts[d]) > 1 else parts[d][0] for d in n_levels],
                        dim=xr.DataArray(list(n_levels), dims="depth", name="depth"))
    stacked = stacked.sel(time=slice(f"{years[0]}-01-01", f"{years[-1]}-12-31"))
    yr, mo = stacked["time"].dt.year.values, stacked["time"].dt.month.values
    out = stacked.transpose("depth", "time", "nlat", "nlon").values.astype(np.float32)          # one compute
    return {d: out[i] for i, d in enumerate(n_levels)}, yr, mo


def seasonal_mean(monthly: np.ndarray, year_of: np.ndarray, month_of: np.ndarray, years: np.ndarray,
                  months: Sequence[int] = range(1, 13)) -> np.ndarray:
    """(..., nmonth, ...) monthly field → (..., nyear, ...) mean over ``months`` of each calendar year (NaN if absent)."""
    axis = monthly.ndim - 3 if monthly.ndim >= 3 else 0            # the time axis sits before (lat, lon)
    out = []
    for y in years:
        sel = (year_of == y) & np.isin(month_of, list(months))
        out.append(np.nanmean(np.take(monthly, np.flatnonzero(sel), axis=axis), axis=axis) if sel.any()
                   else np.full(monthly.shape[:axis] + monthly.shape[axis + 1:], np.nan, np.float32))
    return np.stack(out, axis=axis)
