"""
ohc.py — upper-ocean heat content from the CESM2-LE Zarr store on AWS, without downloading TEMP.

Opens the ocean-monthly TEMP stores lazily (s3fs + zarr + dask), integrates the top levels with the POP
layer thicknesses, takes annual means, regrids the POP T-grid to the CAM grid (nearest neighbour, as for
aice) and returns (nyear, nx, ny) per member. Raw TEMP never touches disk.

OHC_0-H = rho * c_p * sum_k (T_k - T_ref) dz_k over levels whose bottom z_w_bot <= H   [J m^-2]
POP gx1v7: 10-m layers to 160 m, then thickening; 0-100 m = 10 levels, 0-300 m ~ 23, 0-700 m = 33.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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
    return out.sel(year=years)


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


def member_series(ds_hist: xr.Dataset, ds_ssp: xr.Dataset, member: str, n_levels: int, dz_cm: np.ndarray,
                  years: np.ndarray) -> np.ndarray:
    """Annual OHC (nyear, nlat, nlon) for one member across the historical + SSP stores; computes here."""
    parts = []
    for ds in (ds_hist, ds_ssp):
        if ds is None:
            continue
        t = ds["TEMP"].sel(member_id=member)
        parts.append(ohc_column(t, dz_cm, n_levels))
    da = xr.concat(parts, dim="time") if len(parts) > 1 else parts[0]
    da = da.sel(time=slice(f"{years[0]}-01-01", f"{years[-1]}-12-31"))
    return annual_mean(da, years).transpose("year", "nlat", "nlon").values.astype(np.float32)
