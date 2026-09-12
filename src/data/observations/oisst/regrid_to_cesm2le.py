"""
Regrid monthly OISST (0.25°) to the CESM2-LE atmosphere grid by area-weighted
block averaging — every 0.25° cell is assigned to the CESM2 cell whose edges
contain its centre, so no interpolation across the ice edge or coast. The
output has the same layout as the regridded ERSST file (``sst_obs``,
``lat_cesm2``, ``lon_cesm2``) plus ``ice_obs`` and ``nvalid``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr


def cell_edges(centres: np.ndarray, periodic: bool = False, period: float = 360.0) -> np.ndarray:
    """Edges midway between centres; end cells extended symmetrically (lat clipped to ±90)."""
    c = np.asarray(centres, float)
    mid = 0.5 * (c[1:] + c[:-1])
    if periodic:
        first = c[0] - 0.5 * ((c[0] + period) - c[-1])
        return np.concatenate([[first], mid, [first + period]])
    e = np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])
    return np.clip(e, -90.0, 90.0)


def block_average(field: np.ndarray, lat_src: np.ndarray, lon_src: np.ndarray,
                  lat_tgt: np.ndarray, lon_tgt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Area-weighted mean of ``field`` (..., nlat_src, nlon_src) over CESM2 cells.

    Returns (mean (..., nlat_tgt, nlon_tgt), nvalid) where nvalid counts the
    finite source cells in each target cell.
    """
    lat_e, lon_e = cell_edges(lat_tgt), cell_edges(lon_tgt, periodic=True)
    lon_src = np.mod(lon_src, 360.0)
    lon_src_wrapped = np.where(lon_src < lon_e[0], lon_src + 360.0, lon_src)
    i_lat = np.clip(np.digitize(lat_src, lat_e) - 1, 0, lat_tgt.size - 1)
    i_lon = np.clip(np.digitize(lon_src_wrapped, lon_e) - 1, 0, lon_tgt.size - 1)
    w = np.cos(np.deg2rad(lat_src))[:, None] * np.ones((1, lon_src.size))
    lead = field.shape[:-2]
    f = field.reshape(-1, lat_src.size, lon_src.size)
    flat_tgt = (i_lat[:, None] * lon_tgt.size + i_lon[None, :]).ravel()
    ncell = lat_tgt.size * lon_tgt.size
    out = np.full((f.shape[0], ncell), np.nan, np.float32)
    nvalid = np.zeros((f.shape[0], ncell), np.int32)
    for t in range(f.shape[0]):
        v = f[t].ravel(); ok = np.isfinite(v)
        num = np.bincount(flat_tgt[ok], weights=(v * w.ravel())[ok], minlength=ncell)
        den = np.bincount(flat_tgt[ok], weights=w.ravel()[ok], minlength=ncell)
        cnt = np.bincount(flat_tgt[ok], minlength=ncell)
        out[t] = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
        nvalid[t] = cnt
    return (out.reshape(*lead, lat_tgt.size, lon_tgt.size),
            nvalid.reshape(*lead, lat_tgt.size, lon_tgt.size))


def regrid_monthly(ds: xr.Dataset, lat_tgt: np.ndarray, lon_tgt: np.ndarray) -> xr.Dataset:
    """Regrid the concatenated monthly OISST Dataset (sst, ice) to the CESM2 grid."""
    lat_src, lon_src = ds["lat"].values, ds["lon"].values
    sst, nvalid = block_average(ds["sst"].values, lat_src, lon_src, lat_tgt, lon_tgt)
    ice, _ = block_average(np.nan_to_num(ds["ice"].values, nan=0.0), lat_src, lon_src, lat_tgt, lon_tgt)
    lon2d, lat2d = np.meshgrid(lon_tgt, lat_tgt)
    dates = pd.DatetimeIndex(ds["time"].values)
    out = xr.Dataset(
        {"sst_obs": (("nte", "nx", "ny"), sst),
         "ice_obs": (("nte", "nx", "ny"), ice.astype(np.float32)),
         "nvalid": (("nte", "nx", "ny"), nvalid),
         "lat_cesm2": (("nx", "ny"), lat2d), "lon_cesm2": (("nx", "ny"), lon2d)},
        coords={"nte": np.arange(sst.shape[0]), "nx": np.arange(lat_tgt.size), "ny": np.arange(lon_tgt.size),
                "time": ("nte", dates)},
    )
    out.attrs.update(description="OISST v2.1 monthly SST regridded to the CESM2-LE atmosphere grid",
                     source="NOAA OISST v2.1 AVHRR-only", regrid_method="area-weighted block average of 0.25° cells",
                     date_range=f"{dates[0]:%Y-%m} to {dates[-1]:%Y-%m}", target_grid="CESM2-LE f09_g17 (192x288)")
    out["sst_obs"].attrs.update(units="degC"); out["ice_obs"].attrs.update(units="1")
    return out


def save_regridded(ds: xr.Dataset, output_file: Path) -> None:
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_file, format="NETCDF4",
                 encoding={v: {"zlib": True, "complevel": 4} for v in ("sst_obs", "ice_obs", "nvalid")})
    print(f"✓ Saved regridded OISST to: {output_file}  {ds['sst_obs'].shape}")
