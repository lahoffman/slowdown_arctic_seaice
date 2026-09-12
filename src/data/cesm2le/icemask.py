"""
icemask.py — JJA sea-ice cover on the SST grid (revision step 1.6).

In CESM2 the SST under sea ice sits at the freezing point, so an Arctic SST
anomaly map is partly an ice-concentration map. The ``openwater`` CNN variant
sets the SST anomaly to zero wherever JJA ice concentration exceeds a
threshold, so the network only sees open-water SST. This module builds that
mask once (nearest-neighbour from the CICE grid to the 192 x 288 atmosphere
grid) and saves it; ``scripts/03_cesm2le_tvt_splits.py --openwater`` applies it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import netCDF4 as nc
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

JJA = ["JUN", "JUL", "AUG"]


def _unit_vectors(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    la, lo = np.deg2rad(lat).ravel(), np.deg2rad(lon).ravel()
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def nearest_index_map(src_lat: np.ndarray, src_lon: np.ndarray,
                      dst_lat: np.ndarray, dst_lon: np.ndarray) -> np.ndarray:
    """
    Index of the nearest source cell for every destination cell.

    Args:
        src_lat, src_lon: 2-D CICE coordinates (nj, ni).
        dst_lat, dst_lon: 1-D atmosphere-grid coordinates (nx,), (ny,).
    Returns:
        flat indices into the source grid, shape (nx, ny).
    """
    ok = np.isfinite(src_lat.ravel()) & np.isfinite(src_lon.ravel())
    src_idx = np.where(ok)[0]
    tree = cKDTree(_unit_vectors(src_lat.ravel()[ok], src_lon.ravel()[ok]))
    lon2d, lat2d = np.meshgrid(dst_lon, dst_lat)
    _, k = tree.query(_unit_vectors(lat2d, lon2d))
    return src_idx[k].reshape(lat2d.shape)


def load_cice_grid(grid_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    """TLAT, TLON (nj, ni) from a raw CESM2 CICE history file."""
    with nc.Dataset(grid_file) as ds:
        return np.array(ds["TLAT"][:], float), np.array(ds["TLON"][:], float)


def load_jja_aice(aice_monthly_template: Dict[str, str], member_groups: List[str],
                  aice_var: str = "aice_mon") -> np.ndarray:
    """JJA-mean ice concentration for all members and file years, (nens, nyear, nj, ni), 0–1."""
    total = None
    for m in JJA:
        arrs = []
        for g in member_groups:
            f = aice_monthly_template[g].format(month=m)
            if not Path(f).exists():
                raise FileNotFoundError(f"aice file not found: {f}")
            with nc.Dataset(f) as ds:
                a = np.array(ds[aice_var][:], np.float32)
            arrs.append(a)
        a = np.concatenate(arrs, axis=0)
        total = a if total is None else total + a
    jja = total / len(JJA)
    if np.nanmax(jja) > 1.5:            # CICE writes percent in some outputs
        jja = jja / 100.0
    return jja


def regrid_nearest(field: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Apply a nearest-index map to the trailing (nj, ni) axes → (..., nx, ny)."""
    flat = field.reshape(*field.shape[:-2], -1)
    return flat[..., idx]


def build_icemask(aice_monthly_template: Dict[str, str], member_groups: List[str],
                  cice_grid_file: Path, atm_lat: np.ndarray, atm_lon: np.ndarray,
                  years: np.ndarray, threshold: float = 0.15) -> xr.Dataset:
    """JJA ice-cover mask (1 = aice > threshold) on the atmosphere grid, (nens, nyear, nx, ny)."""
    tlat, tlon = load_cice_grid(cice_grid_file)
    idx = nearest_index_map(tlat, tlon, atm_lat, atm_lon)
    jja = load_jja_aice(aice_monthly_template, member_groups)
    if jja.shape[1] != years.size:
        raise ValueError(f"aice has {jja.shape[1]} years, expected {years.size}")
    jja[np.isnan(jja)] = 0.0
    conc = regrid_nearest(jja, idx).astype(np.float32)           # (nens, nyear, nx, ny)
    ds = xr.Dataset(
        {"icemask": (("nens", "year", "nx", "ny"), (conc > threshold).astype(np.int8)),
         "aice_jja": (("nens", "year", "nx", "ny"), conc)},
        coords={"nens": np.arange(conc.shape[0]), "year": years,
                "nx": np.arange(atm_lat.size), "ny": np.arange(atm_lon.size),
                "lat": ("nx", atm_lat), "lon": ("ny", atm_lon)},
    )
    ds.attrs.update(description="JJA-mean CESM2-LE sea-ice concentration, nearest-neighbour on the "
                                "atmosphere grid; icemask = concentration > threshold",
                    threshold=float(threshold), regrid="nearest neighbour from CICE TLAT/TLON")
    return ds


def load_icemask(path: Path, years: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
    """Boolean ice-cover mask (nens, nyear, nx, ny) for the requested years."""
    with xr.open_dataset(path) as ds:
        sub = ds.sel(year=years)
        if threshold is None or np.isclose(threshold, float(ds.attrs["threshold"])):
            return sub["icemask"].values.astype(bool)
        return (sub["aice_jja"].values > threshold)


def apply_openwater(sst_anom: np.ndarray, icemask: np.ndarray) -> np.ndarray:
    """Zero the SST anomaly where ice-covered; shapes (nens, nyear, nx, ny)."""
    if sst_anom.shape != icemask.shape:
        raise ValueError(f"SST {sst_anom.shape} and ice mask {icemask.shape} shapes differ")
    return np.where(icemask, 0.0, sst_anom).astype(sst_anom.dtype)
