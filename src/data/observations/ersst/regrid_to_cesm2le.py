#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ERSSTv5 Regridding to CESM2-LE Grid

Regrids ERSSTv5 (2° resolution, regular lat/lon) to the CESM2-LE
atmospheric grid (192×288, regular lat/lon) using bilinear interpolation.

Author: Lauren Hoffman
Email: lhoffma2@ucsc.edu
"""

import numpy as np
import xarray as xr
import netCDF4 as nc
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path
from typing import Optional, Tuple

try:
    from .download import load_ersst
except ImportError:
    from download import load_ersst


def load_cesm2le_grid(grid_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load lat/lon from a CESM2-LE file to use as regrid target.

    NOTE: Use an atmospheric cam.h0 file. The ocean pop.h grid is
    curvilinear (2D lat/lon) and 3x larger — RegularGridInterpolator
    requires a regular 1D target grid.

    Parameters
    ----------
    grid_file : str
        Path to a CESM2-LE atmospheric (cam.h0) NetCDF file

    Returns
    -------
    lat_cesm2 : np.ndarray
        2D latitude array  (192 x 288 values for cam.h0)
    lon_cesm2 : np.ndarray
        2D longitude array (192 x 288 values for cam.h0, 0–358.75°E)
    """
    dataset = nc.Dataset(grid_file, 'r')

    lat_cesm2 = np.array(dataset.variables['lat'])
    lon_cesm2 = np.array(dataset.variables['lon'])
    grid_type = 'atmospheric (cam.h0)'


    dataset.close()

    print(f"CESM2-LE grid ({grid_type}): {lat_cesm2.shape[0]} lat × {lon_cesm2.shape[1]} lon")
    return lat_cesm2, lon_cesm2


def regrid_ersst_to_cesm2le(
    sst: np.ndarray,
    lat_src: np.ndarray,
    lon_src: np.ndarray,
    lat_tgt: np.ndarray,
    lon_tgt: np.ndarray
) -> np.ndarray:
    """
    Regrid ERSSTv5 from its native 2° grid to the CESM2-LE atmospheric grid.

    Uses bilinear interpolation via RegularGridInterpolator. Points outside
    the source domain (e.g., poles) are filled with NaN.

    Parameters
    ----------
    sst : np.ndarray
        ERSSTv5 SST data, shape (ntime, nlat_src, nlon_src)
    lat_src : np.ndarray
        Source latitude array (ERSSTv5, 88 values)
    lon_src : np.ndarray
        Source longitude array (ERSSTv5, 180 values, 0-358°E)
    lat_tgt : np.ndarray
        Target latitude array (CESM2-LE, 192 values)
    lon_tgt : np.ndarray
        Target longitude array (CESM2-LE, 288 values, 0-358.75°E)

    Returns
    -------
    np.ndarray
        Regridded SST, shape (ntime, nlat_tgt, nlon_tgt)

    Examples
    --------
    >>> sst_regridded = regrid_ersst_to_cesm2le(sst, lat_src, lon_src, lat_tgt, lon_tgt)
    >>> print(sst_regridded.shape)  # (ntime, 192, 288)
    """
    nt = sst.shape[0]
    nlat_tgt = lat_tgt.shape[0]
    nlon_tgt = lon_tgt.shape[1]

    # Build 2D target grid and flatten into (N, 2) point array
    target_points = np.stack(
        [lat_tgt.ravel(), lon_tgt.ravel()], axis=-1
    )

    sst_regridded = np.empty((nt, nlat_tgt, nlon_tgt))

    print(f"Regridding {nt} time steps to CESM2-LE grid ({nlat_tgt}×{nlon_tgt})...")
    for t in range(nt):
        if (t + 1) % 120 == 0 or t == 0:
            print(f"  Time step {t+1}/{nt}")

        interp_func = RegularGridInterpolator(
            (lat_src, lon_src),
            sst[t],
            method='linear',
            bounds_error=False,
            fill_value=np.nan
        )
        interpolated = interp_func(target_points)
        sst_regridded[t] = interpolated.reshape(nlat_tgt, nlon_tgt)

    print(f"✓ Regridding complete. Output shape: {sst_regridded.shape}")
    return sst_regridded


def save_regridded_ersst(
    sst_regridded: np.ndarray,
    lat_cesm2: np.ndarray,
    lon_cesm2: np.ndarray,
    dates: pd.DatetimeIndex,
    output_file: str
) -> None:
    """
    Save regridded ERSSTv5 data to NetCDF.

    Parameters
    ----------
    sst_regridded : np.ndarray
        Regridded SST, shape (ntime, nlat, nlon)
    lat_cesm2 : np.ndarray
        CESM2-LE latitude array
    lon_cesm2 : np.ndarray
        CESM2-LE longitude array
    dates : pd.DatetimeIndex
        Monthly datetime index
    output_file : str
        Output NetCDF file path

    Examples
    --------
    >>> save_regridded_ersst(
    ...     sst_regridded, lat_cesm2, lon_cesm2, dates,
    ...     output_file='/cofast/lhoffman/obs/ersst/ersstv5_regridded_to_cesm2_1854-2024.nc'
    ... )
    """
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    ds = xr.Dataset(
        {
            "sst_obs": (("nte", "nx", "ny"), sst_regridded),
            "lat_cesm2": (("nx","ny"), lat_cesm2),
            "lon_cesm2": (("nx","ny"), lon_cesm2),
        },
        coords={
            "nte": np.arange(sst_regridded.shape[0]),
            "nx": np.arange(sst_regridded.shape[1]),
            "ny": np.arange(sst_regridded.shape[2]),
        },
    )

    ds.attrs['description'] = 'ERSSTv5 SST regridded to CESM2-LE atmospheric grid'
    ds.attrs['source'] = 'NOAA ERSSTv5'
    ds.attrs['regrid_method'] = 'bilinear (RegularGridInterpolator)'
    ds.attrs['date_range'] = f"{dates[0].strftime('%Y-%m')} to {dates[-1].strftime('%Y-%m')}"
    ds.attrs['target_grid'] = 'CESM2-LE f09_g17 (192x288)'

    encoding = {var: {"zlib": True, "complevel": 4} for var in ds.data_vars}
    ds.to_netcdf(output_file, format='NETCDF4', encoding=encoding)

    print(f"✓ Saved regridded ERSSTv5 to: {output_file}")
    print(f"  Shape: {sst_regridded.shape} (ntime, nlat, nlon)")

def process_ersst_regrid(
    ersst_file: str,
    output_file: str,
    cesm2le_grid_file: Optional[str] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
) -> None:
    """
    Complete pipeline: load ERSSTv5, regrid to CESM2-LE grid, and save.

    Parameters
    ----------
    ersst_file : str
        Path to raw ERSSTv5 file (sst.mnmean.nc)
    output_file : str
        Output NetCDF file path
    cesm2le_grid_file : str, optional
        Path to a CESM2-LE atmospheric (cam.h0) file to extract the target
        grid from. If None, uses the standard CESM2-LE f09_g17 grid (192×288).
    start_year : int, optional
        First year to include (default: 1854)
    end_year : int, optional
        Last year to include (default: last full year)

    Examples
    --------
    >>> # Without a grid file (uses standard CESM2-LE atmospheric grid)
    >>> process_ersst_regrid(
    ...     ersst_file='/path/to/sst.mnmean.nc',
    ...     output_file='/path/to/ersstv5_regridded_to_cesm2_1854-2024.nc'
    ... )
    """
    print("="*70)
    print("ERSSTv5 → CESM2-LE Regridding Pipeline")
    print("="*70 + "\n")

    # Load ERSSTv5
    print("Step 1: Loading ERSSTv5...")
    sst, lat_src, lon_src, dates = load_ersst(
        ersst_file, start_year=start_year, end_year=end_year
    )

    # Load target grid
    print("\nStep 2: Loading CESM2-LE target grid...")
    if cesm2le_grid_file is not None:
        lat_tgt, lon_tgt = load_cesm2le_grid(cesm2le_grid_file)
    else:
        print("WARNING: No CESM2-LE grid file provided.")

    # Regrid
    print("\nStep 3: Regridding...")
    sst_regridded = regrid_ersst_to_cesm2le(sst, lat_src, lon_src, lat_tgt, lon_tgt)

    # Save
    print("\nStep 4: Saving...")
    save_regridded_ersst(sst_regridded, lat_tgt, lon_tgt, dates, output_file)

    print("\n" + "="*70)
    print("✓ Pipeline complete!")
    print("="*70)
