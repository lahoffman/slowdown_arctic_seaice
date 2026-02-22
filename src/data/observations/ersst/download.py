#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ERSSTv5 Download

Download NOAA ERSSTv5 monthly mean SST data.

Source: https://psl.noaa.gov/data/gridded/data.noaa.ersst.v5.html

Author: Lauren Hoffman
Email: lhoffma2@ucsc.edu
"""

import subprocess
import numpy as np
import netCDF4 as nc
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

# NOAA PSL download URL for ERSSTv5
ERSST_URL = "https://downloads.psl.noaa.gov/Datasets/noaa.ersst.v5/sst.mnmean.nc"
ERSST_FILENAME = "sst.mnmean.nc"


def download_ersst(
    output_dir: str,
    dry_run: bool = False
) -> str:
    """
    Download ERSSTv5 monthly mean SST data from NOAA PSL.

    Downloads the full ERSSTv5 dataset (1854-present) as a single file.

    Parameters
    ----------
    output_dir : str
        Directory where the file will be saved
    dry_run : bool, optional
        If True, print command without executing (default: False)

    Returns
    -------
    str
        Path to downloaded file

    Examples
    --------
    >>> filepath = download_ersst(
    ...     output_dir='/cofast/lhoffman/obs/ersst/raw'
    ... )
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filepath = output_path / ERSST_FILENAME

    if filepath.exists():
        print(f"✓ Already exists: {filepath}")
        return str(filepath)

    wget_cmd = ['wget', '-P', output_dir, ERSST_URL]

    if dry_run:
        print(f"[DRY RUN] Would execute: {' '.join(wget_cmd)}")
        return str(filepath)

    print(f"Downloading ERSSTv5 from NOAA PSL...")
    print(f"  URL: {ERSST_URL}")
    print(f"  Destination: {filepath}")

    try:
        subprocess.run(wget_cmd, check=True, capture_output=True, text=True)
        print(f"✓ Downloaded: {filepath}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error downloading ERSSTv5: {e.stderr}")
        raise

    return str(filepath)


def load_ersst(
    filepath: str,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Load ERSSTv5 data and return cleaned arrays.

    Handles missing value masking (values < -9e3 set to NaN) and
    optional date subsetting.

    Parameters
    ----------
    filepath : str
        Path to ERSSTv5 NetCDF file (sst.mnmean.nc)
    start_year : int, optional
        First year to include (default: first available = 1854)
    end_year : int, optional
        Last year to include (default: last full year available)

    Returns
    -------
    sst : np.ndarray
        SST data, shape (ntime, nlat, nlon), missing values as NaN
    lat : np.ndarray
        Latitude array (88 values, 2° resolution)
    lon : np.ndarray
        Longitude array (180 values, 2° resolution, 0-358°E)
    dates : pd.DatetimeIndex
        Monthly datetime index

    Examples
    --------
    >>> sst, lat, lon, dates = load_ersst(
    ...     filepath='/cofast/lhoffman/obs/ersst/raw/sst.mnmean.nc',
    ...     start_year=1854,
    ...     end_year=2024
    ... )
    >>> print(sst.shape)  # (ntime, 88, 180)
    """
    dataset = nc.Dataset(filepath, 'r')

    lat = np.array(dataset.variables['lat'])
    lon = np.array(dataset.variables['lon'])
    sst_raw = np.array(dataset.variables['sst'])
    time_raw = np.array(dataset.variables['time'])

    dataset.close()

    # Convert time (days since 1800-01-01) to DatetimeIndex
    dates_full = pd.to_datetime(time_raw, unit='d', origin='1800-01-01')

    # Drop last 6 months (typically incomplete / preliminary data)
    sst = sst_raw[:-6]
    dates = dates_full[:-6]

    # Mask missing values
    sst = sst.astype(float)
    sst[sst < -9e3] = np.nan

    # Optional date subsetting
    if start_year is not None or end_year is not None:
        years = dates.year
        mask = np.ones(len(dates), dtype=bool)
        if start_year is not None:
            mask &= years >= start_year
        if end_year is not None:
            mask &= years <= end_year
        sst = sst[mask]
        dates = dates[mask]

    print(f"Loaded ERSSTv5: {sst.shape} (ntime, nlat, nlon)")
    print(f"  Date range: {dates[0].strftime('%Y-%m')} to {dates[-1].strftime('%Y-%m')}")
    print(f"  Grid: {len(lat)} lat × {len(lon)} lon")

    return sst, lat, lon, dates
