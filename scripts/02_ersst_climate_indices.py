#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ERSSTv5 Climate Indices Calculation

This script loads regridded ERSSTv5 SST and 
calculates climate indices (IPO, Nino3.4, etc.).

Usage
-----
    python scripts/02_ersst_climate_indices.py


Author: Lauren Hoffman
Email:  lhoffma2@ucscledu
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import netCDF4 as nc

# ---------------------------------------------------------------------------
# Project root on sys.path so imports work regardless of working directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths

ERSST_REGRID = paths.ERSST_REGRIDDED

from src.data.observations.ersst.climate_indices import (
    compute_nino34_index,
    compute_enso_cp_tp_indices,
    compute_ipo_index,
    compute_arctic_sst_index,
    load_grid_latlon,
    save_nino34,
    save_enso_cp_tp,
    save_ipo,
    save_arctic_sst,
)
from src.data.observations.ersst.download import load_ersst

ALL_INDICES = ['nino34', 'enso_cptp', 'ipo', 'arctic_sst']


def _sst_dir() -> Path:
    """Per-month SST files (output of 01_ersstv5_preprocessing.py)."""
    return paths.ERSST_REGRIDDED 


def _grid_file() -> Path:
    """Grid file produced by scripts/01_cesm2le_grid.py."""
    return paths.CESM2LE_SST_DIR / 'grid' / 'cesm2le_sst_grid.nc'


def _indices_dir() -> Path:
    """Output directory for climate index NetCDF files."""
    return paths.ERSST_DIR 

def _cesm_forced_dir() -> Path:
    """Directory containing CESM2-LE forced ensemble mean indices."""
    return paths.CESM2LE_DIR / 'climate_indices' / 'cesm2le_arctic_sst_forced_em.nc'

# ---------------------------------------------------------------------------
# Index computation
# ---------------------------------------------------------------------------

def compute_nino34(sst_dir, lat, lon, years) -> None:
    """Compute Niño3.4 index and save."""
    print('\n[1/3] Computing Niño3.4 index ...')
    nino34, labels = compute_nino34_index(sst_dir, lat, lon, years)

    output_file = str(_indices_dir() / 'ersstv5_nino34_index.nc')
    save_nino34(nino34, labels, years, output_file)


def compute_enso_cptp(sst_dir, lat, lon, years) -> None:
    """Compute ENSO CP/TP indices and save."""
    print('\n[2/3] Computing ENSO CP/TP indices ...')
    result = compute_enso_cp_tp_indices(sst_dir, lat, lon, years)

    output_file = str(_indices_dir() / 'ersstv5_enso_cptp_indices.nc')
    save_enso_cp_tp(result, years, output_file)


def compute_ipo(sst_dir, lat, lon, years) -> None:
    """Compute IPO index and save."""
    print('\n[3/4] Computing IPO index ...')
    ipo, ipo_filtered, labels, labels_filtered = compute_ipo_index(
        sst_dir, lat, lon, years
    )

    output_file = str(_indices_dir() / 'ersstv5_ipo_index.nc')
    save_ipo(ipo, ipo_filtered, labels, labels_filtered, years, output_file)


def compute_arctic(sst_dir, lat, lon, years) -> None:
    """Compute Arctic SST index and save."""
    print('\n[4/4] Computing Arctic SST index ...')
    arctic_sst, labels = compute_arctic_sst_index(sst_dir, _cesm_forced_dir(), lat, lon, years)

    output_file = str(_indices_dir() / 'ersstv5_arctic_sst_index.nc')
    save_arctic_sst(arctic_sst, labels, years, output_file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compute ENSO and IPO climate indices from ERSSTv5 SST.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--index', '-i',
        nargs='+',
        choices=ALL_INDICES,
        default=None,
        metavar='INDEX',
        help='Which indices to compute: nino34, enso_cptp, ipo '
             '(default: all three).',
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    indices = args.index or ALL_INDICES

    print('\n' + '=' * 70)
    print('ERSSTv5 Climate Indices')
    print('=' * 70)
    print(f'  Indices    : {indices}')
    print(f'  SST dir    : {_sst_dir()}')
    print(f'  Grid file  : {_grid_file()}')
    print(f'  Output dir : {_indices_dir()}')
    print('=' * 70)

    _indices_dir().mkdir(parents=True, exist_ok=True)

    # Load ERSSTv5
    print('\nLoading ERSSTv5 ...')
    ds = nc.Dataset(ERSST_REGRID, 'r')
    sst = ds.variables['sst_obs'][:]
    lat = ds.variables['lat_cesm2'][:]
    lon = ds.variables['lon_cesm2'][:]

    ds.close()

    ntime  = sst.shape[0]
    dates  = pd.date_range(start='1854-01-01', periods=ntime, freq='MS')
    years  = dates.year.to_numpy()

    print(f'  sst shape : {sst.shape}')

    if 'nino34' in indices:
        compute_nino34(sst, lat, lon, years)

    if 'enso_cptp' in indices:
        compute_enso_cptp(sst, lat, lon, years)

    if 'ipo' in indices:
        compute_ipo(sst, lat, lon, years)

    if 'arctic_sst' in indices:
        compute_arctic(sst, lat, lon, years)

    print('\n' + '=' * 70)
    print('Pipeline complete!')
    print('=' * 70 + '\n')


if __name__ == '__main__':
    main()



