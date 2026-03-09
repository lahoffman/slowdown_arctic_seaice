#!/usr/bin/env python3
"""
02_cesm2le_climate_indices.py
=============================
Compute ENSO and IPO climate indices from CESM2-LE SST data.

Indices computed
----------------
- Niño3.4      — standard ENSO index (5S–5N, 170W–120W)
- ENSO CP/TP   — Niño3, Niño4, cold-tongue (N_CT), warm-pool (N_WP)
- IPO          — Interdecadal Pacific Oscillation (Henley et al. 2015)

Requires
--------
- Monthly SST NetCDF files from 01_cesm2le_preprocessing.py
    DATA_ROOT/cesm2le/sst/mon/sst_cesmle_{group}members_mon_{MON}_{years}.nc

Outputs (one file per index)
------------------------------
    DATA_ROOT/cesm2le/climate_indices/cesm2le_nino34_index.nc
    DATA_ROOT/cesm2le/climate_indices/cesm2le_enso_cptp_indices.nc
    DATA_ROOT/cesm2le/climate_indices/cesm2le_ipo_index.nc

Usage
-----
    python scripts/02_cesm2le_climate_indices.py
    python scripts/02_cesm2le_climate_indices.py --index nino34
    python scripts/02_cesm2le_climate_indices.py --index ipo enso_cptp

Author: Lauren Hoffman
Email:  lhoffma2@ucsc.edu
"""

import argparse
import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths
from src.data.cesm2le.climate_indices import (
    load_grid_latlon,
    compute_nino34_index,
    compute_enso_cp_tp_indices,
    compute_ipo_index,
    save_nino34,
    save_enso_cp_tp,
    save_ipo,
)

ALL_INDICES = ['nino34', 'enso_cptp', 'ipo']


def _sst_monthly_dir() -> Path:
    """Per-month SST files (output of 01_cesm2le_preprocessing.py)."""
    return paths.CESM2LE_SST_DIR / 'mon'


def _grid_file() -> Path:
    """Grid file produced by scripts/01_cesm2le_grid.py."""
    return paths.CESM2LE_SST_DIR / 'grid' / 'cesm2le_sst_grid.nc'


def _indices_dir() -> Path:
    """Output directory for climate index NetCDF files."""
    return paths.CESM2LE_DIR / 'climate_indices'


# ---------------------------------------------------------------------------
# Index computation
# ---------------------------------------------------------------------------

def compute_nino34(sst_dir, lat, lon, years) -> None:
    """Compute Niño3.4 index and save."""
    print('\n[1/3] Computing Niño3.4 index ...')
    nino34, _, labels, _ = compute_nino34_index(sst_dir, lat, lon, years)

    nino34_jja = nino34[:, :, 5:8]
    labels_jja = labels[:, :, 5:8]

    output_file = str(_indices_dir() / 'cesm2le_nino34_index.nc')
    save_nino34(nino34, labels, years, output_file,
                nino34_jja=nino34_jja, labels_jja=labels_jja)


def compute_enso_cptp(sst_dir, lat, lon, years) -> None:
    """Compute ENSO CP/TP indices and save."""
    print('\n[2/3] Computing ENSO CP/TP indices ...')
    result = compute_enso_cp_tp_indices(sst_dir, lat, lon, years)

    output_file = str(_indices_dir() / 'cesm2le_enso_cptp_indices.nc')
    save_enso_cp_tp(result, years, output_file)


def compute_ipo(sst_dir, lat, lon, years) -> None:
    """Compute IPO index and save."""
    print('\n[3/3] Computing IPO index ...')
    ipo, _, ipo_filtered, _, labels, _, labels_filtered, _ = compute_ipo_index(
        sst_dir, lat, lon, years
    )

    output_file = str(_indices_dir() / 'cesm2le_ipo_index.nc')
    save_ipo(ipo, ipo_filtered, labels, labels_filtered, years, output_file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compute ENSO and IPO climate indices from CESM2-LE SST.',
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
    parser.add_argument(
        '--start-year', type=int, default=1990,
        help='First year in the SST data (default: 1990).',
    )
    parser.add_argument(
        '--end-year', type=int, default=2100,
        help='Last year in the SST data (default: 2100).',
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    indices = args.index or ALL_INDICES
    years   = np.arange(args.start_year, args.end_year + 1)

    print('\n' + '=' * 70)
    print('CESM2-LE Climate Indices')
    print('=' * 70)
    print(f'  Indices    : {indices}')
    print(f'  Years      : {args.start_year}–{args.end_year}')
    print(f'  SST dir    : {_sst_monthly_dir()}')
    print(f'  Grid file  : {_grid_file()}')
    print(f'  Output dir : {_indices_dir()}')
    print('=' * 70)

    _indices_dir().mkdir(parents=True, exist_ok=True)

    # Load 2D lat/lon from the grid file (TLAT / TLONG)
    print('\nLoading grid lat/lon ...')
    lat, lon = load_grid_latlon(_grid_file())
    print(f'  lat/lon shape : {lat.shape}')

    if 'nino34' in indices:
        compute_nino34(_sst_monthly_dir(), lat, lon, years)

    if 'enso_cptp' in indices:
        compute_enso_cptp(_sst_monthly_dir(), lat, lon, years)

    if 'ipo' in indices:
        compute_ipo(_sst_monthly_dir(), lat, lon, years)

    print('\n' + '=' * 70)
    print('Pipeline complete!')
    print('=' * 70 + '\n')


if __name__ == '__main__':
    main()
