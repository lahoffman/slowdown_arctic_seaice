#!/usr/bin/env python3
"""
02_cesm2le_forced.py
====================
Compute and save CESM2-LE forced-response fields.

Currently computes:
  - Ensemble-mean JJA SST on the full 2D atmospheric grid (192 x 288).
    This is the forced component subtracted from each ensemble member
    during CNN training.  Persisting it here lets the observation pipeline
    (``scripts/03_ersst_test.py``) reuse the same field without reloading
    all 100 members.

Requires
--------
Monthly SST NetCDF files from ``scripts/01_cesm2le_preprocessing.py``:
    DATA_ROOT/cesm2le/sst/mon/sst_cesmle_{group}members_mon_{MON}_199001-210012.nc

Output
------
    DATA_ROOT/cesm2le/forced/cesm2le_ensmean_jja_sst.nc

Usage
-----
    python scripts/02_cesm2le_forced.py
    python scripts/02_cesm2le_forced.py --start-year 1990 --end-year 2100

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
from src.data.cesm2le.forced import (
    compute_ensmean_jja_sst,
    save_ensmean_jja_sst,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compute CESM2-LE forced-response fields.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--start-year', type=int, default=1990,
        help='First year in the SST data (default: 1990).',
    )
    parser.add_argument(
        '--end-year', type=int, default=2100,
        help='Last year in the SST data (default: 2100).',
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Override output file path (default: paths.CESM2LE_ENSMEAN_JJA).',
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    years = np.arange(args.start_year, args.end_year + 1)
    output = Path(args.output) if args.output else paths.CESM2LE_ENSMEAN_JJA

    sst_monthly_dir = paths.CESM2LE_SST_DIR / 'mon'

    print('\n' + '=' * 70)
    print('CESM2-LE Forced Response')
    print('=' * 70)
    print(f'  SST dir  : {sst_monthly_dir}')
    print(f'  Years    : {args.start_year}–{args.end_year}')
    print(f'  Output   : {output}')
    print('=' * 70)

    print('\nComputing ensemble-mean JJA SST ...')
    ensmean = compute_ensmean_jja_sst(
        sst_dir=str(sst_monthly_dir),
        years=years,
    )

    print('\nSaving ...')
    save_ensmean_jja_sst(ensmean, years, str(output))

    print('\n' + '=' * 70)
    print('Done!')
    print(f'  Shape : {ensmean.shape}  (nyear, nlat, nlon)')
    print('=' * 70 + '\n')


if __name__ == '__main__':
    main()
