#!/usr/bin/env python3
"""
02_cesm2le_slowdowns_gmt.py
===========================
Classify CESM2-LE decadal trends in yearly mean global mean temperature (GMT)
as slowdown events.

A GMT slowdown is defined as a 10-year trend that is anomalously *low*
(warming is slower than expected from the forced trend).  This is the sign-
flipped analogue of the SIE slowdown, where ice loss is anomalously *slow*.

Requires
--------
- Yearly mean GMT files from 01_cesm2le_preprocessing.py
    DATA_ROOT/cesm2le/tref/gmt/trefht_gmt_cesmle_{group}members_1990-2100.nc

Outputs
-------
    DATA_ROOT/cesm2le/slowdowns/cesm2le_gmt_slowdown_1990-2100.nc

Usage
-----
    python scripts/02_cesm2le_slowdowns_gmt.py
    python scripts/02_cesm2le_slowdowns_gmt.py --trend-start-year 1990

Author: Lauren Hoffman
Email:  lhoffma2@ucsc.edu
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths
from src.data.cesm2le.slowdowns_gmt import compute_cesm2le_gmt_slowdowns


def _gmt_dir() -> Path:
    """Yearly mean GMT files (output of 01_cesm2le_preprocessing.py)."""
    return paths.CESM2LE_TREF_DIR / 'gmt'


def _slowdowns_dir() -> Path:
    """Output directory for slowdown NetCDF files."""
    return paths.CESM2LE_SLOWDOWNS_DIR


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Classify CESM2-LE GMT decadal trends as slowdowns.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--start-year', type=int, default=1990,
        help='First year in the GMT data (default: 1990).',
    )
    parser.add_argument(
        '--end-year', type=int, default=2100,
        help='Last year in the GMT data (default: 2100).',
    )
    parser.add_argument(
        '--trend-start-year', type=int, default=1990,
        help='First year to include in trend output (default: 1990).',
    )
    parser.add_argument(
        '--window', type=int, default=10,
        help='Trend window length in years (default: 10).',
    )
    parser.add_argument(
        '--output-dir', default=None,
        help='Output directory (default: DATA_ROOT/cesm2le/slowdowns).',
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    gmt_dir = _gmt_dir()
    output_dir = Path(args.output_dir) if args.output_dir else _slowdowns_dir()

    print('\n' + '=' * 70)
    print('CESM2-LE GMT Slowdown Classification')
    print('=' * 70)
    print(f'  GMT dir       : {gmt_dir}')
    print(f'  Output dir    : {output_dir}')
    print(f'  Data root     : {paths.DATA_ROOT}')
    print(f'  Year range    : {args.start_year}-{args.end_year}')
    print(f'  Trend start   : {args.trend_start_year}')
    print(f'  Window        : {args.window} yr')
    print('=' * 70)

    compute_cesm2le_gmt_slowdowns(
        gmt_dir=str(gmt_dir),
        output_dir=str(output_dir),
        start_year=args.start_year,
        end_year=args.end_year,
        trend_start_year=args.trend_start_year,
        window=args.window,
    )

    print('\n' + '=' * 70)
    print('Pipeline complete!')
    print('=' * 70 + '\n')


if __name__ == '__main__':
    main()
