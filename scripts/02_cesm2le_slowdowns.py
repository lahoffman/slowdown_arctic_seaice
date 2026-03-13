#!/usr/bin/env python3
"""
02_cesm2le_slowdowns.py
=======================
Classify CESM2-LE decadal trends in sea ice extent (SIE) and sea ice area
(SIA) as slowdown or RILES events using NSIDC-derived thresholds.

Requires
--------
- Monthly SIE / SIA NetCDF files from 01_cesm2le_preprocessing.py
    DATA_ROOT/cesm2le/aice/metrics/siextentn_{group}members_{MON}.nc
    DATA_ROOT/cesm2le/aice/metrics/siarean_{group}members_{MON}.nc

- NSIDC slowdown thresholds from 01_slowdown_nsidc_sie_sia.py
    DATA_ROOT/nsidc/nsidc_sie_slowdown_thresholds.nc
    DATA_ROOT/nsidc/nsidc_sia_slowdown_thresholds.nc

Outputs (one file per variable per month)
------------------------------------------
    DATA_ROOT/cesm2le/slowdowns/cesm2le_sie_slowdown_riles_{MON}_1990-2100.nc
    DATA_ROOT/cesm2le/slowdowns/cesm2le_sia_slowdown_riles_{MON}_1990-2100.nc

Usage
-----
    python scripts/02_cesm2le_slowdowns.py
    python scripts/02_cesm2le_slowdowns.py --variable sia --months SEP
    python scripts/02_cesm2le_slowdowns.py --variable sia --months JAN MAR SEP

Author: Lauren Hoffman
Email:  lhoffma2@ucsc.edu
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths
from src.data.cesm2le.slowdowns import compute_cesm2le_slowdowns


MONTH_LABELS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']


def _metrics_dir() -> Path:
    """Per-month SIE / SIA files (output of 01_cesm2le_preprocessing.py)."""
    return paths.CESM2LE_AICE_DIR / 'metrics'


def _slowdowns_dir() -> Path:
    """Output directory for slowdown / RILES NetCDF files."""
    return paths.CESM2LE_DIR / 'slowdowns'


# ---------------------------------------------------------------------------
# Per-variable pipeline
# ---------------------------------------------------------------------------

def run_slowdowns(variable: str, months: list) -> None:
    """Classify slowdown and RILES events for one variable (sie or sia)."""
    threshold_file = (paths.NSIDC_SIE_SLOWDOWN_THRESHOLDS if variable == 'sie'
                      else paths.NSIDC_SIA_SLOWDOWN_THRESHOLDS)

    print(f'\n{"=" * 70}')
    print(f'  Variable : {variable.upper()}')
    print(f'  Months   : {months}')
    print(f'  Input    : {_metrics_dir()}')
    print(f'  Thresholds: {threshold_file}')
    print(f'  Output   : {_slowdowns_dir()}')
    print(f'{"=" * 70}')

    compute_cesm2le_slowdowns(
        sie_dir=str(_metrics_dir()),
        threshold_file=str(threshold_file),
        output_dir=str(_slowdowns_dir()),
        months=months,
        variable=variable,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Classify CESM2-LE sea ice trends as slowdowns or RILES.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--variable', '-v',
        choices=['sie', 'sia', 'all'],
        default='all',
        help='Which variable to process (default: all).',
    )
    parser.add_argument(
        '--months', '-m',
        nargs='+',
        choices=MONTH_LABELS,
        default=None,
        metavar='MON',
        help='Calendar months to process, e.g. SEP or JAN MAR SEP '
             '(default: all 12).',
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    months = args.months or MONTH_LABELS
    variables = ['sie', 'sia'] if args.variable == 'all' else [args.variable]

    print('\n' + '=' * 70)
    print('CESM2-LE Slowdown / RILES Classification')
    print('=' * 70)
    print(f'  Variable(s) : {variables}')
    print(f'  Months      : {months}')
    print(f'  Data root   : {paths.DATA_ROOT}')
    print('=' * 70)

    for variable in variables:
        run_slowdowns(variable=variable, months=months)

    print('\n' + '=' * 70)
    print('Pipeline complete!')
    print('=' * 70 + '\n')


if __name__ == '__main__':
    main()
