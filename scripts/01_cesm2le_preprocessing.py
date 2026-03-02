#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CESM2-LE Data Download and Preprocessing Pipeline

This script downloads and preprocesses CESM2-LE data for SST and AICE,
then computes sea ice extent (SIE) and sea ice area (SIA) from the monthly
AICE files.

Full workflow
-------------
1.  Download  — fetch raw CESM2-LE chunks from UCAR for each variable and
                member group (first50 / last50).
2.  Combine   — concatenate chunks along time → [nens, ntime, nlat, nlon].
3.  Separate  — split combined file into one NetCDF per calendar month
                → {var}_cesmle_{group}members_mon_{MON}_{years}.nc
4.  Metrics   — (AICE only) calculate SIE and SIA from each monthly AICE file
                via batch_process_monthly_files().

Usage examples
--------------
# Process everything (download + preprocess + metrics)
python scripts/01_cesm2le_preprocessing.py --variable all

# SST only (download + preprocess)
python scripts/01_cesm2le_preprocessing.py --variable sst

# AICE only, skip the download step (data already on disk)
python scripts/01_cesm2le_preprocessing.py --variable aice --skip-download

# Only compute sea ice metrics from existing monthly AICE files
python scripts/01_cesm2le_preprocessing.py --variable aice --metrics-only

# Dry-run: print download commands without executing them
python scripts/01_cesm2le_preprocessing.py --variable all --dry-run

# Process only the first 50 ensemble members
python scripts/01_cesm2le_preprocessing.py --variable all --member-groups first50

Author: Lauren Hoffman
Email:  lhoffma2@ucsc.edu
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root on sys.path so imports work regardless of working directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths

from src.data.cesm2le import (
    download_raw_data,
    combine_ensemble_members,
    separate_by_month,
    process_cesmle_variable,
)
from src.data.cesm2le.metrics import batch_process_monthly_files


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _sst_raw_dir() -> Path:
    """Raw SST chunks download directory."""
    return paths.CESM2LE_SST_DIR / 'raw'


def _sst_combined_path(group: str) -> Path:
    """Combined (all-time) SST file for a given member group."""
    return paths.CESM2LE_SST_DIR / 'yearmon' / \
        f'sst_cesmle_{group}members_mon_199001-210012.nc'


def _sst_monthly_dir() -> Path:
    """Directory for per-month SST files."""
    return paths.CESM2LE_SST_DIR / 'mon'


def _aice_raw_dir() -> Path:
    """Raw AICE chunks download directory."""
    return paths.CESM2LE_AICE_DIR / 'raw'


def _aice_combined_path(group: str) -> Path:
    """Combined (all-time) AICE file for a given member group."""
    return paths.CESM2LE_AICE_DIR / 'yearmon' / \
        f'aice_cesmle_{group}members_mon_199001-210012.nc'


def _aice_monthly_dir() -> Path:
    """Directory for per-month AICE files."""
    return paths.CESM2LE_AICE_DIR / 'mon'


def _metrics_dir() -> Path:
    """Output directory for SIE / SIA NetCDF files."""
    return paths.CESM2LE_AICE_DIR / 'metrics'


# ---------------------------------------------------------------------------
# tarea / grid file for sea ice metrics
# ---------------------------------------------------------------------------
# calculate_sea_ice_extent() and calculate_sea_ice_area() need a NetCDF that
# contains:
#   • tarea  — grid-cell area in cm²
#   • TLAT   — grid-cell latitude
#
# These variables live in the raw AICE files themselves (from the CICE
# component), so we point at the first downloaded raw chunk.  Any raw AICE
# file for any member will do — tarea and TLAT are grid constants.
# ---------------------------------------------------------------------------
_TAREA_FILENAME = (
    'b.e21.BHISTcmip6.f09_g17.LE2-1001.001.cice.h.aice.199001-199912.nc'
)
TAREA_FILE = str(_aice_raw_dir() / _TAREA_FILENAME)


# ---------------------------------------------------------------------------
# SST processing
# ---------------------------------------------------------------------------

def download_sst(member_groups: list, dry_run: bool = False) -> None:
    """Download raw SST chunks from UCAR for the requested member groups."""
    print('\n' + '=' * 70)
    print('STEP 1a  —  Download raw SST data from UCAR')
    print('=' * 70)

    raw_dir = _sst_raw_dir()
    raw_dir.mkdir(parents=True, exist_ok=True)

    download_raw_data(
        variable='SST',
        output_dir=str(raw_dir),
        member_groups=member_groups,
        dry_run=dry_run,
    )

    print(f'\n✓ SST raw data {"(dry run)" if dry_run else "downloaded"} → {raw_dir}')


def process_sst(member_groups: list) -> None:
    """Combine raw SST chunks and separate into monthly files."""
    print('\n' + '=' * 70)
    print('STEP 1b  —  Process SST (combine + separate by month)')
    print('=' * 70)

    raw_dir = _sst_raw_dir()
    monthly_dir = _sst_monthly_dir()

    for group in member_groups:
        print(f'\n--- {group} members ---')
        combined_path = _sst_combined_path(group)
        combined_path.parent.mkdir(parents=True, exist_ok=True)

        # Step 1: combine all raw chunks for this member group
        print(f'Combining raw chunks → {combined_path.name}')
        combine_ensemble_members(
            variable='SST',
            raw_data_path=str(raw_dir),
            output_path=str(combined_path),
            member_group=group,
            # component='cam', frequency='h0'  (these are the defaults)
        )

        # Step 2: split combined file into 12 monthly files
        print(f'Separating by month → {monthly_dir}/')
        monthly_dir.mkdir(parents=True, exist_ok=True)
        separate_by_month(
            combined_file=str(combined_path),
            output_dir=str(monthly_dir),
            variable='SST',
            member_label=f'{group}members',
        )

    print('\n✓ SST processing complete')


# ---------------------------------------------------------------------------
# AICE processing
# ---------------------------------------------------------------------------

def download_aice(member_groups: list, dry_run: bool = False) -> None:
    """Download raw AICE chunks from UCAR for the requested member groups."""
    print('\n' + '=' * 70)
    print('STEP 2a  —  Download raw AICE data from UCAR')
    print('=' * 70)

    raw_dir = _aice_raw_dir()
    raw_dir.mkdir(parents=True, exist_ok=True)

    download_raw_data(
        variable='AICE',
        output_dir=str(raw_dir),
        member_groups=member_groups,
        dry_run=dry_run,
    )

    print(f'\n✓ AICE raw data {"(dry run)" if dry_run else "downloaded"} → {raw_dir}')


def process_aice(member_groups: list) -> None:
    """Combine raw AICE chunks and separate into monthly files."""
    print('\n' + '=' * 70)
    print('STEP 2b  —  Process AICE (combine + separate by month)')
    print('=' * 70)

    raw_dir = _aice_raw_dir()
    monthly_dir = _aice_monthly_dir()

    for group in member_groups:
        print(f'\n--- {group} members ---')
        combined_path = _aice_combined_path(group)
        combined_path.parent.mkdir(parents=True, exist_ok=True)

        # Step 1: combine all raw chunks
        # AICE comes from the CICE component → component='cice', frequency='h'
        print(f'Combining raw chunks → {combined_path.name}')
        combine_ensemble_members(
            variable='AICE',
            raw_data_path=str(raw_dir),
            output_path=str(combined_path),
            member_group=group,
            component='cice',
            frequency='h',
        )

        # Step 2: split into monthly files
        print(f'Separating by month → {monthly_dir}/')
        monthly_dir.mkdir(parents=True, exist_ok=True)
        separate_by_month(
            combined_file=str(combined_path),
            output_dir=str(monthly_dir),
            variable='AICE',
            member_label=f'{group}members',
        )

    print('\n✓ AICE processing complete')


# ---------------------------------------------------------------------------
# Sea ice metrics
# ---------------------------------------------------------------------------

def compute_sea_ice_metrics(member_groups: list) -> None:
    """
    Calculate sea ice extent (SIE) and sea ice area (SIA) from monthly AICE
    files using batch_process_monthly_files().

    Output files (one per month per member group):
        siextentn_{group}members_{MON}.nc   — SIE in million km²
        siarean_{group}members_{MON}.nc     — SIA in million km²
    """
    print('\n' + '=' * 70)
    print('STEP 2c  —  Compute sea ice extent and area from monthly AICE files')
    print('=' * 70)

    aice_dir = _aice_monthly_dir()
    metrics_dir = _metrics_dir()
    metrics_dir.mkdir(parents=True, exist_ok=True)

    tarea_file = TAREA_FILE
    if not Path(tarea_file).exists():
        print(
            f'\n⚠  WARNING: tarea/grid file not found:\n   {tarea_file}\n'
            f'   Update TAREA_FILE at the top of this script to an ice grid\n'
            f'   file that contains tarea and TLAT variables, then re-run.'
        )
        return

    for group in member_groups:
        member_label = f'{group}members'
        print(f'\n--- {member_label} ---')

        batch_process_monthly_files(
            aice_dir=str(aice_dir),
            tarea_file=tarea_file,
            output_dir=str(metrics_dir),
            member_label=member_label,
            calculate_extent=True,
            calculate_area=True,
        )

    print(f'\n✓ Sea ice metrics saved to {metrics_dir}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Download and preprocess CESM2-LE data (SST and/or AICE).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        '--variable', '-v',
        choices=['sst', 'aice', 'all'],
        default='all',
        help='Which variable(s) to process (default: all).',
    )
    parser.add_argument(
        '--member-groups', '-m',
        nargs='+',
        choices=['first50', 'last50'],
        default=['first50', 'last50'],
        help='Which ensemble member groups to process (default: both).',
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        default=False,
        help='Skip the download step (raw files must already be present).',
    )
    parser.add_argument(
        '--metrics-only',
        action='store_true',
        default=False,
        help='Only run the sea ice metrics step (skip download and processing).'
             ' Requires monthly AICE files to already exist. Implies --variable aice.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=False,
        help='Print download commands without executing them.',
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --metrics-only implies we only care about AICE
    if args.metrics_only:
        args.variable = 'aice'

    run_sst  = args.variable in ('sst',  'all')
    run_aice = args.variable in ('aice', 'all')

    print('\n' + '=' * 70)
    print('CESM2-LE Preprocessing Pipeline')
    print('=' * 70)
    print(f'  Variable(s)    : {args.variable}')
    print(f'  Member groups  : {args.member_groups}')
    print(f'  Skip download  : {args.skip_download}')
    print(f'  Metrics only   : {args.metrics_only}')
    print(f'  Dry run        : {args.dry_run}')
    print(f'  Data root      : {paths.DATA_ROOT}')
    print('=' * 70)

    # ------------------------------------------------------------------
    # SST
    # ------------------------------------------------------------------
    if run_sst and not args.metrics_only:
        if not args.skip_download:
            download_sst(member_groups=args.member_groups, dry_run=args.dry_run)

        if not args.dry_run:
            process_sst(member_groups=args.member_groups)

    # ------------------------------------------------------------------
    # AICE
    # ------------------------------------------------------------------
    if run_aice:
        if not args.metrics_only:
            if not args.skip_download:
                download_aice(member_groups=args.member_groups, dry_run=args.dry_run)

            if not args.dry_run:
                process_aice(member_groups=args.member_groups)

        if not args.dry_run:
            compute_sea_ice_metrics(member_groups=args.member_groups)

    print('\n' + '=' * 70)
    print('Pipeline complete!')
    print('=' * 70 + '\n')


if __name__ == '__main__':
    main()
