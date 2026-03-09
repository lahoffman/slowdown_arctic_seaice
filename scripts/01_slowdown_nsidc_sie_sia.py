#!/usr/bin/env python3
"""
01_slowdown_nsidc_sie_sia.py
============================
Preprocess NSIDC sea ice extent (SIE) and sea ice area (SIA), compute
10-year sliding decadal trends, define slowdown events (mean + 1σ threshold),
and save results to NetCDF.

Outputs (written to DATA_ROOT/nsidc/):
  nsidc_sie_slowdown_thresholds.nc       — monthly thresholds, all 12 months
  nsidc_sie_slowdown_events_month{MM}.nc — slowdown mask + trend, one per month
  nsidc_sia_slowdown_thresholds.nc
  nsidc_sia_slowdown_events_month{MM}.nc

Usage
-----
  python scripts/01_slowdown_nsidc_sie_sia.py
  python scripts/01_slowdown_nsidc_sie_sia.py --start-year 1990 --window 10

  **
  python scripts/01_slowdown_nsidc_sie_sia.py --year 2025
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths
from src.data.observations.nsidc import (
    preprocess_nsidc_sie,
    compute_decadal_trends,
    define_slowdown_threshold,
    save_slowdown_thresholds,
    save_slowdown_events,
)

MONTH_NAMES = [
    'jan', 'feb', 'mar', 'apr', 'may', 'jun',
    'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute NSIDC SIE/SIA slowdown events from decadal trends.'
    )
    parser.add_argument(
        '--year', type=int, default=None,
        help='Override the current year used to set the data boundary (default: today).'
    )
    parser.add_argument(
        '--window', type=int, default=10,
        help='Trend window length in years (default: 10).'
    )
    parser.add_argument(
        '--start-year', type=int, default=1990,
        help='First year to include in the trend analysis (default: 1990).'
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pipeline for one variable
# ---------------------------------------------------------------------------

def run_pipeline(variable: str, args) -> None:
    """
    Full slowdown pipeline for one variable ('extent' or 'area').

    Steps
    -----
    1. Preprocess monthly NSIDC data from Excel.
    2. Compute 10-year sliding linear trends.
    3. Define slowdown threshold (mean + 1σ).
    4. Save thresholds and per-month event masks to NetCDF.
    """
    label = 'sie' if variable == 'extent' else 'sia'
    print(f"\n{'='*60}")
    print(f"  Variable : {variable.upper()}  ({label.upper()})")
    print(f"  Window   : {args.window} yr   |  Start year : {args.start_year}")
    print(f"{'='*60}")

    # 1. Preprocess -----------------------------------------------------------
    print(f"\n      Input  : {paths.NSIDC_FILE}")
    print(f"      Output : {paths.NSIDC_DIR}/")
    print("\n[1/3] Preprocessing NSIDC data ...")
    data, yearmon, time = preprocess_nsidc_sie(
        str(paths.NSIDC_FILE),
        current_year=args.year,
        variable=variable,
    )
    print(f"      Loaded: {data.shape[1]} years  ({yearmon[0,0]}–{yearmon[0,-1]})")

    # 2. Decadal trends -------------------------------------------------------
    print(f"\n[2/3] Computing {args.window}-yr decadal trends (start {args.start_year}) ...")
    trends, trend_years = compute_decadal_trends(
        data,
        yearmon,
        window=args.window,
        start_year=args.start_year,
    )
    print(f"      Trends shape: {trends.shape}  "
          f"({trend_years[0,0]}–{trend_years[0,-1]})")

    # 3. Slowdown threshold ---------------------------------------------------
    print("\n[3/3] Defining slowdown threshold (mean + 1σ) ...")
    threshold, mask, fraction = define_slowdown_threshold(trends)
    for m in range(12):
        n_events = int(mask[m].sum())
        print(f"      {MONTH_NAMES[m].capitalize():>3}  "
              f"threshold={threshold[m]:+.4f} M km² yr⁻¹  "
              f"fraction={fraction[m]:.3f}  "
              f"n_slowdowns={n_events}")

    # 4. Save -----------------------------------------------------------------
    print("\n  Saving outputs ...")
    import numpy as np

    # Thresholds (all 12 months in one file)
    thr_path = (paths.NSIDC_SIE_SLOWDOWN_THRESHOLDS if variable == 'extent'
                else paths.NSIDC_SIA_SLOWDOWN_THRESHOLDS)
    save_slowdown_thresholds(
        threshold_slowdown=threshold,
        fraction_slowdown=fraction,
        threshold_riles=np.full_like(threshold, np.nan),
        fraction_riles=np.full_like(fraction, np.nan),
        output_file=str(thr_path),
    )

    # Per-month event masks
    events_fn = (paths.nsidc_sie_slowdown_events if variable == 'extent'
                 else paths.nsidc_sia_slowdown_events)
    for m in range(12):
        save_slowdown_events(
            mask=mask,
            linear_trends=trends,
            trend_years=trend_years,
            month_idx=m,
            output_file=str(events_fn(m + 1)),
        )

    print(f"\n  Done — outputs written to: {paths.NSIDC_DIR}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    print(f"Data root : {paths.DATA_ROOT}")

    for variable in ('extent', 'area'):
        run_pipeline(variable, args)

    print("\nAll done.")


if __name__ == '__main__':
    main()
