#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ERSSTv5 Observation-Based CNN Testing Data Preparation

Runs the observation preprocessing pipeline from
``src.data.observations.ersst.test_cnn`` and saves the CNN-ready testing
array to NetCDF.

Input
-----
Regridded ERSSTv5 (output of ``scripts/01_ersst_preprocessing.py``):
    DATA_ROOT/ersst/sst_regrid_cesm2le.nc

Output
------
    DATA_ROOT/ersst/ersstv5_testing_forced_{method}.nc

    where {method} is 'ensmean' or 'linear' (set via --forced-method).

Usage
-----
    python scripts/03_ersst_test.py
    python scripts/03_ersst_test.py --forced-method linear
    python scripts/03_ersst_test.py --forced-method ensmean --start-year 1990 --end-year 2024

Author: Lauren Hoffman
Email:  lhoffma2@ucsc.edu
"""

import sys
import argparse
import numpy as np
import xarray as xr
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root on sys.path so imports work regardless of working directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths

from src.data.observations.ersst.test_cnn import prepare_obs_for_cnn, FORCED_METHODS


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DATA_START_YEAR = 1854        # first year in regridded ERSST file
START_YEAR      = 1990
END_YEAR        = 2024
LAND_FILL       = -10.0       # sentinel value used during CNN training


def _output_file(forced_method: str) -> Path:
    """Default output path (NetCDF), tagged by forced method."""
    return paths.ERSST_DIR / f'ersstv5_testing_forced_{forced_method}.nc'


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_obs_testing(
    result: dict,
    output_file: str,
) -> None:
    """
    Save the observation-based CNN testing data to NetCDF.

    Stores the CNN-ready input array ``X`` along with intermediate products
    (raw JJA means, detrended residuals) and normalisation statistics for
    reproducibility and diagnostics.

    Parameters
    ----------
    result : dict
        Output of ``prepare_obs_for_cnn()``.
    output_file : str or Path
        Destination NetCDF path.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    nyear, nx, ny, nch = result['X'].shape
    years = result['years']
    forced_method = result['forced_method']

    forced_label = {
        'ensmean': 'ensemble-mean forced removal',
        'linear':  'per-pixel linear detrending',
    }

    ds = xr.Dataset(
        {
            # CNN-ready input (with channel dim squeezed for storage)
            "sst": (("ntr", "nx", "ny"), result['X'][:, :, :, 0]),
            # Intermediate products
            "sst_jja":      (("ntr", "nx", "ny"), result['sst_jja']),
            "sst_residual": (("ntr", "nx", "ny"), result['sst_residual']),
            "sst_std":      (("ntr", "nx", "ny"), result['sst_std']),
            # Normalisation statistics
            "mu":    ((), np.float32(result['mu'])),
            "sigma": ((), np.float32(result['sigma'])),
        },
        coords={
            "ntr": np.arange(nyear),
            "nx":  np.arange(nx),
            "ny":  np.arange(ny),
            "years": (("ntr",), years),
        },
    )

    ds.attrs['description'] = (
        'ERSSTv5 observation-based JJA SST processed for CNN inference. '
        'Variable "sst" is standardised and land-masked (sentinel = -10). '
        'Add a trailing channel dimension before feeding to the CNN.'
    )
    ds.attrs['forced_method'] = forced_method
    ds["sst"].attrs["long_name"] = "Standardised JJA SST (land = -10)"
    ds["sst_jja"].attrs["long_name"] = "Raw JJA seasonal-mean SST"
    ds["sst_residual"].attrs["long_name"] = (
        f"JJA SST after {forced_label.get(forced_method, forced_method)}"
    )
    ds["sst_std"].attrs["long_name"] = "JJA SST after standardisation (before land fill)"
    ds["mu"].attrs["long_name"]  = "Global ocean mean used for standardisation"
    ds["mu"].attrs["units"]      = "degC"
    ds["sigma"].attrs["long_name"] = "Global ocean std used for standardisation"
    ds["sigma"].attrs["units"]     = "degC"

    encoding = {v: {"zlib": True, "complevel": 4}
                for v in ds.data_vars if ds[v].ndim > 0}
    ds.to_netcdf(str(output_file), format='NETCDF4', encoding=encoding)
    print(f"  Saved testing data -> {output_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Prepare ERSSTv5 observation-based testing data for CNN inference.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--forced-method', type=str,
        choices=FORCED_METHODS, default='ensmean',
        help=(
            'Method for removing the forced signal: '
            '"ensmean" (CESM2-LE ensemble mean, default) or '
            '"linear" (per-pixel linear trend).'
        ),
    )
    parser.add_argument(
        '--start-year', type=int, default=START_YEAR,
        help=f'First year of the observation window (default: {START_YEAR}).',
    )
    parser.add_argument(
        '--end-year', type=int, default=END_YEAR,
        help=f'Last year of the observation window (default: {END_YEAR}).',
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Override output file path (default: auto-named by forced method).',
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    output = Path(args.output) if args.output else _output_file(args.forced_method)

    # Build kwargs for prepare_obs_for_cnn
    pipeline_kwargs = dict(
        regridded_ersst_path=str(paths.ERSST_REGRIDDED),
        landmask_path=str(paths.LANDMASK_FILE),
        forced_method=args.forced_method,
        data_start_year=DATA_START_YEAR,
        start_year=args.start_year,
        end_year=args.end_year,
        land_fill_value=LAND_FILL,
    )
    if args.forced_method == 'ensmean':
        pipeline_kwargs['ensmean_path'] = str(paths.CESM2LE_ENSMEAN_JJA)

    print('\n' + '=' * 70)
    print('ERSSTv5 CNN Testing Data')
    print('=' * 70)
    print(f'  ERSST regridded : {paths.ERSST_REGRIDDED}')
    print(f'  Land mask       : {paths.LANDMASK_FILE}')
    print(f'  Forced method   : {args.forced_method}')
    if args.forced_method == 'ensmean':
        print(f'  Ensemble mean   : {paths.CESM2LE_ENSMEAN_JJA}')
    print(f'  Years           : {args.start_year}–{args.end_year}')
    print(f'  Output          : {output}')
    print('=' * 70)

    print('\nRunning observation preprocessing pipeline ...')
    result = prepare_obs_for_cnn(**pipeline_kwargs)

    print('\nSaving ...')
    save_obs_testing(result, str(output))

    print('\n' + '=' * 70)
    print('Pipeline complete!')
    print(f'  Output shape : {result["X"].shape}  (nsamples, nx, ny, nch)')
    print(f'  mu={result["mu"]:.4f}  sigma={result["sigma"]:.4f}')
    print('=' * 70 + '\n')


if __name__ == '__main__':
    main()
