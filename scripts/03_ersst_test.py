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
    DATA_ROOT/ersst/ersstv5_testing.nc

Usage
-----
    python scripts/03_ersst_test.py
    python scripts/03_ersst_test.py --start-year 1990 --end-year 2024

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

from src.data.observations.ersst.test_cnn import prepare_obs_for_cnn


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DATA_START_YEAR = 1854        # first year in regridded ERSST file
START_YEAR      = 1990
END_YEAR        = 2024
LAND_FILL       = -10.0       # sentinel value used during CNN training


def _output_file() -> Path:
    """Default output path (NetCDF)."""
    return paths.ERSST_DIR / 'ersstv5_testing.nc'


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
    ds["sst"].attrs["long_name"] = "Standardised JJA SST (land = -10)"
    ds["sst_jja"].attrs["long_name"] = "Raw JJA seasonal-mean SST"
    ds["sst_residual"].attrs["long_name"] = "JJA SST after linear detrending"
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
        '--start-year', type=int, default=START_YEAR,
        help=f'First year of the observation window (default: {START_YEAR}).',
    )
    parser.add_argument(
        '--end-year', type=int, default=END_YEAR,
        help=f'Last year of the observation window (default: {END_YEAR}).',
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Override output file path (default: DATA_ROOT/ersst/ersstv5_testing.nc).',
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    output = Path(args.output) if args.output else _output_file()

    print('\n' + '=' * 70)
    print('ERSSTv5 CNN Testing Data')
    print('=' * 70)
    print(f'  ERSST regridded : {paths.ERSST_REGRIDDED}')
    print(f'  Land mask       : {paths.LANDMASK_FILE}')
    print(f'  Years           : {args.start_year}–{args.end_year}')
    print(f'  Output          : {output}')
    print('=' * 70)

    print('\nRunning observation preprocessing pipeline ...')
    result = prepare_obs_for_cnn(
        regridded_ersst_path=str(paths.ERSST_REGRIDDED),
        landmask_path=str(paths.LANDMASK_FILE),
        data_start_year=DATA_START_YEAR,
        start_year=args.start_year,
        end_year=args.end_year,
        land_fill_value=LAND_FILL,
    )

    print('\nSaving ...')
    save_obs_testing(result, str(output))

    print('\n' + '=' * 70)
    print('Pipeline complete!')
    print(f'  Output shape : {result["X"].shape}  (nsamples, nx, ny, nch)')
    print(f'  mu={result["mu"]:.4f}  sigma={result["sigma"]:.4f}')
    print('=' * 70 + '\n')


if __name__ == '__main__':
    main()
