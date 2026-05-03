#!/usr/bin/env python3
"""
06_cnn_predict_ersst.py
=======================
Precompute CNN predictions on the ERSSTv5 observational data for every
split × seed combination and save them to NetCDF.

For each model (split_idx × run_idx) the script:
1. Loads the ERSSTv5 CNN-ready testing data (standardised, land-masked SST).
2. Loads the trained model.
3. Runs prediction on the observational data.
4. Saves predicted probabilities and binary predictions (at threshold = 0.5,
   the default sigmoid threshold used in the obs notebook) to NetCDF.

Outputs
-------
    RESULTS_DIR/predictions/ersst/forced_{method}/cnn_prediction_ersst_M{split}_{seed}.nc

Each file contains:
    y_prob       (n_years,)  — predicted probability (sigmoid output)
    y_pred       (n_years,)  — binary prediction at default threshold (0.5)
    years        (n_years,)  — observation years
    threshold    scalar      — classification threshold used

Global attributes store split_idx, run_idx, seed, forced_method, and
source data file.

Dependencies
------------
Requires outputs of:
    scripts/03_ersst_test.py           (ERSSTv5 testing data)
    scripts/04_cesm2le_cnn_train.py    (trained models)

Usage
-----
    python scripts/06_cnn_predict_ersst.py
    python scripts/06_cnn_predict_ersst.py --forced-method linear
    python scripts/06_cnn_predict_ersst.py --forced-method ensmean linear

Author: Lauren Hoffman  <lhoffma2@ucsc.edu>
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths
from src.cnn.train import load_model
from src.data.observations.ersst.test_cnn import FORCED_METHODS


# =============================================================================
# Configuration
# =============================================================================

N_SPLITS  = 9
N_SEEDS   = 5
BASE_SEED = 42

# Default classification threshold for observations (sigmoid midpoint).
THRESHOLD = 0.5


# =============================================================================
# Helpers
# =============================================================================

def load_obs_input(testing_file):
    """
    Load the ERSSTv5 CNN-ready observation input.

    Returns
    -------
    X_obs : np.ndarray, shape (n_years, nx, ny, 1)
        Standardised, land-masked SST with channel dimension.
    obs_years : np.ndarray, shape (n_years,)
        Year labels.
    """
    ds = xr.open_dataset(testing_file)
    sst_obs   = ds['sst'].values               # (n_years, nx, ny)
    obs_years = ds['years'].values              # (n_years,)
    ds.close()

    # Add channel dimension: (n_years, nx, ny) → (n_years, nx, ny, 1)
    X_obs = sst_obs[..., np.newaxis].astype(np.float32)
    return X_obs, obs_years


def save_prediction(
    out_path,
    y_prob,
    obs_years,
    threshold,
    split_idx,
    run_idx,
    testing_file,
    forced_method,
):
    """Save one model's ERSSTv5 predictions to NetCDF."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = xr.Dataset(
        {
            'y_prob': ('year', y_prob.astype(np.float32)),
            'y_pred': ('year', (y_prob >= threshold).astype(np.int8)),
            'threshold': ((), np.float32(threshold)),
        },
        coords={
            'year': obs_years,
        },
        attrs={
            'split_idx': split_idx,
            'run_idx':   run_idx,
            'seed':      BASE_SEED + run_idx,
            'forced_method': forced_method,
            'testing_file': str(testing_file),
            'description': (
                f'CNN predictions on ERSSTv5 observations for split {split_idx}, '
                f'seed {BASE_SEED + run_idx} '
                f'(forced removal: {forced_method}). '
                f'Binary predictions use threshold = {threshold}.'
            ),
        },
    )

    ds['y_prob'].attrs['long_name'] = 'Predicted probability of slowdown'
    ds['y_pred'].attrs['long_name'] = 'Binary prediction (1 = slowdown)'
    ds['threshold'].attrs['long_name'] = 'Classification threshold'
    ds['year'].attrs['long_name'] = 'Observation year'

    encoding = {v: {'zlib': True, 'complevel': 4}
                for v in ds.data_vars if ds[v].ndim > 0}
    ds.to_netcdf(out_path, format='NETCDF4', encoding=encoding)


def run_predictions(forced_method: str) -> int:
    """
    Run CNN predictions for one forced method.  Returns number of files saved.
    """
    testing_file = paths.ersst_testing_file(forced_method)
    predictions_dir = paths.ersst_predictions_dir(forced_method)

    print(f'\n{"─" * 70}')
    print(f'Forced method : {forced_method}')
    print(f'Testing file  : {testing_file}')
    print(f'Output dir    : {predictions_dir}')
    print(f'{"─" * 70}')

    if not testing_file.exists():
        raise FileNotFoundError(
            f'ERSSTv5 testing file not found: {testing_file}\n'
            f'Run:  python scripts/03_ersst_test.py --forced-method {forced_method}'
        )

    predictions_dir.mkdir(parents=True, exist_ok=True)

    X_obs, obs_years = load_obs_input(testing_file)
    print(f'  Observation input: {X_obs.shape}  '
          f'years {obs_years[0]}–{obs_years[-1]}')

    n_saved = 0

    for split_idx in range(N_SPLITS):
        print(f'\n  Split {split_idx}')

        for run_idx in range(N_SEEDS):
            model_p = paths.model_path(split_idx, run_idx)
            if not model_p.exists():
                print(f'    [skip] {model_p.name} not found')
                continue

            print(f'    seed {BASE_SEED + run_idx}: ', end='', flush=True)

            model = load_model(paths.MODELS_DIR, split_idx, run_idx)
            y_prob = model.predict(X_obs, verbose=0).ravel()
            print('predicted', end=' → ', flush=True)

            out_path = (predictions_dir /
                        f'cnn_prediction_ersst_M{split_idx}_{run_idx}.nc')
            save_prediction(
                out_path, y_prob, obs_years, THRESHOLD,
                split_idx, run_idx, testing_file, forced_method,
            )
            print('saved')
            n_saved += 1

    return n_saved


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Precompute CNN predictions on ERSSTv5 observations.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--forced-method', type=str, nargs='+',
        choices=FORCED_METHODS, default=['ensmean'],
        help=(
            'Forced-removal method(s) to run predictions for. '
            'Pass one or both: ensmean linear  (default: ensmean).'
        ),
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    print()
    print('=' * 70)
    print('06  —  Precompute CNN predictions on ERSSTv5 observations')
    print('=' * 70)
    print(f'  Data root      : {paths.DATA_ROOT}')
    print(f'  N splits       : {N_SPLITS}')
    print(f'  N seeds        : {N_SEEDS}  (seeds {BASE_SEED}–{BASE_SEED + N_SEEDS - 1})')
    print(f'  Models dir     : {paths.MODELS_DIR}')
    print(f'  Forced methods : {args.forced_method}')
    print(f'  Threshold      : {THRESHOLD}')
    print('=' * 70)

    total_saved = 0
    for method in args.forced_method:
        n = run_predictions(method)
        total_saved += n

    print()
    print('=' * 70)
    print(f'Done.  Saved {total_saved} prediction files total.')
    print('=' * 70 + '\n')


if __name__ == '__main__':
    main()
