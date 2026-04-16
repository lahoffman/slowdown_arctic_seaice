#!/usr/bin/env python3
"""
06_cnn_predict_cesm2le.py
=========================
Precompute CNN predictions on the CESM2-LE training / validation / test data
for every split × seed combination and save them to NetCDF.

This script extracts the inference step that is currently repeated inside
several notebooks (cnn_predict.ipynb, cnn_predict_pdfs.ipynb, xai_kmeans.ipynb)
so those notebooks can reload the predictions without re-running all 45 models.

For each model (split_idx × run_idx) the script:
1. Loads the TVT split that corresponds to the model's split index.
2. Loads the trained model.
3. Runs prediction on train / val / test data.
4. Computes the PR-curve-based optimal threshold (precision ≈ recall).
5. Saves predicted probabilities, thresholded binary predictions, and true
   labels to a single NetCDF file.

Outputs
-------
    PREDICTIONS_DIR/cesm2le/cnn_prediction_cesm2le_M{split}_{seed}.nc

Each file contains:
    y_prob_train   (n_train,)   — predicted probability (sigmoid output)
    y_prob_val     (n_val,)
    y_prob_test    (n_test,)
    y_pred_train   (n_train,)   — binary prediction at optimal threshold
    y_pred_val     (n_val,)
    y_pred_test    (n_test,)
    y_true_train   (n_train,)   — ground-truth slowdown labels
    y_true_val     (n_val,)
    y_true_test    (n_test,)
    threshold      scalar       — PR-curve optimal threshold

Global attributes store split_idx, run_idx, seed, and the TVT split file used.

Dependencies
------------
Requires outputs of:
    scripts/03_cesm2le_tvt_splits.py   (TVT splits)
    scripts/04_cesm2le_cnn_train.py    (trained models)

Usage
-----
    python scripts/06_cnn_predict_cesm2le.py

Author: Lauren Hoffman  <lhoffma2@ucsc.edu>
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr
from sklearn.metrics import precision_recall_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths
from src.cnn.splits import load_tvt_split
from src.cnn.train import load_model, predict_splits


# =============================================================================
# Configuration
# =============================================================================

N_SPLITS  = 9
N_SEEDS   = 5
BASE_SEED = 42

PREDICTIONS_DIR = paths.RESULTS_DIR / 'predictions' / 'cesm2le'


# =============================================================================
# Helpers
# =============================================================================

def compute_optimal_threshold(y_true, y_prob):
    """Threshold where precision ≈ recall (same rule used in the notebooks)."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    if thr.size == 0:
        return 0.5
    best_idx = np.argmin(np.abs(prec[:-1] - rec[:-1]))
    return float(thr[best_idx])


def save_prediction(
    out_path,
    y_scores,
    y_true,
    threshold,
    split_idx,
    run_idx,
    split_file,
):
    """Save one model's predictions to NetCDF."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = xr.Dataset(
        {
            # Predicted probabilities
            'y_prob_train': ('n_train', y_scores['train'].astype(np.float32)),
            'y_prob_val':   ('n_val',   y_scores['val'].astype(np.float32)),
            'y_prob_test':  ('n_test',  y_scores['test'].astype(np.float32)),

            # Binary predictions (at optimal threshold)
            'y_pred_train': ('n_train', (y_scores['train'] >= threshold).astype(np.int8)),
            'y_pred_val':   ('n_val',   (y_scores['val']   >= threshold).astype(np.int8)),
            'y_pred_test':  ('n_test',  (y_scores['test']  >= threshold).astype(np.int8)),

            # True labels
            'y_true_train': ('n_train', y_true['train'].astype(np.int8)),
            'y_true_val':   ('n_val',   y_true['val'].astype(np.int8)),
            'y_true_test':  ('n_test',  y_true['test'].astype(np.int8)),

            # Threshold
            'threshold': ((), np.float32(threshold)),
        },
        coords={
            'n_train': np.arange(len(y_true['train'])),
            'n_val':   np.arange(len(y_true['val'])),
            'n_test':  np.arange(len(y_true['test'])),
        },
        attrs={
            'split_idx': split_idx,
            'run_idx':   run_idx,
            'seed':      BASE_SEED + run_idx,
            'tvt_split_file': str(split_file),
            'description': (
                f'CNN predictions on CESM2-LE TVT data for split {split_idx}, '
                f'seed {BASE_SEED + run_idx}. '
                'Threshold is the PR-curve point where precision ≈ recall '
                '(computed on training data).'
            ),
        },
    )

    # Variable metadata
    ds['y_prob_train'].attrs['long_name'] = 'Predicted probability (train)'
    ds['y_prob_val'].attrs['long_name']   = 'Predicted probability (val)'
    ds['y_prob_test'].attrs['long_name']  = 'Predicted probability (test)'
    ds['y_pred_train'].attrs['long_name'] = 'Binary prediction at optimal threshold (train)'
    ds['y_pred_val'].attrs['long_name']   = 'Binary prediction at optimal threshold (val)'
    ds['y_pred_test'].attrs['long_name']  = 'Binary prediction at optimal threshold (test)'
    ds['y_true_train'].attrs['long_name'] = 'True slowdown label (train)'
    ds['y_true_val'].attrs['long_name']   = 'True slowdown label (val)'
    ds['y_true_test'].attrs['long_name']  = 'True slowdown label (test)'
    ds['threshold'].attrs['long_name']    = 'Optimal threshold (precision ≈ recall on train)'

    encoding = {v: {'zlib': True, 'complevel': 4}
                for v in ds.data_vars if ds[v].ndim > 0}
    ds.to_netcdf(out_path, format='NETCDF4', encoding=encoding)


# =============================================================================
# Main
# =============================================================================

def main():
    print()
    print('=' * 70)
    print('06  —  Precompute CNN predictions on CESM2-LE')
    print('=' * 70)
    print(f'  Data root      : {paths.DATA_ROOT}')
    print(f'  N splits       : {N_SPLITS}')
    print(f'  N seeds        : {N_SEEDS}  (seeds {BASE_SEED}–{BASE_SEED + N_SEEDS - 1})')
    print(f'  Models dir     : {paths.MODELS_DIR}')
    print(f'  Output dir     : {PREDICTIONS_DIR}')
    print('=' * 70)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    n_saved = 0

    for split_idx in range(N_SPLITS):
        print(f'\n{"─" * 60}')
        print(f'Split {split_idx}')
        print(f'{"─" * 60}')

        # ── Load TVT split ────────────────────────────────────────────────
        split_path = paths.tvt_split_path(split_idx)
        if not split_path.exists():
            raise FileNotFoundError(
                f'TVT split not found: {split_path}\n'
                f'Run scripts/03_cesm2le_tvt_splits.py first.'
            )
        split = load_tvt_split(split_path)

        x_tr = split['sst_tr'][:, :, :, np.newaxis]   # (n_tr, nx, ny, 1)
        x_va = split['sst_va'][:, :, :, np.newaxis]
        x_te = split['sst_te'][:, :, :, np.newaxis]

        y_true = {
            'train': split['slow_tr'],
            'val':   split['slow_va'],
            'test':  split['slow_te'],
        }

        print(f'  Train: {x_tr.shape}  prevalence={y_true["train"].mean():.3f}')
        print(f'  Val  : {x_va.shape}  prevalence={y_true["val"].mean():.3f}')
        print(f'  Test : {x_te.shape}  prevalence={y_true["test"].mean():.3f}')

        # ── Loop over seeds ───────────────────────────────────────────────
        for run_idx in range(N_SEEDS):
            model_p = paths.model_path(split_idx, run_idx)
            if not model_p.exists():
                print(f'  [skip] {model_p.name} not found')
                continue

            print(f'  seed {BASE_SEED + run_idx}: ', end='', flush=True)

            # Load model and predict
            model = load_model(paths.MODELS_DIR, split_idx, run_idx)
            y_scores = predict_splits(model, x_tr, x_va, x_te)
            print('predicted', end=' → ', flush=True)

            # Optimal threshold (from training data, same as notebooks)
            threshold = compute_optimal_threshold(y_true['train'], y_scores['train'])

            # Save
            out_path = PREDICTIONS_DIR / f'cnn_prediction_cesm2le_M{split_idx}_{run_idx}.nc'
            save_prediction(
                out_path, y_scores, y_true, threshold,
                split_idx, run_idx, split_path,
            )
            print(f'saved  (thr={threshold:.4f})')
            n_saved += 1

    print()
    print('=' * 70)
    print(f'Done.  Saved {n_saved} prediction files → {PREDICTIONS_DIR}')
    print('=' * 70 + '\n')


if __name__ == '__main__':
    main()
