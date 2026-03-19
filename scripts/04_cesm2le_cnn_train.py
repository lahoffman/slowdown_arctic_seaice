#!/usr/bin/env python3
"""
04_cesm2le_cnn_train.py
=======================
Train the JJA SST CNN for each of the 9 TVT splits and N_RUNS random seeds.

For each split × seed combination this script:
1. Loads the pre-built TVT split from DATA_ROOT/results/tvt_splits/.
2. Adds a channel dimension to the SST arrays  →  (n_samples, nx, ny, 1).
3. Sets the random seed for reproducibility.
4. Computes balanced class weights (with an upward adjustment for slowdowns).
5. Builds and trains the CNN with early stopping on validation loss.
6. Saves the trained model to DATA_ROOT/results/models/.
7. After all seeds for a split are done, evaluates and saves a metrics Dataset
   containing per-run values and 2.5 / 97.5 percentile CIs across runs.

Outputs
-------
    DATA_ROOT/results/models/cnn_jja_split{k}_run{r}.h5       (one per split×seed)
    DATA_ROOT/results/metrics/cnn_jja_metrics_split{k}.nc     (one per split)

Dependencies
------------
Requires outputs of scripts/03_cesm2le_tvt_splits.py.

Usage
-----
    python scripts/04_cesm2le_cnn_train.py

Author: Lauren Hoffman  <lhoffma2@ucsc.edu>
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths
from src.cnn.splits import load_tvt_split
from src.cnn.model import build_cnn, METRIC_NAMES
from src.cnn.train import (
    set_seed,
    compute_class_weights,
    train_model,
    predict_splits,
    collect_metrics_dataset,
    save_model,
    save_metrics_dataset,
)


# =============================================================================
# Configuration
# =============================================================================

N_SPLITS = 9
N_RUNS   = 5                         # number of random seeds per split
BASE_SEED = 42                        # seeds will be BASE_SEED + run_idx

# CNN architecture
RL2  = 1e-5
DROP = 0.2

# Class weight adjustment for the minority (slowdown) class
FRACT_WEIGHT = 1.5

# Training hyperparameters
TRAIN_CONFIG = {
    'learning_rate': 1e-4,
    'num_epochs':    50,
    'batch_size':    120,
    'patience':      10,
    'focal_alpha':   0.75,
    'focal_gamma':   2.0,
}

METRICS_DIR = paths.RESULTS_DIR / 'metrics'


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print()
    print('=' * 70)
    print('04  —  CESM2-LE CNN Training')
    print('=' * 70)
    print(f'  Data root  : {paths.DATA_ROOT}')
    print(f'  N splits   : {N_SPLITS}')
    print(f'  N runs     : {N_RUNS}  (seeds {BASE_SEED}–{BASE_SEED + N_RUNS - 1})')
    print(f'  Models dir : {paths.MODELS_DIR}')
    print(f'  Metrics dir: {METRICS_DIR}')
    print('=' * 70)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    for split_idx in range(N_SPLITS):
        print(f'\n{"─" * 70}')
        print(f'Split {split_idx}')
        print(f'{"─" * 70}')

        # ------------------------------------------------------------------
        # Load TVT split
        # ------------------------------------------------------------------
        split_path = paths.tvt_split_path(split_idx)
        if not split_path.exists():
            raise FileNotFoundError(
                f"TVT split file not found:\n  {split_path}\n"
                f"Run scripts/03_cesm2le_tvt_splits.py first."
            )
        split = load_tvt_split(split_path)

        # ------------------------------------------------------------------
        # Prepare model inputs  →  (n_samples, nx, ny, nch=1)
        # ------------------------------------------------------------------
        x_tr = split['sst_tr'][:, :, :, np.newaxis]   # (ntr, nx, ny, 1)
        x_va = split['sst_va'][:, :, :, np.newaxis]   # (nva, nx, ny, 1)
        x_te = split['sst_te'][:, :, :, np.newaxis]   # (nte, nx, ny, 1)

        y_tr = split['slow_tr']                        # (ntr,)
        y_va = split['slow_va']                        # (nva,)
        y_te = split['slow_te']                        # (nte,)

        nx, ny = x_tr.shape[1], x_tr.shape[2]
        nch    = x_tr.shape[3]

        print(f'  Train : {x_tr.shape}  —  {y_tr.mean():.3f} prevalence')
        print(f'  Val   : {x_va.shape}  —  {y_va.mean():.3f} prevalence')
        print(f'  Test  : {x_te.shape}  —  {y_te.mean():.3f} prevalence')

        y_true = {'train': y_tr, 'val': y_va, 'test': y_te}

        # ------------------------------------------------------------------
        # Loop over seeds
        # ------------------------------------------------------------------
        y_scores_runs = []

        for run_idx in range(N_RUNS):
            seed = BASE_SEED + run_idx
            print(f'\n  Run {run_idx}  (seed={seed})')

            # Reproducibility
            set_seed(seed)

            # Class weights
            cw = compute_class_weights(y_tr, fract_weight=FRACT_WEIGHT)
            print(f'    Class weights: {cw}')

            # Build and train model
            model = build_cnn(nx, ny, nch, rl2=RL2, drop=DROP)
            model, history = train_model(
                model, x_tr, y_tr, x_va, y_va,
                config=TRAIN_CONFIG,
                class_weights=cw,
            )
            n_epochs = len(history.history['loss'])
            val_loss  = history.history['val_loss'][-1]
            print(f'    Stopped at epoch {n_epochs},  val_loss = {val_loss:.4f}')

            # Save model
            save_model(model, paths.MODELS_DIR, split_idx, run_idx)

            # Predict on all splits
            y_scores_runs.append(predict_splits(model, x_tr, x_va, x_te))

        # ------------------------------------------------------------------
        # Evaluate metrics across all runs for this split
        # ------------------------------------------------------------------
        print(f'\n  Collecting metrics across {N_RUNS} runs ...')
        ds_metrics = collect_metrics_dataset(y_true, y_scores_runs)

        # Quick summary: mean AUPRC across runs on test split
        auprc_test = ds_metrics['metric_value'] \
            .sel(metric='AUPRC', split='test').values
        print(f'  AUPRC (test) — '
              f'mean={auprc_test.mean():.3f},  '
              f'min={auprc_test.min():.3f},  '
              f'max={auprc_test.max():.3f}')

        save_metrics_dataset(ds_metrics, METRICS_DIR, split_idx)

    print()
    print('=' * 70)
    print('Done.')
    print(f'  Models   → {paths.MODELS_DIR}')
    print(f'  Metrics  → {METRICS_DIR}')
    print('=' * 70 + '\n')


if __name__ == '__main__':
    main()
