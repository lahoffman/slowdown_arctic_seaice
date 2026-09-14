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
    DATA_ROOT/results/models[/<tag>]/cnn_jja_split{k}_run{r}.h5    (one per split×seed)
    DATA_ROOT/results/metrics[/<tag>]/cnn_jja_metrics_split{k}.nc  (one per split)

Dependencies
------------
Requires outputs of scripts/03_cesm2le_tvt_splits.py (same ``--tag``).
If the split files carry auxiliary scalars (03 ``--aux sie_anom``) the CNN is
built with a matching second input automatically (``--no-aux`` to ignore them).

Usage
-----
    python scripts/04_cesm2le_cnn_train.py                    # original configuration
    python scripts/04_cesm2le_cnn_train.py --tag rel_base     # revision configurations
    python scripts/04_cesm2le_cnn_train.py --tag rel_aux --splits 0 1 --n-runs 2   # quick check

Author: Lauren Hoffman  <lhoffma2@ucsc.edu>
"""

import argparse
import json
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
    load_model,
    save_metrics_dataset,
    model_inputs,
    n_aux_inputs,
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



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tag', default=None,
                        help='Configuration tag (splits read from tvt_splits/<tag>/, '
                             'models/metrics written to models/<tag>/, metrics/<tag>/).')
    parser.add_argument('--splits', type=int, nargs='+', default=list(range(N_SPLITS)),
                        help='Subset of split indices to train (default: all 9).')
    parser.add_argument('--n-runs', type=int, default=N_RUNS,
                        help=f'Seeds per split (default {N_RUNS}).')
    parser.add_argument('--no-aux', action='store_true',
                        help='Ignore auxiliary scalars even if the split files have them.')
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['num_epochs'],
                        help=f"Max epochs (default {TRAIN_CONFIG['num_epochs']}; use 2 for a smoke test).")
    parser.add_argument('--no-warm-start', action='store_true',
                        help='Do NOT initialise the output layer at the logistic fit on the scalar inputs '
                             '(default: warm start whenever the split has aux scalars; step 8.8).')
    parser.add_argument('--stopping', choices=['loss', 'auprc'], default='loss',
                        help="Early-stopping rule: 'loss' = val_loss, patience 10 (original, tags rel_*); "
                             "'auprc' = val AUPRC, patience 15, no stop before epoch 5 (Phase 8 tags).")
    parser.add_argument('--skip-existing', action='store_true',
                        help='Reuse a saved model instead of retraining it (resume after a crash).')
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    tag, n_runs = args.tag, args.n_runs
    models_dir, metrics_dir = paths.models_dir(tag), paths.metrics_dir(tag)
    logs_dir = paths.LOGS_DIR / tag if tag else paths.LOGS_DIR
    use_aux = False if args.no_aux else None      # None = use if present
    train_config = {**TRAIN_CONFIG, 'num_epochs': args.epochs}
    if args.stopping == 'auprc':
        train_config.update(monitor='val_auprc', patience=15, min_epochs=5)

    print()
    print('=' * 70)
    print('04  —  CESM2-LE CNN Training')
    print('=' * 70)
    print(f'  Data root  : {paths.DATA_ROOT}')
    print(f'  Tag        : {tag or "(none — original configuration)"}')
    print(f'  Splits     : {args.splits}')
    print(f'  N runs     : {n_runs}  (seeds {BASE_SEED}–{BASE_SEED + n_runs - 1})')
    print(f'  Models dir : {models_dir}')
    print(f'  Metrics dir: {metrics_dir}')
    print(f'  Stopping   : {train_config.get("monitor", "val_loss")}, patience {train_config["patience"]}, '
          f'min epochs {train_config.get("min_epochs", 0)}')
    print(f'  Epochs     : {args.epochs}' + ('  (SMOKE TEST)' if args.epochs < TRAIN_CONFIG['num_epochs'] else ''))
    print('=' * 70)

    metrics_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    for split_idx in args.splits:
        print(f'\n{"─" * 70}')
        print(f'Split {split_idx}')
        print(f'{"─" * 70}')

        # ------------------------------------------------------------------
        # Load TVT split
        # ------------------------------------------------------------------
        split_path = paths.tvt_split_path(split_idx, tag)
        if not split_path.exists():
            raise FileNotFoundError(
                f"TVT split file not found:\n  {split_path}\n"
                f"Run scripts/03_cesm2le_tvt_splits.py"
                + (f" --tag {tag}" if tag else "") + " first."
            )
        split = load_tvt_split(split_path)

        # ------------------------------------------------------------------
        # Prepare model inputs  →  (n_samples, nx, ny, nch=1) [+ aux scalars]
        # ------------------------------------------------------------------
        x_tr = model_inputs(split, 'tr', use_aux)
        x_va = model_inputs(split, 'va', use_aux)
        x_te = model_inputs(split, 'te', use_aux)
        n_aux = n_aux_inputs(split) if isinstance(x_tr, list) else 0

        y_tr = split['slow_tr']                        # (ntr,)
        y_va = split['slow_va']                        # (nva,)
        y_te = split['slow_te']                        # (nte,)

        maps_tr = x_tr[0] if n_aux else x_tr
        nx, ny = maps_tr.shape[1], maps_tr.shape[2]
        nch    = maps_tr.shape[3]

        print(f'  Train : {maps_tr.shape}  —  {y_tr.mean():.3f} prevalence')
        print(f'  Val   : {(x_va[0] if n_aux else x_va).shape}  —  {y_va.mean():.3f} prevalence')
        print(f'  Test  : {(x_te[0] if n_aux else x_te).shape}  —  {y_te.mean():.3f} prevalence')
        print(f'  Aux   : {n_aux} scalar input(s)'
              + (f" {split.get('aux_names')}" if n_aux else ''))
        for key in ('labels_file', 'demean', 'sst_lag'):
            if key in split['attrs']:
                print(f'  {key:6s}: {split["attrs"][key]}')

        y_true = {'train': y_tr, 'val': y_va, 'test': y_te}

        # ------------------------------------------------------------------
        # Loop over seeds
        # ------------------------------------------------------------------
        y_scores_runs = []

        for run_idx in range(n_runs):
            seed = BASE_SEED + run_idx
            print(f'\n  Run {run_idx}  (seed={seed})')

            # Reproducibility
            set_seed(seed)

            if args.skip_existing and paths.model_path(split_idx, run_idx, tag).exists():
                model = load_model(models_dir, split_idx, run_idx)
                print(f'    reusing saved model {paths.model_path(split_idx, run_idx, tag).name}')
            else:
                # Class weights
                cw = compute_class_weights(y_tr, fract_weight=FRACT_WEIGHT)
                print(f'    Class weights: {cw}')

                # Build and train model (warm start at the scalar-only logistic fit, step 8.8)
                aux_init = None
                if n_aux and not args.no_warm_start:
                    from sklearn.linear_model import LogisticRegression
                    lr_fit = LogisticRegression(class_weight=cw).fit(x_tr[1], y_tr)
                    aux_init = (lr_fit.coef_.ravel(), float(lr_fit.intercept_[0]))
                    if run_idx == 0:
                        print(f'    warm start: aux coef {np.round(aux_init[0], 3)}, intercept {aux_init[1]:.3f}')
                model = build_cnn(nx, ny, nch, rl2=RL2, drop=DROP, n_aux=n_aux, aux_init=aux_init)
                model, history = train_model(
                    model, x_tr, y_tr, x_va, y_va,
                    config=train_config,
                    class_weights=cw,
                )
                n_epochs = len(history.history['loss'])
                val_loss  = history.history['val_loss'][-1]
                print(f'    Stopped at epoch {n_epochs},  val_loss = {val_loss:.4f}', flush=True)

                # Save model + training history (for the learning-curve figure)
                save_model(model, models_dir, split_idx, run_idx)
                hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
                (logs_dir / f'history_split{split_idx}_run{run_idx}.json').write_text(json.dumps(hist))

            # Predict on all splits
            y_scores_runs.append(predict_splits(model, x_tr, x_va, x_te))

        # ------------------------------------------------------------------
        # Evaluate metrics across all runs for this split
        # ------------------------------------------------------------------
        print(f'\n  Collecting metrics across {n_runs} runs ...')
        ds_metrics = collect_metrics_dataset(y_true, y_scores_runs)

        # Quick summary: mean AUPRC across runs on test split
        auprc_test = ds_metrics['metric_value'] \
            .sel(metric='AUPRC', split='test').values
        print(f'  AUPRC (test) — '
              f'mean={auprc_test.mean():.3f},  '
              f'min={auprc_test.min():.3f},  '
              f'max={auprc_test.max():.3f}')

        ds_metrics.attrs.update({k: str(v) for k, v in split['attrs'].items()
                                 if k in ('labels_file', 'demean', 'sst_lag', 'aux', 'tag')})
        save_metrics_dataset(ds_metrics, metrics_dir, split_idx)

    print()
    print('=' * 70)
    print('Done.')
    print(f'  Models   → {models_dir}')
    print(f'  Metrics  → {metrics_dir}')
    print('=' * 70 + '\n')


if __name__ == '__main__':
    main()
