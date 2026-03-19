#!/usr/bin/env python3
"""
05_cesm2le_lrp.py
=================
Compute LRP-z (Layer-wise Relevance Propagation) attribution maps for each
trained CNN model.

For each split × seed combination this script:
1. Loads the TVT split to retrieve the training-set SST (the only split used
   for attributions, following the convention in the original M2 script).
2. Loads the corresponding trained model from DATA_ROOT/results/models/.
3. Strips the sigmoid activation from the output layer (required for LRP).
4. Runs LRP-z in chunks over all training samples.
5. Saves the attribution maps to DATA_ROOT/results/attributions/.

Land pixels (fill value = -10 in the standardised arrays) are replaced with 0
inside compute_lrp_z before analysis, so they receive no spurious relevance.

Outputs
-------
    DATA_ROOT/results/attributions/lrp_jja_split{k}_run{r}.nc

    Each file contains:
        lrp_attributions — shape (n_chunks, chunk_size, nx, ny, 1)
        lat, lon         — CESM2-LE grid coordinates

Dependencies
------------
Requires outputs of:
    scripts/03_cesm2le_tvt_splits.py
    scripts/04_cesm2le_cnn_train.py

Usage
-----
    python scripts/05_cesm2le_lrp.py

Note
----
iNNvestigate requires TF1-style graph mode.  The import of src.xai.lrp
disables eager execution automatically (tf.compat.v1.disable_eager_execution).
Run this script in a separate Python process from scripts that use TF eager
execution (e.g. 04_cesm2le_cnn_train.py).

Author: Lauren Hoffman  <lhoffma2@ucsc.edu>
"""

import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths
from src.cnn.splits import load_tvt_split
from src.cnn.train import load_model
from src.xai.lrp import strip_sigmoid, compute_lrp_z, save_lrp


# =============================================================================
# Configuration
# =============================================================================

N_SPLITS   = 9
N_RUNS     = 5

# Number of samples per LRP chunk.  Reduce if you run out of memory.
CHUNK_SIZE = 100

# Land-fill sentinel value used in the TVT split files
LAND_FILL_VALUE = -10.0


# =============================================================================
# Helpers
# =============================================================================

def load_lat_lon() -> tuple:
    """
    Load CESM2-LE latitude and longitude arrays from the grid file.

    Returns
    -------
    lat : np.ndarray  shape (nx,)
    lon : np.ndarray  shape (ny,)
    """
    if not paths.CESM2LE_GRID_FILE.exists():
        raise FileNotFoundError(
            f"CESM2-LE grid file not found:\n  {paths.CESM2LE_GRID_FILE}\n"
            "Update CESM2LE_GRID_FILE in configs/paths.py to point at any "
            "raw CESM2-LE atmosphere file that contains 'lat' and 'lon'."
        )
    with nc.Dataset(paths.CESM2LE_GRID_FILE, 'r') as ds:
        lat = np.array(ds.variables['lat'][:])
        lon = np.array(ds.variables['lon'][:])
    return lat, lon


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print()
    print('=' * 70)
    print('05  —  CESM2-LE LRP-z Attributions')
    print('=' * 70)
    print(f'  Data root       : {paths.DATA_ROOT}')
    print(f'  N splits        : {N_SPLITS}')
    print(f'  N runs          : {N_RUNS}')
    print(f'  Chunk size      : {CHUNK_SIZE}')
    print(f'  Attributions dir: {paths.ATTRIBUTIONS_DIR}')
    print('=' * 70)

    # ------------------------------------------------------------------
    # Load grid coordinates (shared across all splits / runs)
    # ------------------------------------------------------------------
    print('\nLoading lat / lon from CESM2-LE grid file ...')
    lat, lon = load_lat_lon()
    print(f'  lat shape: {lat.shape},  lon shape: {lon.shape}')

    # ------------------------------------------------------------------
    # Outer loop: splits
    # ------------------------------------------------------------------
    for split_idx in range(N_SPLITS):
        print(f'\n{"─" * 70}')
        print(f'Split {split_idx}')
        print(f'{"─" * 70}')

        # Load the training SST from the saved TVT split
        split_path = paths.tvt_split_path(split_idx)
        if not split_path.exists():
            raise FileNotFoundError(
                f"TVT split file not found:\n  {split_path}\n"
                f"Run scripts/03_cesm2le_tvt_splits.py first."
            )
        split = load_tvt_split(split_path)

        # Training inputs only — LRP is computed on training data
        x_tr = split['sst_tr'][:, :, :, np.newaxis]   # (ntr, nx, ny, 1)
        print(f'  Training input shape: {x_tr.shape}')

        n_samples = x_tr.shape[0]
        n_chunks  = n_samples // CHUNK_SIZE
        if n_samples % CHUNK_SIZE != 0:
            print(f'  Warning: {n_samples} samples not evenly divisible by '
                  f'chunk_size={CHUNK_SIZE}.  '
                  f'Last {n_samples % CHUNK_SIZE} sample(s) will be dropped.')

        # Inner loop: seeds / runs
        for run_idx in range(N_RUNS):
            print(f'\n  Run {run_idx}')

            # Check model exists
            model_path = paths.model_path(split_idx, run_idx)
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found:\n  {model_path}\n"
                    f"Run scripts/04_cesm2le_cnn_train.py first."
                )

            # Load trained model
            model = load_model(paths.MODELS_DIR, split_idx, run_idx)
            print(f'    Loaded model: {model_path.name}')

            # Strip sigmoid for LRP
            model_logits = strip_sigmoid(model)

            # Compute LRP-z in chunks
            print(f'    Running LRP-z over {n_chunks} chunks of {CHUNK_SIZE} ...')
            attributions = compute_lrp_z(
                model_logits=model_logits,
                x_data=x_tr,
                chunk_size=CHUNK_SIZE,
                lrp_method='lrp.z',
                land_fill_value=LAND_FILL_VALUE,
            )
            print(f'    Attribution shape: {attributions.shape}')

            # Save
            out_path = paths.attribution_path(split_idx, run_idx)
            save_lrp(
                attributions=attributions,
                lat=lat,
                lon=lon,
                savepath=out_path,
                split_idx=split_idx,
                run_idx=run_idx,
            )

    print()
    print('=' * 70)
    print('Done.')
    print(f'  Attributions → {paths.ATTRIBUTIONS_DIR}')
    print('=' * 70 + '\n')


if __name__ == '__main__':
    main()
