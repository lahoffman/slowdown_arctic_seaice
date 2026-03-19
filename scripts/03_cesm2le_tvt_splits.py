#!/usr/bin/env python3
"""
03_cesm2le_tvt_splits.py
========================
Build and save the 9 train / validate / test data splits for the JJA SST CNN.

For each of the 9 splits this script:
1. Loads JJA (June–July–August) mean SST from the monthly files produced by
   scripts/01_cesm2le_preprocessing.py.
2. Loads September slowdown labels from the classification file produced by
   scripts/02_cesm2le_slowdowns.py.
3. Aligns SST and label years.
4. Splits the 100-member ensemble into train / validate / test blocks.
5. Standardises SST using global ocean-only statistics from the training data.
6. Applies the land mask (sets land pixels to -10).
7. Saves each split to a NetCDF file under DATA_ROOT/results/tvt_splits/.

Outputs (one per split, k = 0 … 8)
------------------------------------
    DATA_ROOT/results/tvt_splits/cesm2le_sst_jja_slowdown_split{k}.nc

    Each file contains:
        sst_tr, sst_va, sst_te    — standardised, land-masked JJA SST
        slow_tr, slow_va, slow_te — binary September slowdown labels (0/1)
        mu_train, sigma_train     — normalisation statistics for downstream XAI

Usage
-----
    python scripts/03_cesm2le_tvt_splits.py

Author: Lauren Hoffman  <lhoffma2@ucsc.edu>
"""

import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths
from src.cnn.splits import (
    load_jja_sst_demeaned,
    iter_splits,
    standardize,
    apply_landmask,
    save_tvt_split,
)


# =============================================================================
# Configuration
# =============================================================================

# Year range — JJA of year t predicts September SIE of year t.
START_YEAR = 1990
END_YEAR   = 2040

# Slowdown variable and target month
SLOWDOWN_VAR   = 'sie'
SLOWDOWN_MONTH = 'SEP'

# Ensemble structure
N_SPLITS    = 9
N_BLOCKS    = 10
BLOCK_SIZE  = 10

# Member groups (must match keys in paths.CESM2LE_SST_MONTHLY)
MEMBER_GROUPS = ['first50', 'last50']

# NetCDF variable name for SST inside the monthly files
SST_VARNAME = 'sst_mon'


# =============================================================================
# Helpers
# =============================================================================

def load_landmask() -> np.ndarray:
    """Load the CESM2-LE land mask.  0 = ocean, 1 = land."""
    with nc.Dataset(paths.LANDMASK_FILE, 'r') as ds:
        return np.array(ds.variables['landmask'][:])


def load_slowdown_labels(
    variable: str,
    month: str,
    start_year: int,
    end_year: int,
) -> np.ndarray:
    """
    Load binary September slowdown labels for the requested year range.

    Reads the file produced by scripts/02_cesm2le_slowdowns.py.  The ``nyr``
    coordinate in that file is the *starting year* of each 10-year trend
    window, so selecting years [start_year, end_year] gives the label for
    the September SIE trend beginning in each of those years.

    Returns
    -------
    slowdown : np.ndarray
        Binary array, shape ``(nens, nyear)``.  1 = slowdown, 0 = normal.
    """
    fpath = paths.cesm2le_slowdown_file(variable, month)
    if not fpath.exists():
        raise FileNotFoundError(
            f"Slowdown file not found:\n  {fpath}\n"
            f"Run scripts/02_cesm2le_slowdowns.py first."
        )

    with xr.open_dataset(fpath) as ds:
        slowdown = (
            ds['slowdown']
            .sel(nyr=slice(start_year, end_year))
            .values                                    # (nens, nyear)
        )

    print(f"  Slowdown labels: shape {slowdown.shape}, "
          f"years {start_year}–{end_year}, "
          f"prevalence {slowdown.mean():.3f}")
    return slowdown.astype(np.int8)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print()
    print('=' * 70)
    print('03  —  CESM2-LE TVT Splits')
    print('=' * 70)
    print(f'  Data root    : {paths.DATA_ROOT}')
    print(f'  SST years    : {START_YEAR}–{END_YEAR}')
    print(f'  Slowdown     : {SLOWDOWN_VAR.upper()} / {SLOWDOWN_MONTH}')
    print(f'  N splits     : {N_SPLITS}')
    print(f'  Output dir   : {paths.TVT_SPLITS_DIR}')
    print('=' * 70)

    # ------------------------------------------------------------------
    # 1.  Load JJA SST  (ensemble-demeaned)
    # ------------------------------------------------------------------
    print('\n[1] Loading JJA SST ...')
    sst, sst_years = load_jja_sst_demeaned(
        sst_monthly_template=paths.CESM2LE_SST_MONTHLY,
        member_groups=MEMBER_GROUPS,
        start_year=START_YEAR,
        end_year=END_YEAR,
        sst_varname=SST_VARNAME,
    )
    print(f'    SST shape : {sst.shape}  '
          f'(nens={sst.shape[0]}, nyear={sst.shape[1]}, '
          f'nx={sst.shape[2]}, ny={sst.shape[3]})')

    # ------------------------------------------------------------------
    # 2.  Load September slowdown labels
    # ------------------------------------------------------------------
    print(f'\n[2] Loading slowdown labels ({SLOWDOWN_VAR.upper()} / {SLOWDOWN_MONTH}) ...')
    slowdown = load_slowdown_labels(
        variable=SLOWDOWN_VAR,
        month=SLOWDOWN_MONTH,
        start_year=START_YEAR,
        end_year=END_YEAR,
    )

    # Sanity check — ensemble and year dimensions must match
    if sst.shape[0] != slowdown.shape[0]:
        raise ValueError(
            f"Ensemble size mismatch: SST has {sst.shape[0]} members, "
            f"slowdown has {slowdown.shape[0]}."
        )
    if sst.shape[1] != slowdown.shape[1]:
        raise ValueError(
            f"Year mismatch: SST has {sst.shape[1]} years, "
            f"slowdown has {slowdown.shape[1]}."
        )

    # ------------------------------------------------------------------
    # 3.  Load land mask
    # ------------------------------------------------------------------
    print('\n[3] Loading land mask ...')
    landmask = load_landmask()
    print(f'    Landmask shape : {landmask.shape}  '
          f'(ocean fraction: {(landmask == 0).mean():.3f})')

    # ------------------------------------------------------------------
    # 4.  Loop over splits
    # ------------------------------------------------------------------
    print(f'\n[4] Building {N_SPLITS} TVT splits ...\n')

    for split in iter_splits(sst, slowdown, N_SPLITS, N_BLOCKS, BLOCK_SIZE):
        k = split['split_idx']
        print(f'  Split {k}  '
              f'(test block={split["test_block"]}, '
              f'val block={split["val_block"]}, '
              f'train blocks={split["train_blocks"]})')

        # Standardise using training ocean-only statistics
        sst_tr_std, sst_va_std, sst_te_std, mu, sigma = standardize(
            split['sst_tr'], split['sst_va'], split['sst_te'], landmask
        )
        print(f'    μ_train = {mu:.4f},  σ_train = {sigma:.4f}')

        # Apply land mask (land → -10 in standardised space)
        sst_tr_m = apply_landmask(sst_tr_std, landmask)
        sst_va_m = apply_landmask(sst_va_std, landmask)
        sst_te_m = apply_landmask(sst_te_std, landmask)

        # Save
        save_tvt_split(
            sst_tr=sst_tr_m,
            sst_va=sst_va_m,
            sst_te=sst_te_m,
            slow_tr=split['slow_tr'],
            slow_va=split['slow_va'],
            slow_te=split['slow_te'],
            mu_train=mu,
            sigma_train=sigma,
            split_idx=k,
            savepath=paths.tvt_split_path(k),
            attrs={
                'sst_years':       f'{START_YEAR}-{END_YEAR}',
                'slowdown_var':    SLOWDOWN_VAR,
                'slowdown_month':  SLOWDOWN_MONTH,
                'member_groups':   str(MEMBER_GROUPS),
            },
        )

    print()
    print('=' * 70)
    print('Done.  Splits saved to:')
    print(f'  {paths.TVT_SPLITS_DIR}')
    print('=' * 70 + '\n')


if __name__ == '__main__':
    main()
