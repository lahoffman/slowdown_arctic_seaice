#!/usr/bin/env python3
"""
Step 2: Preprocess Data

Preprocess raw data and create train-validate-test splits.
"""

import sys
from pathlib import Path
import numpy as np
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths, model, training
from slowdown.data import (
    load_netcdf,
    save_netcdf,
    load_landmask,
    regrid_to_cesm2,
    compute_nino34_index
)
from slowdown.models import create_tvt_splits


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess data and create TVT splits')
    parser.add_argument('--n-splits', type=int, default=training.SPLIT_CONFIG['n_splits'],
                        help='Number of TVT splits to create')
    return parser.parse_args()


def load_cesm2_sst_seasonal():
    """Load and process seasonal SST data from CESM2-LE."""

    print("Loading CESM2-LE SST data...")

    # TODO: Implement loading from your CESM2-LE files
    # This is where you'd load from:
    # paths.CESM2_SST_MONTHLY['first50'], paths.CESM2_SST_MONTHLY['last50']

    # Placeholder - replace with actual loading
    print("  [Implement CESM2-LE SST loading here]")

    # Return shape: (n_seasons, n_ensemble, n_years, lat, lon)
    return None


def create_splits(data, n_splits):
    """Create train-validate-test splits."""

    print(f"\nCreating {n_splits} TVT splits...")

    # Reshape for block-based splitting
    n_seasons, n_ensemble, n_years, lat, lon = data.shape
    data_reshaped = data.reshape(n_seasons, model.ENSEMBLE_CONFIG['n_blocks'],
                                   model.ENSEMBLE_CONFIG['block_size'],
                                   n_years, lat, lon)

    for split_idx in range(n_splits):
        print(f"  Creating split {split_idx + 1}/{n_splits}...")

        # Determine train/val/test blocks
        test_block = split_idx
        val_block = (split_idx + 1) % model.ENSEMBLE_CONFIG['n_blocks']
        train_blocks = [i for i in range(model.ENSEMBLE_CONFIG['n_blocks'])
                        if i not in [test_block, val_block]]

        # Extract data
        train_data = data_reshaped[:, train_blocks, :, :, :, :]
        train_data = train_data.reshape(n_seasons, -1, lat, lon)

        val_data = data_reshaped[:, val_block, :, :, :, :]
        val_data = val_data.reshape(n_seasons, -1, lat, lon)

        test_data = data_reshaped[:, test_block, :, :, :, :]
        test_data = test_data.reshape(n_seasons, -1, lat, lon)

        # Save split
        output_file = paths.get_data_path(f'tvt_split_{split_idx}.nc')

        data_dict = {
            "sst_train": (("season", "nt_train", "lat", "lon"), train_data),
            "sst_val": (("season", "nt_val", "lat", "lon"), val_data),
            "sst_test": (("season", "nt_test", "lat", "lon"), test_data),
        }

        coords_dict = {
            "season": np.arange(n_seasons),
            "nt_train": np.arange(train_data.shape[1]),
            "nt_val": np.arange(val_data.shape[1]),
            "nt_test": np.arange(test_data.shape[1]),
            "lat": np.arange(lat),
            "lon": np.arange(lon)
        }

        save_netcdf(data_dict, coords_dict, output_file)
        print(f"    ✓ Saved: {output_file.name}")


def main():
    """Main preprocessing workflow."""

    args = parse_args()

    print("=" * 70)
    print("Step 2: Preprocessing Data")
    print("=" * 70)

    # Load data
    sst_data = load_cesm2_sst_seasonal()

    if sst_data is not None:
        # Create splits
        create_splits(sst_data, args.n_splits)

        print("\n" + "=" * 70)
        print("✓ Preprocessing complete!")
        print(f"Created {args.n_splits} TVT splits in {paths.DATA_DIR}")
        print("=" * 70)
    else:
        print("\n⚠ Skipping split creation - implement data loading first")


if __name__ == "__main__":
    main()
