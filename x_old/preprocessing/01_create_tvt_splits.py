#!/usr/bin/env python3
"""
Step 1: Create Train-Validate-Test Splits

Creates 9 different TVT splits for robustness testing.
Based on D2_TVT_CESM2_sst_sie_seasonal_robustness_global.py
"""

import sys
from pathlib import Path
import numpy as np
import netCDF4 as nc

# Add paths
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))
pipeline_root = Path(__file__).parent.parent
sys.path.insert(0, str(pipeline_root))

from src import config as main_config
from src.utils import load_netcdf, save_netcdf, load_landmask
from src.ml import create_tvt_splits
import config


def load_sst_seasonal_data():
    """
    Load and process seasonal SST data from CESM2-LE.

    Returns
    -------
    ndarray
        SST data with shape (n_seasons, n_ensemble, n_years, lat, lon)
    """
    print("Loading SST data from CESM2-LE...")

    months = ['JUN', 'JUL', 'AUG', 'MAR', 'APR', 'MAY', 'DEC', 'JAN', 'FEB', 'SEP', 'OCT', 'NOV']
    mon_index = np.reshape(np.arange(12), (4, 3))  # 4 seasons × 3 months

    # Year range
    years = np.arange(config.YEAR_START, config.YEAR_END + 1)
    y1990_idx = int(np.where(years == 1990)[0][0])
    y2041_idx = int(np.where(years == 2041)[0][0])

    sst_season = []

    for season_idx in range(4):
        print(f"  Processing season {season_idx + 1}/4...")
        sst_season_months = []

        for month_idx in range(3):
            month = months[mon_index[season_idx, month_idx]]

            # Load first and last 50 members
            path1 = config.INPUT_DATA['sst_monthly']['first50'].format(month=month)
            path2 = config.INPUT_DATA['sst_monthly']['last50'].format(month=month)

            with nc.Dataset(path1, 'r') as ds1, nc.Dataset(path2, 'r') as ds2:
                sst1 = ds1.variables['sst_mon'][:, :, :, :]
                sst2 = ds2.variables['sst_mon'][:, :, :, :]

            # Concatenate and subset years
            sst_combined = np.concatenate((sst1[:, :y2041_idx, ...], sst2[:, :y2041_idx, ...]), axis=0)

            # Adjust for seasonal timing
            if (season_idx == 2) and (month_idx == 0):  # DEC
                sst_month = sst_combined[:, y1990_idx:-1, :, :]
            elif (season_idx == 2) and (month_idx in (1, 2)):  # JAN, FEB
                sst_month = sst_combined[:, y1990_idx+1:, :, :]
            elif season_idx == 3:  # SON
                sst_month = sst_combined[:, y1990_idx:-1, :, :]
            else:  # JJA, MAM
                sst_month = sst_combined[:, y1990_idx+1:, :, :]

            sst_season_months.append(sst_month)

        # Average over months in season
        sst_season_avg = np.nanmean(sst_season_months, axis=0)
        sst_season.append(sst_season_avg)

    sst_all = np.array(sst_season)
    print(f"  ✓ Loaded SST: {sst_all.shape}")

    # Remove ensemble mean
    sst_ensemble_mean = np.nanmean(sst_all, axis=1, keepdims=True)
    sst_demeaned = sst_all - sst_ensemble_mean

    return sst_demeaned


def create_and_save_splits(sst_data, sie_slowdown):
    """
    Create TVT splits and save to files.

    Parameters
    ----------
    sst_data : ndarray
        SST data (n_seasons, n_ensemble, n_years, lat, lon)
    sie_slowdown : ndarray
        SIE slowdown labels
    """
    print("\nCreating TVT splits...")

    # Reshape for block-based splitting
    n_seasons, n_ensemble, n_years, lat, lon = sst_data.shape
    sst_reshaped = sst_data.reshape(n_seasons, config.N_BLOCKS, config.BLOCK_SIZE,
                                     n_years, lat, lon)

    for split_idx in range(config.N_SPLITS):
        print(f"\nProcessing split {split_idx + 1}/{config.N_SPLITS}...")

        split_config = config.get_split_config(split_idx)

        # Extract blocks
        train_blocks = split_config['train']
        val_block = split_config['validate'][0]
        test_block = split_config['test'][0]

        # Training data
        sst_train = sst_reshaped[:, train_blocks, :, :, :, :]
        sst_train = sst_train.reshape(n_seasons, 80, n_years, lat, lon)
        sst_train = sst_train.reshape(n_seasons, 80*n_years, lat, lon)

        # Validation data
        sst_val = sst_reshaped[:, val_block, :, :, :, :]
        sst_val = sst_val.reshape(n_seasons, config.BLOCK_SIZE, n_years, lat, lon)
        sst_val = sst_val.reshape(n_seasons, config.BLOCK_SIZE*n_years, lat, lon)

        # Test data
        sst_test = sst_reshaped[:, test_block, :, :, :, :]
        sst_test = sst_test.reshape(n_seasons, config.BLOCK_SIZE, n_years, lat, lon)
        sst_test = sst_test.reshape(n_seasons, config.BLOCK_SIZE*n_years, lat, lon)

        # Save split
        savepath = config.get_tvt_data_path(split_idx, variant='seasonal')

        data_dict = {
            "sst_train": (("season", "nt_train", "lat", "lon"), sst_train),
            "sst_val": (("season", "nt_val", "lat", "lon"), sst_val),
            "sst_test": (("season", "nt_test", "lat", "lon"), sst_test),
        }

        coords_dict = {
            "season": np.arange(n_seasons),
            "nt_train": np.arange(sst_train.shape[1]),
            "nt_val": np.arange(sst_val.shape[1]),
            "nt_test": np.arange(sst_test.shape[1]),
            "lat": np.arange(lat),
            "lon": np.arange(lon)
        }

        save_netcdf(data_dict, coords_dict, savepath)
        print(f"  ✓ Saved split {split_idx} to {savepath.name}")


def main():
    """Main preprocessing workflow."""

    print("=" * 70)
    print("Step 1: Creating Train-Validate-Test Splits")
    print("=" * 70)

    # Load SST data
    sst_data = load_sst_seasonal_data()

    # Load SIE slowdown labels (placeholder - implement as needed)
    sie_slowdown = None  # Load from appropriate file

    # Create and save splits
    create_and_save_splits(sst_data, sie_slowdown)

    print("\n" + "=" * 70)
    print("✓ Preprocessing complete!")
    print(f"Created {config.N_SPLITS} TVT splits")
    print("=" * 70)


if __name__ == "__main__":
    main()
