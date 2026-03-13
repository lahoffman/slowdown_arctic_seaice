#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ERSSTv5 Data Download and Preprocessing Pipeline

This script downloads ERSSTv5 SST and 
re-grids to the CESM2-LE grid.

Usage
-----
    python scripts/01_ersst_preprocessing.py


Author: Lauren Hoffman
Email:  lhoffma2@ucscledu
"""

import sys
from pathlib import Path
import argparse

# ---------------------------------------------------------------------------
# Project root on sys.path so imports work regardless of working directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths

from src.data.observations.ersst.download import load_ersst
from src.data.observations.ersst.regrid_to_cesm2le import process_ersst_regrid

GRID_FILE   = paths.CESM2LE_SST_DIR / 'grid' / 'cesm2le_sst_grid.nc'

def main(cesm2le_grid_file: str | None = None):
    if cesm2le_grid_file is None:
        cesm2le_grid_file = str(GRID_FILE)

    ersst_ds = paths.ERSST_FILE
    print("\nDownloading ERSSTv5 & Regridding to CESM2-LE grid...")
    process_ersst_regrid(
        ersst_ds,
        paths.ERSST_REGRIDDED,
        cesm2le_grid_file=cesm2le_grid_file,
    )
    print("✓ ERSSTv5 regridding complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and preprocess ERSSTv5 data.")
    parser.add_argument(
        '--cesm2le-grid-file', type=str, default=None,
        help='Path to a CESM2-LE atmospheric (cam.h0) file to extract the target grid from.'
    )
    args = parser.parse_args()
    main(cesm2le_grid_file=args.cesm2le_grid_file)
