#!/usr/bin/env python3
"""
01_cesm2le_landmask.py
======================
Build the CESM2-LE land mask from a raw SST file.

SST is NaN (or a masked fill value) over land and a finite number over ocean.
The mask is:  0 = ocean,  1 = land.

Output
------
    DATA_ROOT/cesm2le/cnn_cesm2le_landmask.nc

Usage
-----
    python scripts/01_cesm2le_landmask.py

Author: Lauren Hoffman  <lhoffma2@ucsc.edu>
"""

import glob
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths


# =============================================================================
# Find a raw SST file
# =============================================================================

raw_sst_dir = paths.CESM2LE_SST_DIR / 'raw'
files = sorted(glob.glob(str(raw_sst_dir / '*.nc')))

if not files:
    raise FileNotFoundError(
        f"No NetCDF files found in:\n  {raw_sst_dir}\n"
        "Run scripts/01_cesm2le_preprocessing.py --variable sst first."
    )

fpath = files[0]
print(f"Using: {fpath}")


# =============================================================================
# Build land mask
# =============================================================================

with nc.Dataset(fpath, 'r') as ds:
    sst_raw = ds.variables['SST'][0, 0, :, :]          # one member, one time step

# Handle both masked arrays and plain NaN
sst = np.ma.filled(np.ma.array(sst_raw), fill_value=np.nan)
landmask = np.isnan(sst).astype(np.int8)               # 0 = ocean, 1 = land

print(f"Grid shape  : {landmask.shape}")
print(f"Ocean fraction : {(landmask == 0).mean():.3f}")
print(f"Land  fraction : {(landmask == 1).mean():.3f}")


# =============================================================================
# Save
# =============================================================================

ds_out = xr.Dataset(
    {"landmask": (("nx", "ny"), landmask)},
    attrs={"description": "CESM2-LE land mask.  0 = ocean, 1 = land.",
           "source": str(fpath)},
)

paths.LANDMASK_FILE.parent.mkdir(parents=True, exist_ok=True)
ds_out.to_netcdf(paths.LANDMASK_FILE, format="NETCDF4")
print(f"Saved → {paths.LANDMASK_FILE}")
