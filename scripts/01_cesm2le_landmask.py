#!/usr/bin/env python3
"""
01_cesm2le_landmask.py
======================
Build the CESM2-LE land mask from a raw SST file.

SST is stored as a large fill value (typically ~1e30) or a masked array over land,
and as a finite number over ocean.
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
    var = ds.variables['SST']

    # Read fill value from variable attributes (CESM2 raw files use ~1e30)
    fill_value = None
    for attr in ('_FillValue', 'missing_value'):
        if hasattr(var, attr):
            fill_value = float(getattr(var, attr))
            break
    print(f"SST fill value : {fill_value}")

    sst_raw = var[0, :, :]      # one time step; may be masked array or plain ndarray

# ---- Build mask --------------------------------------------------------
if isinstance(sst_raw, np.ma.MaskedArray) and np.any(sst_raw.mask):
    # netCDF4 auto-masked using _FillValue → mask is True where land
    landmask = np.asarray(sst_raw.mask).astype(np.int8)
elif fill_value is not None:
    # Plain ndarray with a large sentinel fill value (e.g. 1e30)
    sst_arr = np.asarray(sst_raw, dtype=np.float64)
    landmask = (sst_arr >= 0.99 * fill_value).astype(np.int8)
else:
    # Last resort: treat NaN as land
    sst_arr = np.asarray(sst_raw, dtype=np.float64)
    landmask = np.isnan(sst_arr).astype(np.int8)

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
