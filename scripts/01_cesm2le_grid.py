#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CESM2-LE SST Grid Extraction

Extract latitude/longitude from an SST file and save them to a separate
NetCDF grid file.

Usage
-----
    python scripts/01_cesm2le_grid.py b.e21.BHISTcmip6.f09_g17.LE2-1001.001.cam.h0.SST.199001-199912.nc -o cesm2le_sst_grid.nc

Author: Lauren Hoffman
Email: lhoffma2@ucsc.edu
"""

import argparse
import numpy as np
import xarray as xr
from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Project root on sys.path so imports work regardless of working directory
# ---------------------------------------------------------------------------
DATA_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DATA_ROOT))

from configs import paths


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _sst_raw_dir() -> Path:
    """Raw SST chunks download directory."""
    return paths.CESM2LE_SST_DIR / "raw"


def _sst_grid_dir() -> Path:
    """Directory for saved SST grid files."""
    return paths.CESM2LE_SST_DIR / "grid"


def _resolve_sst_file(sst_file: str) -> Path:
    """
    Resolve SST input file.

    If sst_file is an existing path, use it directly.
    Otherwise, look for it inside the raw SST directory.
    """
    candidate = Path(sst_file).expanduser()

    if candidate.exists():
        return candidate.resolve()

    raw_candidate = _sst_raw_dir() / sst_file
    if raw_candidate.exists():
        return raw_candidate.resolve()

    raise FileNotFoundError(
        f"Could not find SST file:\n"
        f"  - as provided: {candidate}\n"
        f"  - in raw SST dir: {raw_candidate}"
    )


def _resolve_output_file(output_file: str) -> Path:
    """
    Resolve output file path.

    If output_file includes a directory, use it directly.
    Otherwise, save it inside the SST grid directory.
    """
    output_path = Path(output_file).expanduser()

    if output_path.parent == Path("."):
        output_path = _sst_grid_dir() / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path.resolve()


def save_sst_grid(
    sst_file: str,
    output_file: str = "cesm2le_sst_grid.nc"
) -> None:
    """
    Load an SST file, extract latitude/longitude, and save them to a
    separate NetCDF file.

    Parameters
    ----------
    sst_file : str
        Path to SST NetCDF file, or filename within the raw SST directory
    output_file : str, optional
        Output NetCDF filename or path (default: 'cesm2le_sst_grid.nc')

    Returns
    -------
    None
        Saves lat/lon arrays to output_file

    Notes
    -----
    - If the SST file uses POP-style 2D coordinates, this saves TLAT/TLONG.
    - If the SST file uses 1D lat/lon coordinates, this converts them to 2D
      with np.meshgrid before saving.
    - Longitudes are converted to [0, 360) to match the rest of your script.
    """
    sst_path = _resolve_sst_file(sst_file)
    output_path = _resolve_output_file(output_file)

    print("Saving SST grid...")
    print(f"  Input: {sst_path}")
    print(f"  Output: {output_path}")

    ds_sst = xr.open_dataset(sst_path)

    try:
        # Get target grid (SST - POP ocean grid)
        if "TLAT" in ds_sst and "TLONG" in ds_sst:
            lat_sst = ds_sst["TLAT"].values
            lon_sst = ds_sst["TLONG"].values
            lat_name = "lat"
            lon_name = "lon"

        elif "lat" in ds_sst and "lon" in ds_sst:
            if ds_sst["lat"].ndim == 1:
                lon_sst, lat_sst = np.meshgrid(
                    ds_sst["lon"].values,
                    ds_sst["lat"].values
                )
                lat_name = "lat"
                lon_name = "lon"
            else:
                lat_sst = ds_sst["lat"].values
                lon_sst = ds_sst["lon"].values
                lat_name = "lat"
                lon_name = "lon"

        else:
            raise ValueError("Cannot find latitude/longitude coordinates in SST file")

        # Ensure longitude is in [0, 360)
        lon_sst = np.where(lon_sst < 0, lon_sst + 360, lon_sst)

        # Require 2D output to match POP-style usage elsewhere
        if lat_sst.ndim != 2 or lon_sst.ndim != 2:
            raise ValueError(
                f"Expected 2D lat/lon after processing, got "
                f"lat shape {lat_sst.shape}, lon shape {lon_sst.shape}"
            )

        ds_out = xr.Dataset(
            data_vars={
                lat_name: (("nj", "ni"), lat_sst),
                lon_name: (("nj", "ni"), lon_sst),
            }
        )

        # Add metadata
        ds_out[lat_name].attrs = {
            "long_name": "latitude",
            "units": "degrees_north",
        }
        ds_out[lon_name].attrs = {
            "long_name": "longitude",
            "units": "degrees_east",
        }

        ds_out.attrs["source_file"] = str(sst_path)
        ds_out.attrs["description"] = "Latitude/longitude grid extracted from CESM2-LE SST file"

        # Save
        encoding = {
            lat_name: {"zlib": True, "complevel": 4},
            lon_name: {"zlib": True, "complevel": 4},
        }
        ds_out.to_netcdf(output_path, encoding=encoding)

        print(f"  ✓ SST grid saved to: {output_path}")
        print(f"    lat shape: {lat_sst.shape}")
        print(f"    lon shape: {lon_sst.shape}")

    finally:
        ds_sst.close()


def main():
    parser = argparse.ArgumentParser(
        description="Extract lat/lon grid from a CESM2-LE SST file and save as NetCDF."
    )
    parser.add_argument(
        "sst_file",
        type=str,
        help=(
            "Input SST NetCDF file. Can be either a full path or a filename "
            f"located in {_sst_raw_dir()}."
        )
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="cesm2le_sst_grid.nc",
        help=(
            "Output grid NetCDF file. If only a filename is given, it will be "
            f"saved in {_sst_grid_dir()}."
        )
    )

    args = parser.parse_args()

    save_sst_grid(
        sst_file=args.sst_file,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
