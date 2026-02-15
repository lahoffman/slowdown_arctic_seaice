#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CESM2-LE Regridding Functions

Functions to regrid AICE data to match SST grid.

Author: Lauren Hoffman
Email: lhoffma2@ucsc.edu
"""

import numpy as np
import xarray as xr
from pathlib import Path
from typing import Optional


def regrid_aice_to_sst(
    aice_file: str,
    sst_grid_file: str,
    output_file: str,
    aice_var: str = 'aice',
    method: str = 'linear'
) -> None:
    """
    Regrid AICE data to match SST grid resolution.

    This function interpolates AICE from the CICE grid to the POP ocean grid
    used by SST. Useful for comparing sea ice and ocean variables on the same grid.

    Parameters
    ----------
    aice_file : str
        Path to AICE NetCDF file (CICE grid)
    sst_grid_file : str
        Path to SST file or grid file (POP ocean grid) to match
    output_file : str
        Path for output regridded NetCDF file
    aice_var : str, optional
        Variable name in aice_file (default: 'aice')
    method : str, optional
        Interpolation method: 'linear' or 'nearest' (default: 'linear')

    Returns
    -------
    None
        Saves regridded data to output_file

    Examples
    --------
    >>> regrid_aice_to_sst(
    ...     aice_file='/path/to/aice_mon_JAN_1990-2100.nc',
    ...     sst_grid_file='/path/to/sst_mon_JAN_1990-2100.nc',
    ...     output_file='/path/to/aice_regridded_JAN_1990-2100.nc'
    ... )

    Notes
    -----
    - AICE is on the CICE grid (typically higher resolution sea ice model grid)
    - SST is on the POP ocean grid
    - This function interpolates AICE to the POP grid for direct comparison
    - For monthly separated files, apply this to each month separately
    """
    print(f"Regridding AICE to SST grid...")
    print(f"  Input: {aice_file}")
    print(f"  Target grid: {sst_grid_file}")

    # Load datasets
    ds_aice = xr.open_dataset(aice_file)
    ds_sst = xr.open_dataset(sst_grid_file)

    # Get AICE data and coordinates
    aice = ds_aice[aice_var].values

    # Get source grid (AICE - CICE grid)
    if 'TLAT' in ds_aice and 'TLONG' in ds_aice:
        lat_aice = ds_aice['TLAT'].values
        lon_aice = ds_aice['TLONG'].values
    elif 'lat' in ds_aice and 'lon' in ds_aice:
        # Handle 1D coordinates
        if ds_aice['lat'].ndim == 1:
            lon_aice, lat_aice = np.meshgrid(ds_aice['lon'].values, ds_aice['lat'].values)
        else:
            lat_aice = ds_aice['lat'].values
            lon_aice = ds_aice['lon'].values
    else:
        raise ValueError("Cannot find latitude/longitude coordinates in AICE file")

    # Get target grid (SST - POP ocean grid)
    if 'TLAT' in ds_sst and 'TLONG' in ds_sst:
        lat_sst = ds_sst['TLAT'].values
        lon_sst = ds_sst['TLONG'].values
    elif 'lat' in ds_sst and 'lon' in ds_sst:
        if ds_sst['lat'].ndim == 1:
            lon_sst, lat_sst = np.meshgrid(ds_sst['lon'].values, ds_sst['lat'].values)
        else:
            lat_sst = ds_sst['lat'].values
            lon_sst = ds_sst['lon'].values
    else:
        raise ValueError("Cannot find latitude/longitude coordinates in SST file")

    # Ensure longitude is in [0, 360) range
    lon_aice = np.where(lon_aice < 0, lon_aice + 360, lon_aice)
    lon_sst = np.where(lon_sst < 0, lon_sst + 360, lon_sst)

    # Get dimensions
    original_shape = aice.shape
    time_dims = original_shape[:-2]  # All dimensions except spatial (nj, ni)
    n_time = np.prod(time_dims) if time_dims else 1

    # Reshape to (time, nj, ni) if needed
    if len(original_shape) > 3:
        aice_reshaped = aice.reshape(n_time, original_shape[-2], original_shape[-1])
    elif len(original_shape) == 3:
        aice_reshaped = aice
    else:
        aice_reshaped = aice[np.newaxis, :, :]
        n_time = 1

    # Initialize output array
    target_shape = lat_sst.shape
    aice_regridded = np.zeros((n_time, target_shape[0], target_shape[1]))

    # Regrid each time step
    print(f"  Regridding {n_time} time steps...")
    for t in range(n_time):
        if (t + 1) % 10 == 0 or t == 0:
            print(f"    Processing time step {t+1}/{n_time}")

        aice_regridded[t] = _regrid_2d(
            data=aice_reshaped[t],
            lat_src=lat_aice,
            lon_src=lon_aice,
            lat_tgt=lat_sst,
            lon_tgt=lon_sst,
            method=method
        )

    # Reshape back to original time dimensions
    if len(original_shape) > 3:
        aice_regridded = aice_regridded.reshape(*time_dims, target_shape[0], target_shape[1])
    elif len(original_shape) == 2:
        aice_regridded = aice_regridded[0]

    # Create output dataset
    ds_out = xr.Dataset(
        {
            f'{aice_var}_regridded': (ds_aice[aice_var].dims[:-2] + ('nj', 'ni'), aice_regridded),
        },
        coords={
            dim: ds_aice[dim] for dim in ds_aice[aice_var].dims[:-2]
        }
    )

    # Add grid coordinates
    ds_out['TLAT'] = (('nj', 'ni'), lat_sst)
    ds_out['TLONG'] = (('nj', 'ni'), lon_sst)

    # Add attributes
    ds_out[f'{aice_var}_regridded'].attrs = ds_aice[aice_var].attrs.copy()
    ds_out[f'{aice_var}_regridded'].attrs['regridded'] = 'True'
    ds_out[f'{aice_var}_regridded'].attrs['regrid_method'] = method
    ds_out[f'{aice_var}_regridded'].attrs['original_grid'] = 'CICE'
    ds_out[f'{aice_var}_regridded'].attrs['target_grid'] = 'POP ocean grid'

    # Save to file
    encoding = {var: {"zlib": True, "complevel": 4} for var in ds_out.data_vars}
    ds_out.to_netcdf(output_file, encoding=encoding)

    print(f"  ✓ Regridded data saved to: {output_file}")
    print(f"    Original shape: {original_shape}")
    print(f"    Regridded shape: {aice_regridded.shape}")

    # Close datasets
    ds_aice.close()
    ds_sst.close()


def _regrid_2d(
    data: np.ndarray,
    lat_src: np.ndarray,
    lon_src: np.ndarray,
    lat_tgt: np.ndarray,
    lon_tgt: np.ndarray,
    method: str = 'linear'
) -> np.ndarray:
    """
    Regrid a single 2D field using interpolation.

    This is a helper function that handles the actual interpolation.
    Uses nearest-neighbor lookup for irregular grids.
    """
    from scipy.spatial import cKDTree

    # Flatten source coordinates
    points_src = np.column_stack([lat_src.ravel(), lon_src.ravel()])
    values_src = data.ravel()

    # Flatten target coordinates
    points_tgt = np.column_stack([lat_tgt.ravel(), lon_tgt.ravel()])

    # Handle NaN values in source data
    valid_mask = ~np.isnan(values_src)
    points_src_valid = points_src[valid_mask]
    values_src_valid = values_src[valid_mask]

    if method == 'nearest':
        # Use k-d tree for nearest neighbor interpolation
        tree = cKDTree(points_src_valid)
        distances, indices = tree.query(points_tgt)
        values_tgt = values_src_valid[indices]
    else:  # linear
        # Use k-d tree with weighted average of nearest neighbors
        tree = cKDTree(points_src_valid)
        distances, indices = tree.query(points_tgt, k=4)

        # Avoid division by zero
        weights = 1.0 / (distances + 1e-10)
        weights = weights / weights.sum(axis=1, keepdims=True)

        values_tgt = (values_src_valid[indices] * weights).sum(axis=1)

    # Reshape to target grid shape
    return values_tgt.reshape(lat_tgt.shape)


