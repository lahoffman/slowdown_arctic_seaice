#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sea Ice Metrics and Regridding Functions

This module provides functions to:
1. Calculate sea ice extent (15% concentration threshold)
2. Calculate sea ice area (no threshold)
3. Regrid AICE data to match SST grid

Author: Lauren Hoffman
Email: lhoffma2@ucsc.edu
"""

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path
from typing import Union, Optional, Tuple


def calculate_sea_ice_extent(
    aice_file: str,
    tarea_file: str,
    lat_threshold: float = 50.0,
    concentration_threshold: float = 0.15,
    aice_var: str = 'aice',
    save_output: Optional[str] = None
) -> np.ndarray:
    """
    Calculate Northern Hemisphere sea ice extent.

    Sea ice extent is the total area where sea ice concentration >= 15%.

    Parameters
    ----------
    aice_file : str
        Path to AICE NetCDF file
    tarea_file : str
        Path to file containing tarea (grid cell area) variable
    lat_threshold : float, optional
        Latitude threshold for Northern Hemisphere (default: 50°N)
    concentration_threshold : float, optional
        Sea ice concentration threshold (default: 0.15 for 15%)
    aice_var : str, optional
        Variable name in aice_file (default: 'aice')
    save_output : str, optional
        If provided, save extent timeseries to this NetCDF file

    Returns
    -------
    np.ndarray
        Sea ice extent timeseries in millions of km²
        Shape depends on input: [time] or [ensemble, time] or [ensemble, years]

    Examples
    --------
    >>> extent = calculate_sea_ice_extent(
    ...     aice_file='/path/to/aice_mon_JAN_1990-2100.nc',
    ...     tarea_file='/path/to/grid_file.nc'
    ... )
    >>> print(f"Mean extent: {extent.mean():.2f} million km²")
    """
    # Load data
    ds_aice = xr.open_dataset(aice_file)
    ds_grid = xr.open_dataset(tarea_file)

    # Get variables
    aice = ds_aice[aice_var].values
    tarea = ds_grid['tarea'].values  # Grid cell area in cm²
    lat = ds_grid['TLAT'].values if 'TLAT' in ds_grid else ds_grid['lat'].values

    # Create masks
    ice_mask = aice >= concentration_threshold
    arctic_mask = lat >= lat_threshold

    # Broadcast arctic mask to match aice dimensions
    arctic_mask_broadcast = np.broadcast_to(arctic_mask, aice.shape)

    # Combined mask
    arctic_ice = ice_mask & arctic_mask_broadcast

    # Broadcast tarea to match aice dimensions
    tarea_broadcast = np.broadcast_to(tarea, aice.shape)

    # Calculate extent: sum area where ice >= threshold
    # Sum over spatial dimensions (last 2 dimensions)
    spatial_dims = tuple(range(aice.ndim - 2, aice.ndim))
    siextentn = (arctic_ice * tarea_broadcast).sum(axis=spatial_dims) / 1e12  # million km²

    # Close datasets
    ds_aice.close()
    ds_grid.close()

    # Save if requested
    if save_output:
        _save_metric(siextentn, save_output, 'siextentn',
                     'Northern Hemisphere sea ice extent',
                     'million km²')

    return siextentn


def calculate_sea_ice_area(
    aice_file: str,
    tarea_file: str,
    lat_threshold: float = 50.0,
    aice_var: str = 'aice',
    save_output: Optional[str] = None
) -> np.ndarray:
    """
    Calculate Northern Hemisphere sea ice area.

    Sea ice area is the total area weighted by sea ice concentration
    (no threshold applied).

    Parameters
    ----------
    aice_file : str
        Path to AICE NetCDF file
    tarea_file : str
        Path to file containing tarea (grid cell area) variable
    lat_threshold : float, optional
        Latitude threshold for Northern Hemisphere (default: 50°N)
    aice_var : str, optional
        Variable name in aice_file (default: 'aice')
    save_output : str, optional
        If provided, save area timeseries to this NetCDF file

    Returns
    -------
    np.ndarray
        Sea ice area timeseries in millions of km²
        Shape depends on input: [time] or [ensemble, time] or [ensemble, years]

    Examples
    --------
    >>> area = calculate_sea_ice_area(
    ...     aice_file='/path/to/aice_mon_JAN_1990-2100.nc',
    ...     tarea_file='/path/to/grid_file.nc'
    ... )
    >>> print(f"Mean area: {area.mean():.2f} million km²")
    """
    # Load data
    ds_aice = xr.open_dataset(aice_file)
    ds_grid = xr.open_dataset(tarea_file)

    # Get variables
    aice = ds_aice[aice_var].values
    tarea = ds_grid['tarea'].values  # Grid cell area in cm²
    lat = ds_grid['TLAT'].values if 'TLAT' in ds_grid else ds_grid['lat'].values

    # Create arctic mask
    arctic_mask = lat >= lat_threshold

    # Broadcast masks to match aice dimensions
    arctic_mask_broadcast = np.broadcast_to(arctic_mask, aice.shape)
    tarea_broadcast = np.broadcast_to(tarea, aice.shape)

    # Apply arctic mask to concentration
    arctic_aice = np.where(arctic_mask_broadcast, aice, 0)

    # Calculate area: sum (concentration * area)
    # Sum over spatial dimensions (last 2 dimensions)
    spatial_dims = tuple(range(aice.ndim - 2, aice.ndim))
    siarean = (arctic_aice * tarea_broadcast).sum(axis=spatial_dims) / 1e12  # million km²

    # Close datasets
    ds_aice.close()
    ds_grid.close()

    # Save if requested
    if save_output:
        _save_metric(siarean, save_output, 'siarean',
                     'Northern Hemisphere sea ice area',
                     'million km²')

    return siarean


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


def _save_metric(
    data: np.ndarray,
    filepath: str,
    varname: str,
    long_name: str,
    units: str
) -> None:
    """Helper function to save a metric timeseries to NetCDF."""
    # Determine dimensions based on array shape
    if data.ndim == 1:
        dims = ('time',)
        coords = {'time': np.arange(len(data))}
    elif data.ndim == 2:
        dims = ('ensemble', 'time')
        coords = {
            'ensemble': np.arange(data.shape[0]),
            'time': np.arange(data.shape[1])
        }
    else:
        raise ValueError(f"Cannot save {data.ndim}D array. Expected 1D or 2D.")

    # Create dataset
    ds = xr.Dataset(
        {varname: (dims, data)},
        coords=coords
    )

    # Add attributes
    ds[varname].attrs['long_name'] = long_name
    ds[varname].attrs['units'] = units

    # Save
    encoding = {varname: {"zlib": True, "complevel": 4}}
    ds.to_netcdf(filepath, encoding=encoding)
    print(f"  Saved {varname} to: {filepath}")


def batch_process_monthly_files(
    aice_dir: str,
    tarea_file: str,
    output_dir: str,
    member_label: str = 'first50members',
    months: list = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                    'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'],
    calculate_extent: bool = True,
    calculate_area: bool = True
) -> None:
    """
    Process all monthly AICE files to calculate extent and/or area.

    Parameters
    ----------
    aice_dir : str
        Directory containing monthly AICE files
    tarea_file : str
        Path to grid file with tarea variable
    output_dir : str
        Directory for output files
    member_label : str, optional
        Member label in filenames (default: 'first50members')
    months : list, optional
        List of month labels to process
    calculate_extent : bool, optional
        Whether to calculate sea ice extent (default: True)
    calculate_area : bool, optional
        Whether to calculate sea ice area (default: True)

    Examples
    --------
    >>> batch_process_monthly_files(
    ...     aice_dir='/path/to/aice/mon',
    ...     tarea_file='/path/to/grid.nc',
    ...     output_dir='/path/to/metrics',
    ...     member_label='first50members'
    ... )
    """
    from pathlib import Path

    aice_path = Path(aice_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print(f"Batch processing monthly AICE files")
    print("="*70)

    for month in months:
        print(f"\nProcessing {month}...")

        # Find AICE file for this month
        aice_pattern = f"aice_cesmle_{member_label}_mon_{month}_*.nc"
        aice_files = list(aice_path.glob(aice_pattern))

        if not aice_files:
            print(f"  ⚠ No file found matching: {aice_pattern}")
            continue

        aice_file = str(aice_files[0])

        if calculate_extent:
            extent_output = output_path / f"siextentn_{member_label}_{month}.nc"
            extent = calculate_sea_ice_extent(
                aice_file=aice_file,
                tarea_file=tarea_file,
                save_output=str(extent_output)
            )
            print(f"  ✓ Extent: {extent.mean():.2f} ± {extent.std():.2f} million km²")

        if calculate_area:
            area_output = output_path / f"siarean_{member_label}_{month}.nc"
            area = calculate_sea_ice_area(
                aice_file=aice_file,
                tarea_file=tarea_file,
                save_output=str(area_output)
            )
            print(f"  ✓ Area: {area.mean():.2f} ± {area.std():.2f} million km²")

    print("\n" + "="*70)
    print("Batch processing complete!")
    print("="*70)


if __name__ == '__main__':
    print("Sea Ice Metrics Module")
    print("=" * 70)
    print("\nAvailable functions:")
    print("  - calculate_sea_ice_extent(): Calculate NH sea ice extent (15% threshold)")
    print("  - calculate_sea_ice_area(): Calculate NH sea ice area (concentration-weighted)")
    print("  - regrid_aice_to_sst(): Regrid AICE to SST grid")
    print("  - batch_process_monthly_files(): Process all monthly files at once")
    print("\nSee function docstrings for usage examples.")
