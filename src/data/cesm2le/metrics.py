#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CESM2-LE Sea Ice Metrics

Functions to calculate sea ice extent and area from CESM2-LE data.

Author: Lauren Hoffman
Email: lhoffma2@ucsc.edu
"""

import numpy as np
import xarray as xr
from pathlib import Path
from typing import Optional


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
