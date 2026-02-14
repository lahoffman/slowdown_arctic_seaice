"""
Grid operations and regridding utilities.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def regrid_to_cesm2(data, lat_source, lon_source, lat_target, lon_target):
    """
    Regrid data from source grid to CESM2 target grid using linear interpolation.

    Parameters
    ----------
    data : ndarray
        Data array with shape (time, lat, lon) or (lat, lon)
    lat_source : ndarray
        Source latitude values
    lon_source : ndarray
        Source longitude values
    lat_target : ndarray
        Target latitude values (CESM2 grid)
    lon_target : ndarray
        Target longitude values (CESM2 grid)

    Returns
    -------
    ndarray
        Regridded data on CESM2 grid
    """
    # Handle 2D or 3D data
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
        squeeze_output = True
    else:
        squeeze_output = False

    nt, nlat_src, nlon_src = data.shape

    # Create target grid
    lon_target_grid, lat_target_grid = np.meshgrid(lon_target, lat_target)
    target_points = np.stack([lat_target_grid.ravel(), lon_target_grid.ravel()], axis=-1)

    # Prepare output array
    nlat_target = len(lat_target)
    nlon_target = len(lon_target)
    data_regridded = np.empty((nt, nlat_target, nlon_target))

    # Interpolate each time step
    for t in range(nt):
        interp_func = RegularGridInterpolator(
            (lat_source, lon_source),
            data[t],
            bounds_error=False,
            fill_value=np.nan
        )
        interpolated = interp_func(target_points)
        data_regridded[t] = interpolated.reshape((nlat_target, nlon_target))

    if squeeze_output:
        return data_regridded[0]

    return data_regridded


def subset_region(data, lat, lon, lat_min, lat_max, lon_min, lon_max):
    """
    Subset data to a specific geographic region.

    Parameters
    ----------
    data : ndarray
        Data array with shape (time, lat, lon) or (lat, lon)
    lat : ndarray
        Latitude values
    lon : ndarray
        Longitude values
    lat_min, lat_max : float
        Latitude bounds
    lon_min, lon_max : float
        Longitude bounds

    Returns
    -------
    tuple
        (data_subset, lat_subset, lon_subset, lat_indices, lon_indices)
    """
    lat_inds = np.where((lat >= lat_min) & (lat <= lat_max))[0]
    lon_inds = np.where((lon >= lon_min) & (lon <= lon_max))[0]

    if data.ndim == 2:
        data_subset = data[np.ix_(lat_inds, lon_inds)]
    elif data.ndim == 3:
        data_subset = data[:, lat_inds, :][:, :, lon_inds]
    else:
        raise ValueError(f"Expected 2D or 3D data, got shape {data.shape}")

    return data_subset, lat[lat_inds], lon[lon_inds], lat_inds, lon_inds


def compute_area_weights(lat):
    """
    Compute cosine latitude weights for area-weighted averaging.

    Parameters
    ----------
    lat : ndarray
        Latitude values in degrees

    Returns
    -------
    ndarray
        Normalized weights (sum to 1)
    """
    weights = np.cos(np.deg2rad(lat))
    weights = weights / np.nansum(weights)
    return weights
