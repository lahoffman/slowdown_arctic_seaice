"""
Data input/output utilities.

Functions for loading and saving NetCDF files and common datasets.
"""

import numpy as np
import xarray as xr
import netCDF4 as nc
from pathlib import Path
from ..config import CESM2_GRID_FILE, LANDMASK_FILE


def load_netcdf(filepath, variables=None):
    """
    Load NetCDF file and optionally extract specific variables.

    Parameters
    ----------
    filepath : str or Path
        Path to NetCDF file
    variables : list of str, optional
        List of variable names to extract. If None, returns the dataset object.

    Returns
    -------
    dict or Dataset
        Dictionary of {var_name: array} if variables specified, otherwise Dataset
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"NetCDF file not found: {filepath}")

    dataset = nc.Dataset(filepath, 'r')

    if variables is None:
        return dataset

    data = {}
    for var in variables:
        if var not in dataset.variables:
            raise ValueError(f"Variable '{var}' not found in {filepath}")
        data[var] = np.array(dataset.variables[var])

    dataset.close()
    return data


def save_netcdf(data_dict, coords_dict, savepath, compression=True):
    """
    Save data to NetCDF file using xarray.

    Parameters
    ----------
    data_dict : dict
        Dictionary of {var_name: (dims_tuple, data_array)}
    coords_dict : dict
        Dictionary of {coord_name: (dims, values)}
    savepath : str or Path
        Path where to save the NetCDF file
    compression : bool, optional
        Whether to use compression (default: True)

    Example
    -------
    >>> data_dict = {
    ...     "sst": (("time", "lat", "lon"), sst_data),
    ...     "lat": (("lat",), lat_values),
    ...     "lon": (("lon",), lon_values)
    ... }
    >>> coords_dict = {
    ...     "time": np.arange(100),
    ...     "lat": np.arange(192),
    ...     "lon": np.arange(288)
    ... }
    >>> save_netcdf(data_dict, coords_dict, "output.nc")
    """
    savepath = Path(savepath)
    savepath.parent.mkdir(parents=True, exist_ok=True)

    ds = xr.Dataset(data_dict, coords=coords_dict)

    if compression:
        encoding = {var: {"zlib": True, "complevel": 4} for var in ds.data_vars}
    else:
        encoding = {}

    ds.to_netcdf(savepath, format='NETCDF4', encoding=encoding)
    print(f'NetCDF file saved to {savepath}')


def load_cesm2_grid():
    """
    Load CESM2 latitude and longitude grid.

    Returns
    -------
    tuple
        (lat, lon) arrays
    """
    data = load_netcdf(CESM2_GRID_FILE, variables=['lat', 'lon'])
    return data['lat'], data['lon']


def load_landmask():
    """
    Load CESM2 landmask.

    Returns
    -------
    ndarray
        Landmask array (0 for ocean, 1 for land)
    """
    data = load_netcdf(LANDMASK_FILE, variables=['landmask'])
    return data['landmask']
