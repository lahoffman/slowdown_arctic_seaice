"""
General utility functions.

These are the functions that were previously in functions_general.py
"""

import numpy as np
from scipy.ndimage import uniform_filter1d


def ncdisp(dataset):
    """
    Display NetCDF dataset information.

    Parameters
    ----------
    dataset : netCDF4.Dataset
        NetCDF dataset object
    """
    print("Dimensions:")
    for dim in dataset.dimensions:
        print(f"  {dim}: {len(dataset.dimensions[dim])}")

    print("\nVariables:")
    for var in dataset.variables:
        var_obj = dataset.variables[var]
        print(f"  {var}: {var_obj.shape} {var_obj.dtype}")


def movmean(data, window, axis=0, mode='nearest'):
    """
    Calculate moving mean along specified axis.

    Parameters
    ----------
    data : ndarray
        Input data array
    window : int
        Window size for moving average
    axis : int, optional
        Axis along which to compute the moving mean (default: 0)
    mode : str, optional
        How to handle boundaries. Options: 'nearest', 'constant', 'reflect', 'mirror', 'wrap'

    Returns
    -------
    ndarray
        Smoothed data with same shape as input
    """
    return uniform_filter1d(data, size=window, axis=axis, mode=mode)


def detrend_linear(data, axis=0):
    """
    Remove linear trend from data along specified axis.

    Parameters
    ----------
    data : ndarray
        Input data
    axis : int, optional
        Axis along which to detrend (default: 0)

    Returns
    -------
    tuple
        (detrended_data, trend_line)
    """
    shape = data.shape
    n = shape[axis]

    # Move axis to first position for easier processing
    data_moved = np.moveaxis(data, axis, 0)
    original_shape = data_moved.shape

    # Reshape to 2D: (n, other_dims_flattened)
    data_2d = data_moved.reshape(n, -1)

    # Compute trend for each column
    x = np.arange(n)
    trends = np.zeros_like(data_2d)
    detrended = np.zeros_like(data_2d)

    for i in range(data_2d.shape[1]):
        # Skip if all NaN
        if np.all(np.isnan(data_2d[:, i])):
            detrended[:, i] = data_2d[:, i]
            continue

        # Fit linear trend
        mask = ~np.isnan(data_2d[:, i])
        if np.sum(mask) < 2:
            detrended[:, i] = data_2d[:, i]
            continue

        coeffs = np.polyfit(x[mask], data_2d[mask, i], 1)
        trend = np.polyval(coeffs, x)
        trends[:, i] = trend
        detrended[:, i] = data_2d[:, i] - trend

    # Reshape back
    trends = trends.reshape(original_shape)
    detrended = detrended.reshape(original_shape)

    # Move axis back
    trends = np.moveaxis(trends, 0, axis)
    detrended = np.moveaxis(detrended, 0, axis)

    return detrended, trends
