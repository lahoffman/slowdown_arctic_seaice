"""
Trend analysis functions.
"""

import numpy as np


def compute_linear_trend(data, axis=0, return_stats=False):
    """
    Compute linear trend along specified axis.

    Parameters
    ----------
    data : ndarray
        Input data
    axis : int, optional
        Axis along which to compute trend (default: 0)
    return_stats : bool, optional
        If True, return additional statistics (p-value, r-squared)

    Returns
    -------
    tuple
        (slope, intercept) or (slope, intercept, pvalue, rsquared) if return_stats=True
    """
    from scipy import stats

    # Move axis to first position
    data_moved = np.moveaxis(data, axis, 0)
    n = data_moved.shape[0]
    x = np.arange(n)

    # Reshape for vectorized computation
    original_shape = data_moved.shape
    data_2d = data_moved.reshape(n, -1)

    slopes = np.full(data_2d.shape[1], np.nan)
    intercepts = np.full(data_2d.shape[1], np.nan)

    if return_stats:
        pvalues = np.full(data_2d.shape[1], np.nan)
        rsquared = np.full(data_2d.shape[1], np.nan)

    for i in range(data_2d.shape[1]):
        mask = ~np.isnan(data_2d[:, i])
        if np.sum(mask) < 2:
            continue

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            x[mask], data_2d[mask, i]
        )

        slopes[i] = slope
        intercepts[i] = intercept

        if return_stats:
            pvalues[i] = p_value
            rsquared[i] = r_value ** 2

    # Reshape back to original spatial dimensions
    new_shape = original_shape[1:]
    slopes = slopes.reshape(new_shape)
    intercepts = intercepts.reshape(new_shape)

    if return_stats:
        pvalues = pvalues.reshape(new_shape)
        rsquared = rsquared.reshape(new_shape)
        return slopes, intercepts, pvalues, rsquared

    return slopes, intercepts


def remove_linear_trend(data, axis=0):
    """
    Remove linear trend from data.

    Parameters
    ----------
    data : ndarray
        Input data
    axis : int, optional
        Axis along which to remove trend (default: 0)

    Returns
    -------
    tuple
        (detrended_data, trend_line)
    """
    # Move axis to first position
    data_moved = np.moveaxis(data, axis, 0)
    n = data_moved.shape[0]
    x = np.arange(n)

    # Reshape
    original_shape = data_moved.shape
    data_2d = data_moved.reshape(n, -1)

    detrended = np.zeros_like(data_2d)
    trends = np.zeros_like(data_2d)

    for i in range(data_2d.shape[1]):
        mask = ~np.isnan(data_2d[:, i])
        if np.sum(mask) < 2:
            detrended[:, i] = data_2d[:, i]
            continue

        coeffs = np.polyfit(x[mask], data_2d[mask, i], 1)
        trend = np.polyval(coeffs, x)
        trends[:, i] = trend
        detrended[:, i] = data_2d[:, i] - trend

    # Reshape back
    detrended = detrended.reshape(original_shape)
    trends = trends.reshape(original_shape)

    # Move axis back
    detrended = np.moveaxis(detrended, 0, axis)
    trends = np.moveaxis(trends, 0, axis)

    return detrended, trends
