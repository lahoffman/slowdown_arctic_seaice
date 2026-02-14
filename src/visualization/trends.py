"""
Trend analysis utilities for figure generation.
"""

import numpy as np


def moving_decadal_trend(y, window=10):
    """
    Compute moving linear trend (slope) over a rolling window.

    Parameters
    ----------
    y : ndarray
        Time series data
    window : int, optional
        Window size in years (default: 10)

    Returns
    -------
    ndarray
        Slopes of length len(y)-window
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    slopes = np.full(n - window, np.nan)

    x = np.arange(window, dtype=float)
    for j in range(n - window):
        yy = y[j : j + window]
        mask = np.isfinite(yy)
        if mask.sum() < 2:
            continue
        # Re-fit on valid indices only
        xx = x[mask]
        yy2 = yy[mask]
        m, b = np.polyfit(xx, yy2, 1)
        slopes[j] = m

    return slopes


def classify_slowdown(slopes, threshold):
    """
    Classify trends as slowdown or not based on threshold.

    Parameters
    ----------
    slopes : ndarray
        Trend slopes (nt,) or (n_member, nt)
    threshold : ndarray
        Threshold values (broadcastable to slopes shape)

    Returns
    -------
    ndarray
        Binary mask: 1 for slowdown, 0 for no slowdown
    """
    return (slopes > threshold).astype(int)


def compute_trend_statistics(trends, axis=None):
    """
    Compute mean and standard deviation of trends.

    Parameters
    ----------
    trends : ndarray
        Trend values
    axis : int or tuple, optional
        Axis along which to compute statistics

    Returns
    -------
    tuple
        (mean_trend, std_trend)
    """
    mean_trend = np.nanmean(trends, axis=axis)
    std_trend = np.nanstd(trends, axis=axis)

    return mean_trend, std_trend


def compute_slowdown_threshold(trends, axis=None, method='mean_plus_std'):
    """
    Compute slowdown threshold from trend distribution.

    Parameters
    ----------
    trends : ndarray
        Trend values
    axis : int or tuple, optional
        Axis along which to compute threshold
    method : str, optional
        Method for computing threshold: 'mean_plus_std', 'percentile'

    Returns
    -------
    ndarray
        Threshold values
    """
    if method == 'mean_plus_std':
        mean_trend = np.nanmean(trends, axis=axis)
        std_trend = np.nanstd(trends, axis=axis)
        threshold = mean_trend + std_trend

    elif method == 'percentile':
        threshold = np.nanpercentile(trends, 75, axis=axis)

    else:
        raise ValueError(f"Unknown method: {method}")

    return threshold
