"""
Statistical analysis functions.
"""

import numpy as np


def compute_climatology(data, years, months, baseline=(1990, 2020)):
    """
    Compute monthly climatology over a baseline period.

    Parameters
    ----------
    data : ndarray
        Data array (1D time series or higher dimensions with time as first axis)
    years : ndarray
        Year for each time step
    months : ndarray
        Month (0-11) for each time step
    baseline : tuple
        (start_year, end_year) for climatology period

    Returns
    -------
    ndarray
        Monthly climatology (12 values, one per month)
    """
    baseline_mask = (years >= baseline[0]) & (years <= baseline[1])

    if data.ndim == 1:
        climatology = np.full(12, np.nan)
        for m in range(12):
            month_mask = baseline_mask & (months == m)
            if np.any(month_mask):
                climatology[m] = np.nanmean(data[month_mask])
    else:
        # Handle multi-dimensional data
        clim_shape = (12,) + data.shape[1:]
        climatology = np.full(clim_shape, np.nan)
        for m in range(12):
            month_mask = baseline_mask & (months == m)
            if np.any(month_mask):
                climatology[m] = np.nanmean(data[month_mask], axis=0)

    return climatology


def compute_anomalies(data, years, months, baseline=(1990, 2020)):
    """
    Compute anomalies by removing monthly climatology.

    Parameters
    ----------
    data : ndarray
        Data array
    years : ndarray
        Year for each time step
    months : ndarray
        Month (0-11) for each time step
    baseline : tuple
        (start_year, end_year) for climatology period

    Returns
    -------
    ndarray
        Anomalies (same shape as input data)
    """
    climatology = compute_climatology(data, years, months, baseline)

    # Map climatology to each time step
    if data.ndim == 1:
        clim_by_time = climatology[months]
    else:
        clim_by_time = climatology[months]

    anomalies = data - clim_by_time

    return anomalies


def normalize_by_std(data, baseline_mask=None):
    """
    Normalize data by standard deviation.

    Parameters
    ----------
    data : ndarray
        Data to normalize
    baseline_mask : ndarray, optional
        Boolean mask for baseline period. If None, uses all data.

    Returns
    -------
    ndarray
        Normalized data
    """
    if baseline_mask is None:
        baseline_mask = ~np.isnan(data)

    if data.ndim == 1:
        sigma = np.nanstd(data[baseline_mask])
    else:
        sigma = np.nanstd(data[baseline_mask], axis=0, keepdims=True)

    # Avoid division by zero
    sigma = np.where(sigma == 0, 1, sigma)

    return data / sigma


def compute_seasonal_mean(data, years, months, season_months):
    """
    Compute seasonal mean (e.g., JJA mean).

    Parameters
    ----------
    data : ndarray
        Data with shape (time, ...)
    years : ndarray
        Year for each time step
    months : ndarray
        Month (1-12) for each time step
    season_months : list
        List of months in season (e.g., [6, 7, 8] for JJA)

    Returns
    -------
    tuple
        (seasonal_data, unique_years) where seasonal_data has one value per year
    """
    unique_years = np.unique(years)

    # Get spatial dimensions
    if data.ndim == 1:
        output_shape = (len(unique_years),)
    else:
        output_shape = (len(unique_years),) + data.shape[1:]

    seasonal_data = np.full(output_shape, np.nan)

    for i, year in enumerate(unique_years):
        mask = (years == year) & np.isin(months, season_months)
        if np.any(mask):
            seasonal_data[i] = np.nanmean(data[mask], axis=0)

    return seasonal_data, unique_years
