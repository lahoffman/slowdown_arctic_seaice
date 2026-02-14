"""
Climate index calculations (ENSO, IPO, etc.).
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from ..config import NINO34_REGION, BASELINE_PERIOD, NINO34_THRESHOLD, NINO34_MIN_LENGTH
from ..utils.grid_ops import subset_region, compute_area_weights
from .statistics import compute_anomalies, normalize_by_std


def compute_nino34_index(sst, lat, lon, years,
                         baseline=None,
                         region=None,
                         smooth_window=5,
                         normalize=True):
    """
    Compute Niño 3.4 index from SST data.

    Parameters
    ----------
    sst : ndarray
        SST data with shape (time, lat, lon)
    lat : ndarray
        Latitude values
    lon : ndarray
        Longitude values
    years : ndarray
        Year for each time step
    baseline : tuple, optional
        (start_year, end_year) for climatology baseline (default: from config)
    region : dict, optional
        Region boundaries (lat_min, lat_max, lon_min, lon_max)
    smooth_window : int, optional
        Window size for moving average (default: 5)
    normalize : bool, optional
        Whether to normalize by standard deviation (default: True)

    Returns
    -------
    ndarray
        Niño 3.4 index values for each time step
    """
    if baseline is None:
        baseline = BASELINE_PERIOD

    if region is None:
        region = NINO34_REGION

    # Subset Niño 3.4 region
    sst_subset, lat_subset, lon_subset, _, _ = subset_region(
        sst, lat, lon,
        region['lat_min'], region['lat_max'],
        region['lon_min'], region['lon_max']
    )

    # Compute area-weighted mean
    weights = compute_area_weights(lat_subset)
    w2d = weights[:, None]  # Broadcast over longitude

    # Area-weighted mean
    sst_lat_weighted = np.nansum(sst_subset * w2d[None, :, :], axis=1)
    sst_regional_mean = np.nanmean(sst_lat_weighted, axis=1)

    # Compute anomalies
    months = np.arange(len(years)) % 12
    nino34_anom = compute_anomalies(sst_regional_mean, years, months, baseline)

    # Remove linear trend
    coefficients = np.polyfit(np.arange(len(nino34_anom)), nino34_anom, 1)
    trend = np.polyval(coefficients, np.arange(len(nino34_anom)))
    nino34_detrended = nino34_anom - trend

    # Apply smoothing
    if smooth_window > 1:
        nino34_smooth = uniform_filter1d(nino34_detrended, size=smooth_window,
                                         axis=0, mode='nearest')
    else:
        nino34_smooth = nino34_detrended

    # Normalize by standard deviation
    if normalize:
        baseline_mask = (years >= baseline[0]) & (years <= baseline[1])
        sigma = np.nanstd(nino34_smooth[baseline_mask])
        nino34_smooth = nino34_smooth / sigma

    return nino34_smooth


def compute_ipo_index(sst, lat, lon, years, baseline=None):
    """
    Compute Interdecadal Pacific Oscillation (IPO) index.

    This is a placeholder - implement based on your specific IPO calculation method.

    Parameters
    ----------
    sst : ndarray
        SST data with shape (time, lat, lon)
    lat : ndarray
        Latitude values
    lon : ndarray
        Longitude values
    years : ndarray
        Year for each time step
    baseline : tuple, optional
        (start_year, end_year) for baseline period

    Returns
    -------
    ndarray
        IPO index values
    """
    # TODO: Implement IPO calculation
    # This would involve computing SST patterns over multiple Pacific regions
    # and potentially performing EOF analysis
    raise NotImplementedError("IPO calculation not yet implemented")


def label_enso_phases(nino34_index, threshold=None, min_length=None):
    """
    Label ENSO phases (El Niño, La Niña, Neutral) from Niño 3.4 index.

    Parameters
    ----------
    nino34_index : ndarray
        Niño 3.4 index values
    threshold : float, optional
        Threshold for defining El Niño/La Niña (default: from config)
    min_length : int, optional
        Minimum consecutive months for an event (default: from config)

    Returns
    -------
    ndarray
        Phase labels: 1 (El Niño), -1 (La Niña), 0 (Neutral)
    """
    if threshold is None:
        threshold = NINO34_THRESHOLD

    if min_length is None:
        min_length = NINO34_MIN_LENGTH

    ntime = len(nino34_index)
    labels = np.zeros_like(nino34_index, dtype=int)

    # Label El Niño events
    mask_pos = nino34_index >= threshold
    i = 0
    while i < ntime:
        if mask_pos[i]:
            j = i
            while j < ntime and mask_pos[j]:
                j += 1
            if j - i >= min_length:
                labels[i:j] = 1
            i = j
        else:
            i += 1

    # Label La Niña events
    mask_neg = nino34_index <= -threshold
    i = 0
    while i < ntime:
        if mask_neg[i]:
            j = i
            while j < ntime and mask_neg[j]:
                j += 1
            if j - i >= min_length:
                labels[i:j] = -1
            i = j
        else:
            i += 1

    return labels
