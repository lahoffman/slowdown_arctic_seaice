#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ERSSTv5 Climate Indices

Calculate ENSO, ENSO CP/TP, and IPO indices from regridded ERSSTv5 data.

Indices
-------
- Niño3.4: Standard ENSO index (5S-5N, 170W-120W)
- Niño3:   Eastern Pacific ENSO (5S-5N, 150W-90W)
- Niño4:   Central Pacific ENSO (5S-5N, 160E-150W)
- N_CT:    Cold-tongue index = N3 - α*N4
- N_WP:    Warm-pool index   = N4 - α*N3  (α = 2/5 if N3*N4 > 0, else 0)
- IPO:     Interdecadal Pacific Oscillation (tropical - 0.5*(N.Pac + S.Pac))
- Arctic SST Index: cos(lat)-weighted mean SST north of 60N, all longitudes

Reference: Henley et al. (2015) for IPO; Takahashi et al. for CP/TP split

Author: Lauren Hoffman
Email: lhoffma2@ucsc.edu
"""

import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc
from scipy.ndimage import uniform_filter1d
from scipy.signal import cheby1, filtfilt
from pathlib import Path
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _area_mean_monthly(
    sst: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    latmin: float,
    latmax: float,
    lonmin: float,
    lonmax: float
) -> np.ndarray:
    """
    Compute cos(lat)-weighted area mean over a lat/lon box.

    Parameters
    ----------
    sst : np.ndarray
        SST data, shape (ntime, nlat, nlon)
    lat, lon : np.ndarray
        1D coordinate arrays
    latmin, latmax, lonmin, lonmax : float
        Box bounds (lon in 0-360°E)

    Returns
    -------
    np.ndarray
        Monthly area-mean timeseries, shape (ntime,)
    """
    # 2D spatial mask for the box
    mask = (lat >= latmin) & (lat <= latmax) & (lon >= lonmin) & (lon <= lonmax)
    # (nlat, nlon) boolean

    # Cos(lat) weights — zero outside the box
    wlat = np.where(mask, np.cos(np.deg2rad(lat)), 0.0)  # (nlat, nlon)
    wsum = np.nansum(wlat)

    # Weighted sum over spatial dims (axis 1 and 2), normalised by weight sum
    # sst: (ntime, nlat, nlon); wlat: (nlat, nlon)
    return np.nansum(sst * wlat[None, :, :], axis=(1, 2)) / wsum



def _monthly_anoms(
    ts: np.ndarray,
    years: np.ndarray,
    baseline: Tuple[int, int] = (1990, 2020)
) -> np.ndarray:
    """
    Remove monthly climatology over a baseline period.

    Parameters
    ----------
    ts : np.ndarray
        Monthly timeseries, shape (ntime,)
    years : np.ndarray
        Year for each time step, shape (ntime,)
    baseline : tuple, optional
        (start_year, end_year) inclusive (default: 1990-2020)

    Returns
    -------
    np.ndarray
        Anomalies, shape (ntime,)
    """
    ntime = len(ts)
    months = np.arange(ntime) % 12  # assumes series starts in January
    mask_base = (years >= baseline[0]) & (years <= baseline[1])

    clim = np.full(12, np.nan)
    for m in range(12):
        sel = mask_base & (months == m)
        if np.any(sel):
            clim[m] = np.nanmean(ts[sel])

    return ts - clim[months]


def _detrend_linear(x: np.ndarray) -> np.ndarray:
    """Remove linear trend from a 1D array."""
    t = np.arange(len(x))
    coeffs = np.polyfit(t, x, 1)
    return x - np.polyval(coeffs, t)


def _normalize_by_baseline_std(
    x: np.ndarray,
    years: np.ndarray,
    baseline: Tuple[int, int] = (1990, 2020)
) -> np.ndarray:
    """Normalize by standard deviation over baseline period."""
    mask = (years >= baseline[0]) & (years <= baseline[1])
    sigma = np.nanstd(x[mask])
    return x / sigma

def enso_phase_labels(
    index_ts: np.ndarray,
    threshold: float = 0.4,
    min_length: int = 6
) -> np.ndarray:
    """
    Label each time step as El Niño (+1), La Niña (-1), or Neutral (0).

    Parameters
    ----------
    index_ts : np.ndarray
        Normalized ENSO index timeseries (σ ≈ 1)
    threshold : float, optional
        Standard deviation threshold (default: 0.4)
    min_length : int, optional
        Minimum consecutive months to qualify as an event (default: 6).
        Set to 1 to disable (simple threshold labeling).

    Returns
    -------
    np.ndarray
        Integer labels: +1 (El Niño), -1 (La Niña), 0 (Neutral)
    """
    ntime = len(index_ts)
    labels = np.zeros(ntime, dtype=int)

    if min_length <= 1:
        # Simple threshold
        labels[index_ts >= threshold] = 1
        labels[index_ts <= -threshold] = -1
        return labels

    # El Niño: consecutive months above threshold
    mask_pos = index_ts >= threshold
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

    # La Niña: consecutive months below -threshold
    mask_neg = index_ts <= -threshold
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


def load_grid_latlon(grid_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the 2D CESM2 ocean grid lat/lon from the grid file produced by
    scripts/01_cesm2le_grid.py.

    Parameters
    ----------
    grid_file : str or Path
        Path to the grid NetCDF file (e.g.
        ``paths.CESM2LE_SST_DIR / 'grid' / 'cesm2le_sst_grid.nc'``).

    Returns
    -------
    lat : np.ndarray, shape (nj, ni)
        Latitude in degrees North.
    lon : np.ndarray, shape (nj, ni)
        Longitude in 0–360°E.
    """
    grid_file = Path(grid_file)
    ds = nc.Dataset(str(grid_file), 'r')
    ds.set_auto_mask(False)

    lat = np.array(ds.variables['lat'][:], dtype=np.float64)
    lon = np.array(ds.variables['lon'][:], dtype=np.float64)

    ds.close()
    return lat, lon

# ─────────────────────────────────────────────────────────────────────────────
# ENSO: Niño3.4 index
# ─────────────────────────────────────────────────────────────────────────────

def compute_nino34_index(
    sst_obs: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    years: np.ndarray,
    baseline: Tuple[int, int] = (1990, 2020),
    smooth_months: int = 5,
    threshold: float = 0.4,
    min_length: int = 6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Niño3.4 ENSO index.

    Steps: area mean → monthly anomalies → detrend → smooth → normalize.

    Parameters
    ----------
    sst_obs : np.ndarray
        Regridded SST, shape (ntime, nlat, nlon)
    lat, lon : np.ndarray
        1D coordinate arrays (lon in 0-360°E)
    years : np.ndarray
        Year for each time step
    baseline : tuple, optional
        Baseline period for climatology and normalization (default: 1990-2020)
    smooth_months : int, optional
        Running mean window in months (default: 5)
    threshold : float, optional
        Normalized threshold for phase labeling (default: 0.4)
    min_length : int, optional
        Minimum event length in months (default: 6)

    Returns
    -------
    nino34 : np.ndarray
        Normalized Niño3.4 index, shape (ntime,)
    labels : np.ndarray
        Phase labels (+1, 0, -1), shape (ntime,)

    Examples
    --------
    >>> nino34, labels = compute_nino34_index(sst_obs, lat, lon, years)
    """

    # Area mean over Niño3.4 box (5S-5N, 170W-120W = 190-240°E)
    ts = _area_mean_monthly(sst_obs, lat, lon,
                            latmin=-5, latmax=5, lonmin=190, lonmax=240)

    # Monthly anomalies
    anoms = _monthly_anoms(ts, years, baseline=baseline)

    # Linear detrend
    anoms = _detrend_linear(anoms)

    # Smooth
    anoms = uniform_filter1d(anoms, size=smooth_months, mode='nearest')

    # Normalize
    nino34 = _normalize_by_baseline_std(anoms, years, baseline=baseline)

    # Phase labels
    labels = enso_phase_labels(nino34, threshold=threshold, min_length=min_length)

    return nino34, labels


# ─────────────────────────────────────────────────────────────────────────────
# ENSO CP/TP: Niño3, Niño4, N_CT, N_WP
# ─────────────────────────────────────────────────────────────────────────────

def compute_enso_cp_tp_indices(
    sst_obs: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    years: np.ndarray,
    baseline: Tuple[int, int] = (1990, 2020),
    smooth_months: int = 5,
    threshold: float = 0.4
) -> dict:
    """
    Compute Niño3, Niño4, Niño3.4, cold-tongue (N_CT), and warm-pool (N_WP) indices.

    The CT/WP split follows Takahashi et al.:
        N_CT = N3 - α*N4
        N_WP = N4 - α*N3
        α = 2/5 if N3*N4 > 0, else 0

    Parameters
    ----------
    sst_obs : np.ndarray
        Regridded SST, shape (ntime, nlat, nlon)
    lat, lon : np.ndarray
        1D coordinate arrays (lon in 0-360°E)
    years : np.ndarray
        Year for each time step
    baseline : tuple, optional
        Baseline period (default: 1990-2020)
    smooth_months : int, optional
        Running mean window (default: 5)
    threshold : float, optional
        Threshold for phase labeling (default: 0.4)

    Returns
    -------
    dict with keys:
        'n3', 'n4', 'n34', 'n_ct', 'n_wp' : np.ndarray, shape (ntime,)
        'labels_n3', 'labels_n4', 'labels_n34',
        'labels_n_ct', 'labels_n_wp'        : np.ndarray, shape (ntime,)

    Examples
    --------
    >>> result = compute_enso_cp_tp_indices(sst_obs, lat, lon, years)
    >>> nino_ct = result['n_ct']
    >>> labels  = result['labels_n_ct']
    """
    # Area means for three boxes
    n34_ts = _area_mean_monthly(sst_obs, lat, lon,
                                latmin=-5, latmax=5, lonmin=190, lonmax=240)
    n3_ts  = _area_mean_monthly(sst_obs, lat, lon,
                                latmin=-5, latmax=5, lonmin=210, lonmax=270)
    n4_ts  = _area_mean_monthly(sst_obs, lat, lon,
                                latmin=-5, latmax=5, lonmin=160, lonmax=210)

    # Monthly anomalies
    n34 = _monthly_anoms(n34_ts, years, baseline=baseline)
    n3  = _monthly_anoms(n3_ts,  years, baseline=baseline)
    n4  = _monthly_anoms(n4_ts,  years, baseline=baseline)

    # Linear detrend
    n34, n3, n4 = [_detrend_linear(x) for x in (n34, n3, n4)]

    # CT / WP split: α = 2/5 when N3 and N4 have the same sign
    alpha = np.where(n3 * n4 > 0, 2.0 / 5.0, 0.0)
    n_ct = n3 - alpha * n4
    n_wp = n4 - alpha * n3

    # Smooth all five
    indices = [n34, n3, n4, n_ct, n_wp]
    n34, n3, n4, n_ct, n_wp = [
        uniform_filter1d(x, size=smooth_months, mode='nearest') for x in indices
    ]

    # Normalize by baseline std
    n34, n3, n4, n_ct, n_wp = [
        _normalize_by_baseline_std(x, years, baseline=baseline)
        for x in (n34, n3, n4, n_ct, n_wp)
    ]

    # Phase labels (simple threshold, no minimum length for CT/WP)
    def _labels(x):
        return enso_phase_labels(x, threshold=threshold, min_length=1)

    return {
        'n34': n34, 'n3': n3, 'n4': n4, 'n_ct': n_ct, 'n_wp': n_wp,
        'labels_n34':  _labels(n34),
        'labels_n3':   _labels(n3),
        'labels_n4':   _labels(n4),
        'labels_n_ct': _labels(n_ct),
        'labels_n_wp': _labels(n_wp),
    }


# ─────────────────────────────────────────────────────────────────────────────
# IPO: Interdecadal Pacific Oscillation
# ─────────────────────────────────────────────────────────────────────────────

def chebyshev_lowpass(
    ts: np.ndarray,
    cutoff_years: float = 13.0,
    order: int = 4,
    rp: float = 0.05
) -> np.ndarray:
    """
    Apply a Chebyshev Type-I low-pass filter to a monthly timeseries.

    Follows Henley et al. (2015): 13-year cutoff.

    Parameters
    ----------
    ts : np.ndarray
        Monthly timeseries, shape (ntime,) or (n, ntime)
    cutoff_years : float, optional
        Low-pass cutoff period in years (default: 13)
    order : int, optional
        Filter order (default: 4)
    rp : float, optional
        Maximum ripple in the passband in dB (default: 0.05)

    Returns
    -------
    np.ndarray
        Filtered timeseries, same shape as input
    """
    f_sampling = 12.0           # monthly
    f_nyquist  = f_sampling / 2.0
    f_cutoff   = 1.0 / cutoff_years
    fc_norm    = f_cutoff / f_nyquist

    b, a = cheby1(order, rp, fc_norm, btype='low')
    return filtfilt(b, a, ts, axis=-1)


def compute_ipo_index(
    sst_obs: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    years: np.ndarray,
    baseline: Tuple[int, int] = (1990, 2020),
    filter_cutoff_years: float = 13.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Interdecadal Pacific Oscillation (IPO) index.

    IPO = Tropical Pacific SSTa - 0.5 * (N.Pacific SSTa + S.Pacific SSTa)

    Regions (Henley et al. 2015):
        Tropical:  10S-10N, 170E-90W  (170-270°E)
        N.Pacific: 25N-45N, 140E-145W (140-215°E)
        S.Pacific: 50S-15S, 150E-160W (150-200°E)

    Parameters
    ----------
    sst_obs : np.ndarray
        Regridded SST, shape (ntime, nlat, nlon)
    lat, lon : np.ndarray
        1D coordinate arrays (lon in 0-360°E)
    years : np.ndarray
        Year for each time step
    baseline : tuple, optional
        Baseline period for monthly climatology (default: 1990-2020)
    filter_cutoff_years : float, optional
        Chebyshev filter cutoff in years (default: 13)

    Returns
    -------
    ipo : np.ndarray
        Unfiltered IPO index, shape (ntime,)
    ipo_filtered : np.ndarray
        Low-pass filtered IPO index, shape (ntime,)
    labels : np.ndarray
        Phase labels for unfiltered IPO (+1, 0, -1)
    labels_filtered : np.ndarray
        Phase labels for filtered IPO (+1, 0, -1)

    Examples
    --------
    >>> ipo, ipo_f, labels, labels_f = compute_ipo_index(sst_obs, lat, lon, years)
    """
    ntime = sst_obs.shape[0]
    nlat, nlon = sst_obs.shape[1], sst_obs.shape[2]

    # Monthly climatology over baseline (full 2D field)
    month_idx = np.arange(ntime) % 12
    mask_base = (years >= baseline[0]) & (years <= baseline[1])

    clim = np.full((12, nlat, nlon), np.nan)
    for m in range(12):
        sel = mask_base & (month_idx == m)
        if np.any(sel):
            clim[m] = np.nanmean(sst_obs[sel], axis=0)

    sst_anom = sst_obs - clim[month_idx]


    # Box means
    trop = _area_mean_monthly(sst_anom, lat, lon, -10,  10,  170, 270)
    npac = _area_mean_monthly(sst_anom, lat, lon,  25,  45,  140, 215)
    spac = _area_mean_monthly(sst_anom, lat, lon, -50, -15,  150, 200)

    ipo = trop - 0.5 * (npac + spac)

    # Low-pass filter
    ipo_filtered = chebyshev_lowpass(ipo, cutoff_years=filter_cutoff_years)

    # Labels (simple sign threshold)
    labels          = np.sign(ipo).astype(int)
    labels_filtered = np.sign(ipo_filtered).astype(int)

    return ipo, ipo_filtered, labels, labels_filtered


# ─────────────────────────────────────────────────────────────────────────────
# Arctic SST Index
# ─────────────────────────────────────────────────────────────────────────────

def _arctic_phase_labels(
    index_ts: np.ndarray,
    threshold: float = 1.0,
) -> np.ndarray:
    """
    Label each time step as positive (+1), negative (-1), or neutral (0)
    based on a threshold applied to a normalized index.

    For a normalized index (mean ~0, std ~1) with threshold=1.0 this gives:
        positive : index > +1
        negative : index < -1
        neutral  : otherwise

    Parameters
    ----------
    index_ts : np.ndarray
        Normalized index timeseries, shape (ntime,).
    threshold : float, optional
        Number of standard deviations for labelling (default: 1.0).

    Returns
    -------
    np.ndarray
        Integer labels: +1 (positive/warm), -1 (negative/cold), 0 (neutral).
    """
    labels = np.zeros(len(index_ts), dtype=int)
    labels[index_ts > threshold]  =  1
    labels[index_ts < -threshold] = -1
    return labels


def compute_arctic_sst_index(
    sst_obs: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    years: np.ndarray,
    baseline: Tuple[int, int] = (1990, 2020),
    smooth_months: int = 5,
    threshold: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Arctic SST Index from observed SST.

    Defined as the cos(lat)-weighted area mean of SST over all longitudes
    and all latitudes north of 60N, processed in the same way as Nino3.4:
    area mean -> monthly anomalies -> linear detrend -> smooth -> normalize.

    Labels use a +/- 1 standard deviation threshold on the normalized index.

    Parameters
    ----------
    sst_obs : np.ndarray
        Regridded SST, shape (ntime, nlat, nlon).
    lat, lon : np.ndarray
        1D coordinate arrays (lon in 0-360 E).
    years : np.ndarray
        Year for each time step.
    baseline : tuple, optional
        Baseline period for climatology and normalization (default: 1990-2020).
    smooth_months : int, optional
        Running mean window in months (default: 5).
    threshold : float, optional
        Standard-deviation threshold for phase labelling (default: 1.0).

    Returns
    -------
    arctic_sst : np.ndarray
        Normalized Arctic SST index, shape (ntime,).
    labels : np.ndarray
        Phase labels (+1 positive, -1 negative, 0 neutral), shape (ntime,).
    """
    # Area mean: all longitudes, latitudes north of 60N
    # lonmin=0, lonmax=360 captures the full longitude range
    ts = _area_mean_monthly(sst_obs, lat, lon,
                            latmin=60, latmax=90, lonmin=0, lonmax=360)

    # Monthly anomalies
    anoms = _monthly_anoms(ts, years, baseline=baseline)

    # Linear detrend
    anoms = _detrend_linear(anoms)

    # Smooth
    anoms = uniform_filter1d(anoms, size=smooth_months, mode='nearest')

    # Normalize by baseline std
    arctic_sst = _normalize_by_baseline_std(anoms, years, baseline=baseline)

    # Phase labels
    labels = _arctic_phase_labels(arctic_sst, threshold=threshold)

    return arctic_sst, labels


# ─────────────────────────────────────────────────────────────────────────────
# Save helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_nino34(
    nino34: np.ndarray,
    labels: np.ndarray,
    dates: pd.DatetimeIndex,
    output_file: str
) -> None:
    """Save Niño3.4 index and labels to NetCDF."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    ds = xr.Dataset(
        {
            "nino34": (("nt",), nino34),
            "labels": (("nt",), labels),
        },
        coords={"nt": np.arange(len(nino34))},
    )
    ds.attrs['description'] = 'ERSSTv5 Niño3.4 index and ENSO phase labels'

    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(output_file, encoding=encoding)
    print(f"✓ Saved Niño3.4 to: {output_file}")


def save_enso_cp_tp(
    result: dict,
    dates: pd.DatetimeIndex,
    output_file: str
) -> None:
    """Save CP/TP ENSO indices and labels to NetCDF."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    data_vars = {k: (("nt",), v) for k, v in result.items()}
    ds = xr.Dataset(data_vars, coords={"nt": np.arange(len(result['n34']))})
    ds.attrs['description'] = 'ERSSTv5 Niño3/4/3.4/CT/WP indices and ENSO phase labels'

    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(output_file, encoding=encoding)
    print(f"✓ Saved ENSO CP/TP to: {output_file}")


def save_ipo(
    ipo: np.ndarray,
    ipo_filtered: np.ndarray,
    labels: np.ndarray,
    labels_filtered: np.ndarray,
    dates: pd.DatetimeIndex,
    output_file: str
) -> None:
    """Save IPO index and labels to NetCDF."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    ds = xr.Dataset(
        {
            "ipo":              (("nt",), ipo),
            "ipo_filtered":     (("nt",), ipo_filtered),
            "labels":           (("nt",), labels),
            "labels_filtered":  (("nt",), labels_filtered),
        },
        coords={"nt": np.arange(len(ipo))},
    )
    ds.attrs['description'] = 'ERSSTv5 IPO index (Henley et al. 2015) and phase labels'
    ds.attrs['filter'] = 'Chebyshev Type-I low-pass, 13-year cutoff'

    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(output_file, encoding=encoding)
    print(f"✓ Saved IPO to: {output_file}")


def save_arctic_sst(
    arctic_sst: np.ndarray,
    labels: np.ndarray,
    dates: pd.DatetimeIndex,
    output_file: str
) -> None:
    """Save Arctic SST Index and labels to NetCDF."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    ds = xr.Dataset(
        {
            "arctic_sst": (("nt",), arctic_sst),
            "labels":     (("nt",), labels),
        },
        coords={"nt": np.arange(len(arctic_sst))},
    )
    ds.attrs['description'] = (
        'ERSSTv5 Arctic SST Index (cos-lat weighted mean, >60N all lons) '
        'and phase labels (+1 positive, -1 negative, 0 neutral)'
    )

    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(output_file, encoding=encoding)
    print(f"✓ Saved Arctic SST to: {output_file}")
