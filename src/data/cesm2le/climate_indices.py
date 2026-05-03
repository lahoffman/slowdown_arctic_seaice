#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CESM2-LE Climate Indices

Calculate ENSO, ENSO CP/TP, and IPO indices from CESM2 Large Ensemble SST data.

Indices
-------
- Niño3.4: Standard ENSO index (5S-5N, 170W-120W)
- Niño3:   Eastern Pacific ENSO (5S-5N, 150W-90W)
- Niño4:   Central Pacific ENSO (5S-5N, 160E-150W)
- N_CT:    Cold-tongue index = N3 - α*N4
- N_WP:    Warm-pool index   = N4 - α*N3  (α = 2/5 if N3*N4 > 0, else 0)
- IPO:     Interdecadal Pacific Oscillation (tropical - 0.5*(N.Pac + S.Pac))

Key difference from ERSSTv5 version: the forced signal (ensemble mean) is removed
before computing indices to isolate internal variability in each ensemble member.

Reference: Henley et al. (2015) for IPO; Takahashi et al. for CP/TP split.

Author: Lauren Hoffman
Email: lhoffma2@ucsc.edu
"""

import numpy as np
import xarray as xr
import netCDF4 as nc
from scipy.ndimage import uniform_filter1d
from scipy.signal import cheby1, filtfilt
from pathlib import Path
from typing import Optional, Tuple, List


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MONTH_LABELS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _remove_forced_signal(sst: np.ndarray) -> np.ndarray:
    """
    Remove the forced signal (ensemble mean) from SST.

    The forced signal is estimated as the ensemble mean across all members at
    each time step and grid point. Subtracting it isolates the internal
    variability component for each member.

    Parameters
    ----------
    sst : np.ndarray
        SST data, shape (nens, ntime, nlat, nlon)

    Returns
    -------
    np.ndarray
        Internal variability SST, shape (nens, ntime, nlat, nlon)
    """
    forced = np.nanmean(sst, axis=0, keepdims=True)  # (1, ntime, nlat, nlon)
    return sst - forced


def _area_mean_ensemble(
    sst: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    latmin: float,
    latmax: float,
    lonmin: float,
    lonmax: float
) -> np.ndarray:
    """
    Compute cos(lat)-weighted area mean over a lat/lon box for an ensemble.

    Parameters
    ----------
    sst : np.ndarray
        SST data, shape (nens, ntime_or_nyears, nlat, nlon)
    lat, lon : np.ndarray
        2D coordinate arrays (nlat, nlon) — lat/lon from ``load_grid_latlon``.
        Longitude must be in 0–360°E.
    latmin, latmax, lonmin, lonmax : float
        Box bounds (lon in 0–360°E)

    Returns
    -------
    np.ndarray
        Area-mean timeseries, shape (nens, ntime_or_nyears)
    """
    # 2D spatial mask for the box
    mask = (lat >= latmin) & (lat <= latmax) & (lon >= lonmin) & (lon <= lonmax)
    # (nlat, nlon) boolean

    # Cos(lat) weights — zero outside the box
    wlat = np.where(mask, np.cos(np.deg2rad(lat)), 0.0)  # (nlat, nlon)
    wsum = np.nansum(wlat)

    # Weighted sum over spatial dims (axis 2 and 3), normalised by weight sum
    # sst: (nens, ntime, nlat, nlon); wlat: (nlat, nlon)
    return np.nansum(sst * wlat[None, None, :, :], axis=(2, 3)) / wsum


def _monthly_anoms_ensemble(
    ts: np.ndarray,
    years: np.ndarray,
    baseline: Tuple[int, int] = (1990, 2020)
) -> np.ndarray:
    """
    Remove the baseline-period climatology from an ensemble data array.

    Axis 0 is ensemble members; axis 1 is the time dimension indexed by
    `years`. Works for any trailing shape — e.g. 2D (nens, nyears) for a
    1-D time series or 4D (nens, nyears, nlat, nlon) for a gridded field.

    When computing the IPO per-month loop, this is called on a single
    calendar month's data (nens, nyears, nlat, nlon), where `years` is the
    annual array np.arange(start_year, end_year+1).

    Parameters
    ----------
    ts : np.ndarray
        Data with shape (nens, nyears, ...).
    years : np.ndarray
        Year value for each step along axis 1, shape (nyears,).
    baseline : tuple, optional
        (start_year, end_year) inclusive (default: 1990-2020)

    Returns
    -------
    np.ndarray
        Anomalies relative to baseline mean, same shape as ts.
    """
    mask_base = (years >= baseline[0]) & (years <= baseline[1])
    clim_m = np.nanmean(ts[:, mask_base], axis=1, keepdims=True)
    return ts - clim_m



def _normalize_by_baseline_std_ensemble(
    x: np.ndarray,
    years: np.ndarray,
    baseline: Tuple[int, int] = (1990, 2020)
) -> np.ndarray:
    """
    Normalize each ensemble member by its baseline-period standard deviation.

    Parameters
    ----------
    x : np.ndarray
        Timeseries, shape (nens, ntime)
    years : np.ndarray
        Year for each time step, shape (ntime,)
    baseline : tuple, optional
        (start_year, end_year) inclusive (default: 1990-2020)

    Returns
    -------
    np.ndarray
        Normalized timeseries, shape (nens, ntime)
    """
    mask = (years >= baseline[0]) & (years <= baseline[1])
    sigma = np.nanstd(x[:, mask], axis=1, keepdims=True)   # (nens, 1)
    return x / sigma


def chebyshev_lowpass(
    ts: np.ndarray,
    cutoff_years: float = 13.0,
    order: int = 4,
    rp: float = 0.05
) -> np.ndarray:
    """
    Apply a Chebyshev Type-I low-pass filter to a monthly timeseries.

    Follows Henley et al. (2015): 13-year cutoff. Operates on the last axis,
    so it handles both 1D (ntime,) and 2D (nens, ntime) inputs.

    Parameters
    ----------
    ts : np.ndarray
        Monthly timeseries, shape (ntime,) or (nens, ntime)
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
    f_sampling = 12.0           # monthly data
    f_nyquist  = f_sampling / 2.0
    f_cutoff   = 1.0 / cutoff_years
    fc_norm    = f_cutoff / f_nyquist

    b, a = cheby1(order, rp, fc_norm, btype='low')
    return filtfilt(b, a, ts, axis=-1)



def enso_phase_labels_ensemble(
    index_arr: np.ndarray,
    threshold: float = 0.4,
    min_length: int = 6
) -> np.ndarray:
    """
    Label each time step for each ensemble member as El Niño (+1), La Niña (-1),
    or Neutral (0).

    Parameters
    ----------
    index_arr : np.ndarray
        Normalized ENSO index, shape (nens, ntime)
    threshold : float, optional
        Standard deviation threshold (default: 0.4)
    min_length : int, optional
        Minimum consecutive months to qualify as an event (default: 6).
        Set to 1 to disable minimum-length requirement.

    Returns
    -------
    np.ndarray
        Integer labels: +1 (El Niño), -1 (La Niña), 0 (Neutral), shape (nens, ntime)
    """
    nens, ntime = index_arr.shape
    labels = np.zeros((nens, ntime), dtype=int)

    for m in range(nens):
        ts = index_arr[m, :]

        if min_length <= 1:
            labels[m, ts >= threshold]  =  1
            labels[m, ts <= -threshold] = -1
            continue

        # El Niño: consecutive months above threshold
        mask_pos = ts >= threshold
        i = 0
        while i < ntime:
            if mask_pos[i]:
                j = i
                while j < ntime and mask_pos[j]:
                    j += 1
                if j - i >= min_length:
                    labels[m, i:j] = 1
                i = j
            else:
                i += 1

        # La Niña: consecutive months below -threshold
        mask_neg = ts <= -threshold
        i = 0
        while i < ntime:
            if mask_neg[i]:
                j = i
                while j < ntime and mask_neg[j]:
                    j += 1
                if j - i >= min_length:
                    labels[m, i:j] = -1
                i = j
            else:
                i += 1

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────


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


def _load_month_sst_all_members(
    sst_dir: Path,
    month_label: str,
    member_groups: List[str] = ['first50', 'last50'],
) -> np.ndarray:
    """
    Load the SST field for all ensemble members for a single calendar month.

    Reads the pre-separated monthly files:
        sst_cesmle_{group}members_mon_{MONTH}_199001-210012.nc

    The variable name inside each file must be ``sst_mon``.

    Parameters
    ----------
    sst_dir : Path
        Directory containing the monthly SST files.
    month_label : str
        Three-letter calendar month, e.g. ``'JAN'``, ``'SEP'``.
    member_groups : list of str, optional
        Group names to concatenate (default: ``['first50', 'last50']``).

    Returns
    -------
    sst : np.ndarray, shape (nens, nyears, nlat, nlon)
    """
    sst_dir = Path(sst_dir)
    arrays  = []
    for group in member_groups:
        fpath = sst_dir / f'sst_cesmle_{group}members_mon_{month_label}_199001-210012.nc'
        if not fpath.exists():
            raise FileNotFoundError(f'Monthly SST file not found: {fpath}')
        ds   = nc.Dataset(str(fpath), 'r')
        ds.set_auto_mask(False)
        data = ds.variables['sst_mon'][:].astype(np.float32)   # (n_group, nyears, nlat, nlon)
        ds.close()
        arrays.append(data)
    return np.concatenate(arrays, axis=0)                       # (nens, nyears, nlat, nlon)



# ─────────────────────────────────────────────────────────────────────────────
# ENSO: Niño3.4 index
# ─────────────────────────────────────────────────────────────────────────────

def compute_nino34_index(
    sst_dir: str,
    lat: np.ndarray,
    lon: np.ndarray,
    years: np.ndarray,
    baseline: Tuple[int, int] = (1990, 2020),
    smooth_months: int = 5,
    threshold: float = 0.4,
    min_length: int = 6,
    remove_forced: bool = True,
    member_groups: List[str] = ['first50', 'last50'],
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Niño3.4 ENSO index for the CESM2-LE ensemble.

    Niño3.4 box: 5S–5N, 170W–120W (190–240°E).

    Processing (per calendar month, in order):
        1. Load monthly SST file → (nens, nyears, nlat, nlon)
        2a. Remove forced signal (ensemble mean)
        2b+c. Subset Niño3.4 box + cos(lat)-weighted area mean → (nens, nyears)
        2d. Monthly anomalies: subtract baseline mean for that calendar month
       Append per-month result → list of 12 × (nens, nyears)
       Stack  → (12, nens, nyears)
       Reshape → (nens, ntime)  [chronological: 12 months of yr1, yr2, …]
       5-month running mean
       Normalize by baseline-period standard deviation
       Phase labels

    Parameters
    ----------
    sst_dir : str or Path
        Directory containing the monthly SST files.
    lat, lon : np.ndarray
        2D arrays from ``load_grid_latlon(grid_file)``.
    years : np.ndarray
        Annual year values, shape (nyears,), e.g. np.arange(1990, 2101).
    baseline : tuple, optional
        Baseline period for anomaly and normalization (default: 1990-2020).
    smooth_months : int, optional
        Running mean window in months (default: 5).
    threshold : float, optional
        Normalized threshold for phase labeling (default: 0.4).
    min_length : int, optional
        Minimum consecutive months for an ENSO event (default: 6).
    remove_forced : bool, optional
        Subtract ensemble mean before computing (default: True).
    member_groups : list of str, optional
        SST file group names (default: ['first50', 'last50']).
    verbose : bool, optional
        Print progress (default: True).

    Returns
    -------
    nino34 : np.ndarray
        Normalized Niño3.4 index, shape (nens, nyear, 12)
    nino34_flat : np.ndarray
        Normalized Niño3.4 index, shape (nens, ntime)  [chronological]
    labels : np.ndarray
        Phase labels (+1, 0, -1), shape (nens, nyear, 12)
    labels_flat : np.ndarray
        Phase labels, shape (nens, ntime)

    Examples
    --------
    >>> lat, lon = load_grid_latlon(grid_file)   # 2D 
    >>> years    = np.arange(1990, 2101)
    >>> nino34, nino34_flat, labels, labels_flat = compute_nino34_index(
    ...     sst_dir, lat, lon, years)
    >>> nino34_jja = nino34[:, :, 5:8]   # June–August subset
    """
    sst_dir = Path(sst_dir)
    nyear   = len(years)
    ntime   = nyear * 12

    # Chronological years_flat: [y0,y0,...x12, y1,y1,...x12, ...]
    # np.tile(years,(12,1)) → (12,nyear); .transpose(1,0) → (nyear,12); .reshape → (ntime,)
    years_flat = np.tile(years, (12, 1)).transpose(1, 0).reshape(ntime)

    nino34_monthly = []   # will accumulate (nens, nyears) per calendar month

    for m_label in MONTH_LABELS:
        if verbose:
            print(f'  Niño3.4: processing {m_label} ...', end='\r')

        # 1. Load all members for this calendar month
        sst_m = _load_month_sst_all_members(sst_dir, m_label, member_groups)
        # sst_m: (nens, nyears, nlat, nlon)

        # 2a. Remove forced signal
        sst_int = _remove_forced_signal(sst_m) if remove_forced else sst_m

        # 2b+c. Subset Niño3.4 box + cos(lat)-weighted area mean → (nens, nyears)
        ts_m = _area_mean_ensemble(sst_int, lat, lon,
                                   latmin=-5, latmax=5, lonmin=190, lonmax=240)

        # 2d. Monthly anomaly: subtract baseline mean for this calendar month
        anom_m = _monthly_anoms_ensemble(ts_m, years, baseline=baseline)

        nino34_monthly.append(anom_m)   # (nens, nyears)

    if verbose:
        print('  Niño3.4: all months done.           ')

    # 3. Stack → (12, nens, nyears)
    nino34_stack = np.array(nino34_monthly)   # (12, nens, nyears)

    # 4. Reshape to (nens, ntime) chronologically
    #    transpose(1,2,0) → (nens, nyears, 12) → reshape → (nens, ntime)
    #    time runs as: [Jan_y0, Feb_y0, ..., Dec_y0, Jan_y1, ..., Dec_yN]
    nens             = nino34_stack.shape[1]
    nino34_transpose = nino34_stack.transpose(1, 2, 0)          # (nens, nyears, 12)
    nino34_flat      = nino34_transpose.reshape(nens, ntime)    # (nens, ntime)

    # 5. 5-month running mean
    nino34_flat = uniform_filter1d(nino34_flat, size=smooth_months,
                                   axis=1, mode='nearest')

    # 6. Normalize by baseline-period standard deviation
    nino34_flat = _normalize_by_baseline_std_ensemble(nino34_flat,
                                                      years_flat, baseline=baseline)

    # Phase labels
    labels_flat = enso_phase_labels_ensemble(nino34_flat,
                                             threshold=threshold,
                                             min_length=min_length)

    # Reshape to (nens, nyear, 12) for convenience
    nino34 = nino34_flat.reshape(nens, nyear, 12)
    labels = labels_flat.reshape(nens, nyear, 12)

    return nino34, nino34_flat, labels, labels_flat


# ─────────────────────────────────────────────────────────────────────────────
# ENSO CP/TP: Niño3, Niño4, N_CT, N_WP
# ─────────────────────────────────────────────────────────────────────────────

def compute_enso_cp_tp_indices(
    sst_dir: str,
    lat: np.ndarray,
    lon: np.ndarray,
    years: np.ndarray,
    baseline: Tuple[int, int] = (1990, 2020),
    smooth_months: int = 5,
    threshold: float = 0.4,
    remove_forced: bool = True,
    member_groups: List[str] = ['first50', 'last50'],
    verbose: bool = True,
) -> dict:
    """
    Compute Niño3, Niño4, Niño3.4, cold-tongue (N_CT), and warm-pool (N_WP)
    indices for the CESM2-LE ensemble.

    CT/WP split (Takahashi et al.):
        N_CT = N3 − α × N4
        N_WP = N4 − α × N3
        α = 2/5  if N3 × N4 > 0,  else 0
    (α is computed from the raw area means before smoothing/normalization.)

    Processing (per calendar month, in order):
        1. Load monthly SST file → (nens, nyears, nlat, nlon)
        2a. Remove forced signal (ensemble mean)
        2b+c. Subset Niño box + cos(lat)-weighted area mean → (nens, nyears)
              (done for N34, N3, N4 simultaneously)
        2d. Monthly anomalies for each box time series
       Append per-month result → list of 12 × (nens, nyears) for each box
       Stack + reshape → (nens, ntime)  [chronological]
       CT/WP split using raw N3/N4 flat time series
       5-month running mean for all five indices
       Normalize by baseline-period standard deviation
       Phase labels

    Parameters
    ----------
    sst_dir : str or Path
        Directory containing the monthly SST files.
    lat, lon : np.ndarray
        1D coordinate arrays (nlat,) and (nlon,), lon in 0–360°E.
        2D lat/lon arrays from ``load_grid_latlon(grid_file)``.
        Pass them directly:
            lat, lon = load_grid_latlon(grid_file)
    years : np.ndarray
        Annual year values, shape (nyears,), e.g. np.arange(1990, 2101).
    baseline : tuple, optional
        Baseline period (default: 1990-2020).
    smooth_months : int, optional
        Running mean window (default: 5).
    threshold : float, optional
        Threshold for phase labeling (default: 0.4).
    remove_forced : bool, optional
        Subtract ensemble mean (default: True).
    member_groups : list of str, optional
        SST file group names (default: ['first50', 'last50']).
    verbose : bool, optional
        Print progress (default: True).

    Returns
    -------
    dict with keys (reshaped arrays, shape (nens, nyear, 12)):
        'n34', 'n3', 'n4', 'n_ct', 'n_wp'
        'labels_n34', 'labels_n3', 'labels_n4', 'labels_n_ct', 'labels_n_wp'
    And flat arrays (shape (nens, ntime)) with '_flat' suffix.

    Examples
    --------
    >>> lat, lon = load_grid_latlon(grid_file)   # 2D lat/lon
    >>> years    = np.arange(1990, 2101)
    >>> result   = compute_enso_cp_tp_indices(sst_dir, lat, lon, years)
    >>> nino_ct  = result['n_ct']          # (nens, nyear, 12)
    >>> labels   = result['labels_n_ct']
    """
    sst_dir = Path(sst_dir)
    nyear   = len(years)
    ntime   = nyear * 12

    # Chronological years_flat: [y0,y0,...x12, y1,y1,...x12, ...]
    years_flat = np.tile(years, (12, 1)).transpose(1, 0).reshape(ntime)

    # Boxes: key → (latmin, latmax, lonmin, lonmax)
    BOXES = {
        'n34': dict(latmin=-5, latmax=5, lonmin=190, lonmax=240),
        'n3':  dict(latmin=-5, latmax=5, lonmin=210, lonmax=270),
        'n4':  dict(latmin=-5, latmax=5, lonmin=160, lonmax=210),
    }

    monthly = {k: [] for k in BOXES}   # each: list of (nens, nyears)

    for m_label in MONTH_LABELS:
        if verbose:
            print(f'  ENSO CP/TP: processing {m_label} ...', end='\r')

        # 1. Load all members for this calendar month
        sst_m = _load_month_sst_all_members(sst_dir, m_label, member_groups)
        # sst_m: (nens, nyears, nlat, nlon)

        # 2a. Remove forced signal
        sst_int = _remove_forced_signal(sst_m) if remove_forced else sst_m

        # 2b+c+d. Area mean + monthly anomaly for each Niño box
        for key, box in BOXES.items():
            ts_m   = _area_mean_ensemble(sst_int, lat, lon, **box)   # (nens, nyears)
            anom_m = _monthly_anoms_ensemble(ts_m, years, baseline=baseline)
            monthly[key].append(anom_m)                              # (nens, nyears)

    if verbose:
        print('  ENSO CP/TP: all months done.           ')

    # 3. Stack → (12, nens, nyears) and reshape → (nens, ntime) for each box
    nens = monthly['n34'][0].shape[0]
    flat = {}
    for key in BOXES:
        arr       = np.array(monthly[key])                         # (12, nens, nyears)
        flat[key] = arr.transpose(1, 2, 0).reshape(nens, ntime)   # (nens, ntime)

    n34_flat = flat['n34']
    n3_flat  = flat['n3']
    n4_flat  = flat['n4']

    # CT/WP split on the raw (pre-smooth) time series
    alpha     = np.where(n3_flat * n4_flat > 0, 2.0 / 5.0, 0.0)
    n_ct_flat = n3_flat - alpha * n4_flat
    n_wp_flat = n4_flat - alpha * n3_flat

    # 5-month running mean for all five indices
    n34_flat, n3_flat, n4_flat, n_ct_flat, n_wp_flat = [
        uniform_filter1d(x, size=smooth_months, axis=1, mode='nearest')
        for x in (n34_flat, n3_flat, n4_flat, n_ct_flat, n_wp_flat)
    ]

    # Normalize by baseline-period standard deviation
    n34_flat, n3_flat, n4_flat, n_ct_flat, n_wp_flat = [
        _normalize_by_baseline_std_ensemble(x, years_flat, baseline=baseline)
        for x in (n34_flat, n3_flat, n4_flat, n_ct_flat, n_wp_flat)
    ]

    # Phase labels (no min_length for CT/WP)
    def _lbl(x):
        return enso_phase_labels_ensemble(x, threshold=threshold, min_length=1)

    out = {}
    for key, arr_flat in [('n34', n34_flat), ('n3', n3_flat), ('n4', n4_flat),
                           ('n_ct', n_ct_flat), ('n_wp', n_wp_flat)]:
        lbl = _lbl(arr_flat)
        out[f'{key}_flat']        = arr_flat
        out[f'labels_{key}_flat'] = lbl
        out[key]                  = arr_flat.reshape(nens, nyear, 12)
        out[f'labels_{key}']      = lbl.reshape(nens, nyear, 12)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# IPO: Interdecadal Pacific Oscillation
# ─────────────────────────────────────────────────────────────────────────────

def compute_ipo_index(
    sst_dir: str,
    lat: np.ndarray,
    lon: np.ndarray,
    years: np.ndarray,
    baseline: Tuple[int, int] = (1990, 2020),
    filter_cutoff_years: float = 13.0,
    member_groups: List[str] = ['first50', 'last50'],
    remove_forced: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray,
           np.ndarray, np.ndarray,
           np.ndarray, np.ndarray,
           np.ndarray, np.ndarray]:
    """
    Compute the Interdecadal Pacific Oscillation (IPO) index for CESM2-LE.

    IPO = Tropical Pacific SSTa − 0.5 × (N.Pacific SSTa + S.Pacific SSTa)
    following Henley et al. (2015).

    Regions:
        Tropical:  10S–10N,   170E–90W  (170–270°E)
        N.Pacific: 25N–45N,   140E–145W (140–215°E)
        S.Pacific: 50S–15S,   150E–160W (150–200°E)

    Processing (per calendar month, in order):
        1. Load monthly SST file → (nens, nyears, nlat, nlon)
        2. Remove forced signal (ensemble mean subtracted at each grid point)
        3. Monthly anomalies: subtract baseline-period mean for that month
        4. Cos(lat)-weighted area means for the three IPO boxes
        5. IPO = trop − 0.5 × (npac + spac)  →  (nens, nyears)
       Append per-month result → list of 12 × (nens, nyears)
       Stack → (12, nens, nyears)
       Transpose + reshape → (nens, ntime) chronologically
       13-year Chebyshev low-pass filter
       Phase labels (sign of filtered index)

    Parameters
    ----------
    sst_dir : str or Path
        Directory containing the monthly SST files
        (sst_cesmle_{group}members_mon_{MONTH}_199001-210012.nc).
    lat, lon : np.ndarray
        1D coordinate arrays (nlat,) and (nlon,), lon in 0–360°E.
        2D lat/lon arrays from ``load_grid_latlon(grid_file)``.
        Pass them directly:
            lat, lon = load_grid_latlon(grid_file)
    years : np.ndarray
        Annual year values, shape (nyears,), e.g. np.arange(1990, 2101).
    baseline : tuple, optional
        Baseline period for anomaly computation (default: 1990-2020).
    filter_cutoff_years : float, optional
        Chebyshev low-pass cutoff period in years (default: 13).
    member_groups : list of str, optional
        SST file group names to concatenate (default: ['first50', 'last50']).
    remove_forced : bool, optional
        Subtract ensemble mean before computing (default: True).
    verbose : bool, optional
        Print progress for each calendar month (default: True).

    Returns
    -------
    ipo : np.ndarray
        Unfiltered IPO index, shape (nens, nyear, 12)
    ipo_flat : np.ndarray
        Unfiltered IPO index, shape (nens, ntime)  [chronological]
    ipo_filtered : np.ndarray
        Low-pass filtered IPO index, shape (nens, nyear, 12)
    ipo_filtered_flat : np.ndarray
        Low-pass filtered IPO index, shape (nens, ntime)
    labels : np.ndarray
        Phase labels for unfiltered IPO (+1, 0, -1), shape (nens, nyear, 12)
    labels_flat : np.ndarray
        Phase labels for unfiltered IPO, shape (nens, ntime)
    labels_filtered : np.ndarray
        Phase labels for filtered IPO (+1, 0, -1), shape (nens, nyear, 12)
    labels_filtered_flat : np.ndarray
        Phase labels for filtered IPO, shape (nens, ntime)

    Examples
    --------
    >>> lat, lon = load_grid_latlon(grid_file)   # 2D lat/lon
    >>> years = np.arange(1990, 2101)
    >>> (ipo, ipo_f,
    ...  ipo_filt, ipo_filt_f,
    ...  lbl, lbl_f,
    ...  lbl_filt, lbl_filt_f) = compute_ipo_index(sst_dir, lat, lon, years)
    """
    sst_dir = Path(sst_dir)
    nyear   = len(years)
    ntime   = nyear * 12

    # ── Loop over the 12 calendar months ─────────────────────────────────────
    ipo_monthly = []   # will accumulate (nens, nyears) for each month

    for m_label in MONTH_LABELS:
        if verbose:
            print(f'  IPO: processing {m_label} ...', end='\r')

        # Step 1: load all members for this calendar month
        sst_m = _load_month_sst_all_members(sst_dir, m_label, member_groups)
        # sst_m: (nens, nyears, nlat, nlon)

        # Step 2: remove forced signal (ensemble mean at each grid point & year)
        sst_int = _remove_forced_signal(sst_m) if remove_forced else sst_m

        # Step 3: monthly anomalies — subtract baseline mean for this month
        sst_anom = _monthly_anoms_ensemble(sst_int, years, baseline=baseline)

        # Step 4: cos(lat)-weighted area means for the three IPO boxes
        trop = _area_mean_ensemble(sst_anom, lat, lon, -10,  10,  170, 270)
        npac = _area_mean_ensemble(sst_anom, lat, lon,  25,  45,  140, 215)
        spac = _area_mean_ensemble(sst_anom, lat, lon, -50, -15,  150, 200)

        # Step 5: IPO for this calendar month → (nens, nyears)
        ipo_monthly.append(trop - 0.5 * (npac + spac))

    if verbose:
        print('  IPO: all months done.           ')

    # ── Reshape to chronological (nens, ntime) ───────────────────────────────
    # Stack → (12, nens, nyears)
    ipo_stack = np.array(ipo_monthly)   # (12, nens, nyears)

    # transpose(1,2,0) → (nens, nyears, 12) → reshape → (nens, ntime)
    # time runs as: [Jan_y0, Feb_y0, ..., Dec_y0, Jan_y1, ..., Dec_yN]
    nens     = ipo_stack.shape[1]
    ipo_flat = ipo_stack.transpose(1, 2, 0).reshape(nens, ntime)

    # ── 13-year Chebyshev low-pass filter ────────────────────────────────────
    ipo_filtered_flat = chebyshev_lowpass(ipo_flat, cutoff_years=filter_cutoff_years)

    # ── Phase labels (sign-based, Henley et al. 2015) ────────────────────────
    labels_flat          = np.sign(ipo_flat).astype(int)
    labels_filtered_flat = np.sign(ipo_filtered_flat).astype(int)

    # ── Reshape back to (nens, nyear, 12) for convenience ───────────────────
    ipo             = ipo_flat.reshape(nens, nyear, 12)
    ipo_filtered    = ipo_filtered_flat.reshape(nens, nyear, 12)
    labels          = labels_flat.reshape(nens, nyear, 12)
    labels_filtered = labels_filtered_flat.reshape(nens, nyear, 12)

    return (ipo, ipo_flat,
            ipo_filtered, ipo_filtered_flat,
            labels, labels_flat,
            labels_filtered, labels_filtered_flat)


# ─────────────────────────────────────────────────────────────────────────────
# Arctic SST Index
# ─────────────────────────────────────────────────────────────────────────────

def compute_arctic_sst_forced_em(
    sst_dir: str,
    lat: np.ndarray,
    lon: np.ndarray,
    years: np.ndarray,
    member_groups: List[str] = ['first50', 'last50'],
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute the CESM2 ensemble-mean forced Arctic SST time series.

    This helper:
      1. loads CESM2 SST separately for each calendar month
      2. computes the Arctic (>60N, all longitudes) area mean for each member
      3. computes the ensemble mean of that Arctic area-mean series
      4. reshapes to a chronological monthly time series

    Parameters
    ----------
    sst_dir : str or Path
        Directory containing monthly CESM2 SST files:
        sst_cesmle_{group}members_mon_{MONTH}_199001-210012.nc
    lat, lon : np.ndarray
        2D latitude/longitude arrays on the CESM2 grid
    years : np.ndarray
        Annual year values, shape (nyears,)
    member_groups : list of str, optional
        Ensemble-member groups to concatenate
    verbose : bool, optional
        Print progress

    Returns
    -------
    forced_em_flat : np.ndarray
        Ensemble-mean Arctic SST time series, shape (ntime,)
        with chronological ordering:
        [Jan_y0, Feb_y0, ..., Dec_y0, Jan_y1, ..., Dec_yN]
    """
    sst_dir = Path(sst_dir)
    nyear = len(years)
    ntime = nyear * 12

    forced_monthly = []  # each entry: (nyears,)

    for m_label in MONTH_LABELS:
        if verbose:
            print(f'  Arctic forced EM: processing {m_label} ...', end='\r')

        # Load all members for this calendar month
        sst_m = _load_month_sst_all_members(sst_dir, m_label, member_groups)
        # sst_m: (nens, nyears, nlat, nlon)

        # Arctic area mean for each member
        ts_m = _area_mean_ensemble(
            sst_m,
            lat,
            lon,
            latmin=60,
            latmax=90,
            lonmin=0,
            lonmax=360,
        )
        # ts_m: (nens, nyears)

        # Ensemble mean of the Arctic area mean
        forced_em_m = np.nanmean(ts_m, axis=0)
        # forced_em_m: (nyears,)

        forced_monthly.append(forced_em_m)

    if verbose:
        print('  Arctic forced EM: all months done.           ')

    # Stack -> (12, nyears)
    forced_stack = np.array(forced_monthly)

    # Reshape to chronological monthly series -> (ntime,)
    forced_em_flat = forced_stack.transpose(1, 0).reshape(ntime)

    return forced_em_flat

def _arctic_phase_labels_ensemble(
    index_arr: np.ndarray,
    threshold: float = 1.0,
) -> np.ndarray:
    """
    Label each time step for each ensemble member as positive (+1),
    negative (-1), or neutral (0) based on a +/- threshold on a normalized
    index.

    Parameters
    ----------
    index_arr : np.ndarray
        Normalized index, shape (nens, ntime).
    threshold : float, optional
        Standard-deviation threshold for labelling (default: 1.0).

    Returns
    -------
    np.ndarray
        Integer labels, shape (nens, ntime).
    """
    labels = np.zeros_like(index_arr, dtype=int)
    labels[index_arr > threshold]  =  1
    labels[index_arr < -threshold] = -1
    return labels


def compute_arctic_sst_index(
    sst_dir: str,
    lat: np.ndarray,
    lon: np.ndarray,
    years: np.ndarray,
    baseline: Tuple[int, int] = (1990, 2020),
    smooth_months: int = 5,
    threshold: float = 1.0,
    remove_forced: bool = True,
    member_groups: List[str] = ['first50', 'last50'],
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Arctic SST Index for the CESM2-LE ensemble.

    Defined as the cos(lat)-weighted area mean of SST over all longitudes
    and all latitudes north of 60N.  Processing follows the same pipeline
    as Nino3.4: per-month loading -> forced signal removal -> area mean ->
    monthly anomalies -> stack to chronological -> smooth -> normalize.

    Labels use a +/- 1 standard deviation threshold on the normalized index.

    Parameters
    ----------
    sst_dir : str or Path
        Directory containing the monthly SST files.
    lat, lon : np.ndarray
        2D arrays from ``load_grid_latlon(grid_file)``.
    years : np.ndarray
        Annual year values, shape (nyears,).
    baseline : tuple, optional
        Baseline period (default: 1990-2020).
    smooth_months : int, optional
        Running mean window in months (default: 5).
    threshold : float, optional
        Standard-deviation threshold for phase labelling (default: 1.0).
    remove_forced : bool, optional
        Subtract ensemble mean before computing (default: True).
    member_groups : list of str, optional
        SST file group names (default: ['first50', 'last50']).
    verbose : bool, optional
        Print progress (default: True).

    Returns
    -------
    arctic_sst : np.ndarray
        Normalized Arctic SST index, shape (nens, nyear, 12).
    arctic_sst_flat : np.ndarray
        Normalized Arctic SST index, shape (nens, ntime) [chronological].
    labels : np.ndarray
        Phase labels (+1 positive, -1 negative, 0 neutral), shape (nens, nyear, 12).
    labels_flat : np.ndarray
        Phase labels, shape (nens, ntime).
    """
    sst_dir = Path(sst_dir)
    nyear   = len(years)
    ntime   = nyear * 12

    # Chronological years_flat for normalize function
    years_flat = np.tile(years, (12, 1)).transpose(1, 0).reshape(ntime)

    arctic_monthly = []   # (nens, nyears) per calendar month

    for m_label in MONTH_LABELS:
        if verbose:
            print(f'  Arctic SST: processing {m_label} ...', end='\r')

        # Load all members for this calendar month
        sst_m = _load_month_sst_all_members(sst_dir, m_label, member_groups)

        # Remove forced signal
        sst_int = _remove_forced_signal(sst_m) if remove_forced else sst_m

        # Area mean: all longitudes, latitudes > 60N
        ts_m = _area_mean_ensemble(sst_int, lat, lon,
                                   latmin=60, latmax=90, lonmin=0, lonmax=360)

        # Monthly anomaly
        anom_m = _monthly_anoms_ensemble(ts_m, years, baseline=baseline)

        arctic_monthly.append(anom_m)

    if verbose:
        print('  Arctic SST: all months done.           ')

    # Stack -> (12, nens, nyears) -> transpose -> (nens, nyears, 12) -> flatten
    arctic_stack = np.array(arctic_monthly)
    nens = arctic_stack.shape[1]
    arctic_flat = arctic_stack.transpose(1, 2, 0).reshape(nens, ntime)

    # Smooth
    arctic_flat = uniform_filter1d(arctic_flat, size=smooth_months,
                                   axis=1, mode='nearest')

    # Normalize by baseline-period std
    arctic_flat = _normalize_by_baseline_std_ensemble(arctic_flat,
                                                      years_flat, baseline=baseline)

    # Phase labels
    labels_flat = _arctic_phase_labels_ensemble(arctic_flat, threshold=threshold)

    # Reshape to (nens, nyear, 12)
    arctic_sst = arctic_flat.reshape(nens, nyear, 12)
    labels     = labels_flat.reshape(nens, nyear, 12)

    return arctic_sst, arctic_flat, labels, labels_flat


# ─────────────────────────────────────────────────────────────────────────────
# Save helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_nino34(
    nino34: np.ndarray,
    labels: np.ndarray,
    years: np.ndarray,
    output_file: str,
    nino34_jja: Optional[np.ndarray] = None,
    labels_jja: Optional[np.ndarray] = None
) -> None:
    """
    Save CESM2-LE Niño3.4 index and labels to NetCDF.

    Parameters
    ----------
    nino34 : np.ndarray
        Normalized Niño3.4 index, shape (nens, nyear, nmon=12)
    labels : np.ndarray
        Phase labels, shape (nens, nyear, nmon=12)
    years : np.ndarray
        Unique years, shape (nyear,)
    output_file : str
        Output NetCDF path
    nino34_jja : np.ndarray, optional
        JJA subset, shape (nens, nyear, 3)
    labels_jja : np.ndarray, optional
        JJA labels, shape (nens, nyear, 3)
    """
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    nens, nyear, nmon = nino34.shape

    data_vars = {
        "nino34_months": (("nens", "nyr", "nm"), nino34),
        "nino_labels":   (("nens", "nyr", "nm"), labels),
    }
    coords = {
        "nens": np.arange(nens),
        "nyr":  years,
        "nm":   np.arange(nmon),
    }

    if nino34_jja is not None:
        data_vars["nino34_jja"] = (("nens", "nyr", "njja"), nino34_jja)
        coords["njja"] = np.arange(nino34_jja.shape[2])
    if labels_jja is not None:
        data_vars["nino_labels_jja"] = (("nens", "nyr", "njja"), labels_jja)
        if "njja" not in coords:
            coords["njja"] = np.arange(labels_jja.shape[2])

    ds = xr.Dataset(data_vars, coords=coords)
    ds.attrs['description'] = (
        'CESM2-LE Niño3.4 index and ENSO phase labels (internal variability)'
    )

    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(output_file, format='NETCDF4', encoding=encoding)
    print(f"✓ Saved Niño3.4 to: {output_file}")


def save_enso_cp_tp(
    result: dict,
    years: np.ndarray,
    output_file: str
) -> None:
    """
    Save CESM2-LE CP/TP ENSO indices and labels to NetCDF.

    Parameters
    ----------
    result : dict
        Output of compute_enso_cp_tp_indices (reshaped arrays, shape (nens, nyear, nmon))
    years : np.ndarray
        Unique years, shape (nyear,)
    output_file : str
        Output NetCDF path
    """
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Only save the reshaped (nens, nyear, nmon) arrays; skip flat versions
    shaped_keys = [k for k in result if not k.endswith('_flat')]
    sample = result[shaped_keys[0]]
    nens, nyear, nmon = sample.shape

    data_vars = {k: (("nens", "nyr", "nm"), result[k]) for k in shaped_keys}
    coords = {
        "nens": np.arange(nens),
        "nyr":  years,
        "nm":   np.arange(nmon),
    }

    ds = xr.Dataset(data_vars, coords=coords)
    ds.attrs['description'] = (
        'CESM2-LE Niño3/4/3.4/CT/WP indices and phase labels (internal variability)'
    )

    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(output_file, format='NETCDF4', encoding=encoding)
    print(f"✓ Saved ENSO CP/TP to: {output_file}")


def save_ipo(
    ipo: np.ndarray,
    ipo_filtered: np.ndarray,
    labels: np.ndarray,
    labels_filtered: np.ndarray,
    years: np.ndarray,
    output_file: str
) -> None:
    """
    Save CESM2-LE IPO index and labels to NetCDF.

    Parameters
    ----------
    ipo : np.ndarray
        Unfiltered IPO index, shape (nens, nyear, nmon=12)
    ipo_filtered : np.ndarray
        Low-pass filtered IPO index, shape (nens, nyear, nmon=12)
    labels : np.ndarray
        Phase labels for unfiltered IPO, shape (nens, nyear, nmon=12)
    labels_filtered : np.ndarray
        Phase labels for filtered IPO, shape (nens, nyear, nmon=12)
    years : np.ndarray
        Unique years, shape (nyear,)
    output_file : str
        Output NetCDF path
    """
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    nens, nyear, nmon = ipo.shape

    ds = xr.Dataset(
        {
            "ipo_unfiltered":   (("nens", "nyr", "nmon"), ipo),
            "ipo_filtered":     (("nens", "nyr", "nmon"), ipo_filtered),
            "labels":           (("nens", "nyr", "nmon"), labels),
            "labels_filtered":  (("nens", "nyr", "nmon"), labels_filtered),
        },
        coords={
            "nens": np.arange(nens),
            "nyr":  years,
            "nmon": np.arange(nmon),
        },
    )
    ds.attrs['description'] = (
        'CESM2-LE IPO index (Henley et al. 2015) and phase labels (internal variability)'
    )
    ds.attrs['filter'] = 'Chebyshev Type-I low-pass, 13-year cutoff'

    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(output_file, format='NETCDF4', encoding=encoding)
    print(f"✓ Saved IPO to: {output_file}")


def save_arctic_sst(
    arctic_sst: np.ndarray,
    labels: np.ndarray,
    years: np.ndarray,
    output_file: str,
    arctic_sst_jja: Optional[np.ndarray] = None,
    labels_jja: Optional[np.ndarray] = None,
) -> None:
    """
    Save CESM2-LE Arctic SST Index and labels to NetCDF.

    Parameters
    ----------
    arctic_sst : np.ndarray
        Normalized Arctic SST index, shape (nens, nyear, nmon=12).
    labels : np.ndarray
        Phase labels, shape (nens, nyear, nmon=12).
    years : np.ndarray
        Unique years, shape (nyear,).
    output_file : str
        Output NetCDF path.
    arctic_sst_jja : np.ndarray, optional
        JJA subset, shape (nens, nyear, 3).
    labels_jja : np.ndarray, optional
        JJA labels, shape (nens, nyear, 3).
    """
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    nens, nyear, nmon = arctic_sst.shape

    data_vars = {
        "arctic_sst_months": (("nens", "nyr", "nm"), arctic_sst),
        "arctic_labels":     (("nens", "nyr", "nm"), labels),
    }
    coords = {
        "nens": np.arange(nens),
        "nyr":  years,
        "nm":   np.arange(nmon),
    }

    if arctic_sst_jja is not None:
        data_vars["arctic_sst_jja"] = (("nens", "nyr", "njja"), arctic_sst_jja)
        coords["njja"] = np.arange(arctic_sst_jja.shape[2])
    if labels_jja is not None:
        data_vars["arctic_labels_jja"] = (("nens", "nyr", "njja"), labels_jja)
        if "njja" not in coords:
            coords["njja"] = np.arange(labels_jja.shape[2])

    ds = xr.Dataset(data_vars, coords=coords)
    ds.attrs['description'] = (
        'CESM2-LE Arctic SST Index (cos-lat weighted mean, >60N all lons) '
        'and phase labels (internal variability). '
        '+1 = positive/warm, -1 = negative/cold, 0 = neutral'
    )

    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(output_file, format='NETCDF4', encoding=encoding)
    print(f"✓ Saved Arctic SST to: {output_file}")


def save_arctic_sst_forced_em(
    arctic_sst_forced_em_flat: np.ndarray,
    years: np.ndarray,
    output_file: str,
) -> None:
    """
    Save the CESM2-LE ensemble-mean Arctic SST forced-response time series.

    Parameters
    ----------
    arctic_sst_forced_em_flat : np.ndarray
        Chronological monthly ensemble-mean Arctic SST time series,
        shape (ntime,), ordered as:
        [Jan_y0, Feb_y0, ..., Dec_y0, Jan_y1, ..., Dec_yN]
    years : np.ndarray
        Unique years, shape (nyear,).
    output_file : str
        Output NetCDF path.
    """
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    nyear = len(years)
    ntime = len(arctic_sst_forced_em_flat)

    if ntime != nyear * 12:
        raise ValueError(
            f'Expected arctic_sst_forced_em_flat to have length {nyear * 12} '
            f'for {nyear} years, but got {ntime}.'
        )

    arctic_sst_forced_em = arctic_sst_forced_em_flat.reshape(nyear, 12)

    ds = xr.Dataset(
        {
            "arctic_sst_forced_em":      (("nyr", "nm"), arctic_sst_forced_em),
            "arctic_sst_forced_em_flat": (("nt",), arctic_sst_forced_em_flat),
        },
        coords={
            "nyr": years,
            "nm":  np.arange(12),
            "nt":  np.arange(ntime),
        },
    )

    ds.attrs["description"] = (
        "CESM2-LE ensemble-mean Arctic SST forced-response time series "
        "(cos-lat weighted mean, >60N all longitudes). "
        "Saved both as (nyear, 12) and chronological monthly flat series."
    )

    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(output_file, format="NETCDF4", encoding=encoding)

    print(f"✓ Saved Arctic SST forced ensemble mean to: {output_file}")


# ─────────────────────────────────────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Compute CESM2-LE climate indices (Niño3.4, ENSO CP/TP, IPO)'
    )
    parser.add_argument('data_dir',   help='Directory containing monthly SST files')
    parser.add_argument('grid_file',  help='Grid NetCDF file from 01_cesm2le_grid.py')
    parser.add_argument('output_dir', help='Directory for output NetCDF files')
    parser.add_argument('--start-year', type=int, default=1990)
    parser.add_argument('--end-year',   type=int, default=2100)
    parser.add_argument('--baseline',   type=int, nargs=2, default=[1990, 2020],
                        metavar=('BASELINE_START', 'BASELINE_END'))
    args = parser.parse_args()

    baseline   = tuple(args.baseline)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    years      = np.arange(args.start_year, args.end_year + 1)

    # Load 2D lat/lon from the grid file (NOT from SST files)
    print("Loading grid lat/lon...")
    lat, lon = load_grid_latlon(args.grid_file)   # (nj, ni) each
    print(f"  lat shape: {lat.shape}, lon shape: {lon.shape}")

    # --- Niño3.4 ---
    print("\nComputing Niño3.4 index...")
    nino34, nino34_flat, labels, labels_flat = compute_nino34_index(
        args.data_dir, lat, lon, years, baseline=baseline
    )
    nino34_jja = nino34[:, :, 5:8]    # June–July–August
    labels_jja = labels[:, :, 5:8]
    save_nino34(
        nino34, labels, years,
        str(output_dir / 'cesm2le_nino34_index_labels.nc'),
        nino34_jja=nino34_jja, labels_jja=labels_jja
    )

    # --- ENSO CP/TP ---
    print("\nComputing ENSO CP/TP indices...")
    cp_tp = compute_enso_cp_tp_indices(args.data_dir, lat, lon, years, baseline=baseline)
    save_enso_cp_tp(
        cp_tp, years,
        str(output_dir / 'cesm2le_enso_cptp_indices.nc')
    )

    # --- IPO ---
    print("\nComputing IPO index...")
    (ipo, ipo_f,
     ipo_filt, ipo_filt_f,
     lbl, lbl_f,
     lbl_filt, lbl_filt_f) = compute_ipo_index(args.data_dir, lat, lon, years,
                                                baseline=baseline)
    save_ipo(
        ipo, ipo_filt, lbl, lbl_filt, years,
        str(output_dir / 'cesm2le_ipo_index_labels.nc')
    )

    print("\nDone.")
