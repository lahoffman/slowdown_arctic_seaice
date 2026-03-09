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
        SST data, shape (nens, ntime, nlat, nlon)
    lat, lon : np.ndarray
        1D coordinate arrays (lon in 0-360°E)
    latmin, latmax, lonmin, lonmax : float
        Box bounds

    Returns
    -------
    np.ndarray
        Area-mean timeseries, shape (nens, ntime)
    """
    lat_inds = np.where((lat >= latmin) & (lat <= latmax))[0]
    lon_inds = np.where((lon >= lonmin) & (lon <= lonmax))[0]
    subset = sst[:, :, lat_inds, :][:, :, :, lon_inds]   # (nens, ntime, nlat_sub, nlon_sub)

    wlat = np.cos(np.deg2rad(lat[lat_inds]))
    wlat = wlat / np.nansum(wlat)

    lat_weighted = np.nanmean(subset * wlat[None, None, :, None], axis=(2,3))  # (nens, ntime, nlon_sub)
    return lat_weighted   # (nens, ntime)


def _monthly_anoms_ensemble(
    ts: np.ndarray,
    years: np.ndarray,
    baseline: Tuple[int, int] = (1990, 2020)
) -> np.ndarray:
    """
    Remove per-member monthly climatology over a baseline period.

    Parameters
    ----------
    ts : np.ndarray
        Monthly timeseries, shape (nmonths, nens, ntime)
    years : np.ndarray
        Year for each time step, shape (ntime,). Assumes series starts in January.
    baseline : tuple, optional
        (start_year, end_year) inclusive (default: 1990-2020)

    Returns
    -------
    np.ndarray
        Anomalies, shape (nmonths, nens, ntime)
    """

    mask_base = (years >= baseline[0]) & (years <= baseline[1])
    clim_m = np.nanmean(ts[:,mask_base], axis=1, keepdims=True) 
    anoms = ts - clim_m
    return anoms

#TODO: edit this as it comes later in the code after smoothing 
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


#TODO: double check this, enso is getting nans
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


#TODO: EDIT THIS TO USE MONTHLY FILES... might not even need this function
def load_sst_monthly_files(
    data_dir: str,
    start_year: int = 1990,
    end_year: int = 2100,
    member_groups: List[str] = ['first50', 'last50'],
    file_start_year: Optional[int] = None,
    file_end_year: Optional[int] = None,
    file_pattern: Optional[str] = None,
    var_name: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load CESM2-LE SST from per-month NetCDF files and assemble into a single array.

    Loads 12 monthly files (one per calendar month), concatenates across ensemble
    member groups, and interleaves months to produce a contiguous monthly timeseries.

    Expected file naming (output of separate_by_month):
        sst_cesmle_{group}members_mon_{MONTH}_{file_start_year}01-{file_end_year}12.nc

    Parameters
    ----------
    data_dir : str
        Directory containing the monthly NetCDF files
    start_year : int, optional
        First year of the data range (used to build the years array, default: 1990)
    end_year : int, optional
        Last year of the data range (used to build the years array, default: 2100)
    member_groups : list of str, optional
        Which member groups to load and concatenate (default: ['first50', 'last50'])
    file_start_year : int, optional
        Start year embedded in the filename. Defaults to start_year if not provided.
    file_end_year : int, optional
        End year embedded in the filename. Defaults to end_year if not provided.
    file_pattern : str, optional
        Custom file pattern with {month}, {group}, {member_label}, {file_start_year},
        and {file_end_year} placeholders. Overrides the default naming convention.
    var_name : str, optional
        Variable name inside the NetCDF files. If None, auto-detected from
        ['sst_mon', 'ssthm', 'sst', 'SST'].

    Returns
    -------
    sst : np.ndarray
        SST array, shape (nens, ntime, nlat, nlon)
    lat : np.ndarray
        Latitude array, shape (nlat,)
    lon : np.ndarray
        Longitude array, shape (nlon,)
    years : np.ndarray
        Year for each time step, shape (ntime,)

    Examples
    --------
    >>> sst, lat, lon, years = load_sst_monthly_files(
    ...     '/data/cesm2le/sst/mon', start_year=1990, end_year=2100)
    >>> sst.shape  # (100, 1332, 192, 288)
    """
    data_dir = Path(data_dir)
    nyear = end_year - start_year + 1
    years = np.repeat(np.arange(start_year, end_year + 1), 12)

    # Years used in the filename may differ from the data years array
    fname_start = file_start_year if file_start_year is not None else start_year
    fname_end   = file_end_year   if file_end_year   is not None else end_year

    monthly_data = []   # list of 12 arrays, each (nens, nyear, nlat, nlon)
    lat = lon = None

    for month_label in MONTH_LABELS:
        group_arrays = []

        for group in member_groups:
            member_label = f'{group}members'
            if file_pattern is not None:
                fname = file_pattern.format(
                    month=month_label, group=group,
                    member_label=member_label,
                    file_start_year=fname_start, file_end_year=fname_end
                )
            else:
                fname = (f'sst_cesmle_{member_label}_mon_{month_label}_'
                         f'{fname_start}01-{fname_end}12.nc')

            fpath = data_dir / fname
            if not fpath.exists():
                raise FileNotFoundError(
                    f"Monthly SST file not found: {fpath}\n"
                    f"Expected pattern: sst_cesmle_{{group}}members_mon_{{MONTH}}_"
                    f"{start_year}01-{end_year}12.nc"
                )

            ds = nc.Dataset(fpath, 'r')
            ds.set_auto_mask(False)

            # Auto-detect SST variable name
            if var_name is not None:
                vname = var_name
            else:
                candidates = ['sst_mon', 'ssthm', 'sst', 'SST']
                vname = next((v for v in candidates if v in ds.variables), None)
                if vname is None:
                    skip = {'nem', 'nm', 'nx', 'ny', 'nyr', 'unique_years', 'lat', 'lon'}
                    remaining = [v for v in ds.variables if v not in skip]
                    if len(remaining) == 1:
                        vname = remaining[0]
                    else:
                        ds.close()
                        raise KeyError(
                            f"Cannot identify SST variable in {fpath}. "
                            f"Available: {list(ds.variables.keys())}"
                        )

            data_chunk = ds.variables[vname][:].astype(np.float32)  # (nens, nyear, nlat, nlon)

            # Extract lat/lon on first successful load
            if lat is None:
                lat_key = next((k for k in ('lat', 'nx') if k in ds.variables), None)
                lon_key = next((k for k in ('lon', 'ny') if k in ds.variables), None)
                if lat_key:
                    lat = np.array(ds.variables[lat_key])
                if lon_key:
                    lon = np.array(ds.variables[lon_key])

            ds.close()
            group_arrays.append(data_chunk)

        # Concatenate member groups along ensemble axis (axis=0)
        month_all = np.concatenate(group_arrays, axis=0)  # (nens, nyear, nlat, nlon)
        monthly_data.append(month_all)

    nens = monthly_data[0].shape[0]
    nlat = monthly_data[0].shape[2]
    nlon = monthly_data[0].shape[3]

    if lat is None:
        lat = np.arange(nlat, dtype=np.float32)
        print("Warning: lat coordinates not found in files; using index array.")
    if lon is None:
        lon = np.arange(nlon, dtype=np.float32)
        print("Warning: lon coordinates not found in files; using index array.")

    # Normalise longitude to 0–360°E.
    # CESM SST files store lon as −180→180; all box bounds in this module
    # use the 0–360 convention, so boxes like Niño3.4 (190–240°E) would
    # find zero grid points if lon is not converted first.
    if np.any(lon < 0):
        lon = np.where(lon < 0, lon + 360.0, lon)

    # Interleave 12 monthly arrays into contiguous monthly timeseries
    # monthly_data[m] has shape (nens, nyear, nlat, nlon)
    # Stack along a new month axis → (nens, nyear, 12, nlat, nlon)
    # then reshape → (nens, ntime, nlat, nlon)  where ntime = nyear * 12
    monthly_stack = np.stack(monthly_data, axis=2)              # (nens, nyear, 12, nlat, nlon)
    ntime = nyear * 12
    sst = monthly_stack.reshape(nens, ntime, nlat, nlon)

    return sst, lat, lon, years


# ─────────────────────────────────────────────────────────────────────────────
# ENSO: Niño3.4 index
# ─────────────────────────────────────────────────────────────────────────────

def compute_nino34_index(
    sst: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    years: np.ndarray,
    baseline: Tuple[int, int] = (1990, 2020),
    smooth_months: int = 5,
    threshold: float = 0.4,
    min_length: int = 6,
    remove_forced: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Niño3.4 ENSO index for the CESM2-LE ensemble.

    Steps: (optional) remove forced signal → area mean → monthly anomalies
           → smooth → normalize → phase labels.

    Outputs are provided in both reshaped (nens, nyear, nmon) and flat
    (nens, ntime) forms. The JJA subset can be extracted from the reshaped
    output as nino34[:, :, 5:8].

    Parameters
    ----------
    sst : np.ndarray
        SST data, shape (nens, ntime, nlat, nlon)
    lat, lon : np.ndarray
        1D coordinate arrays (lon in 0-360°E)
    years : np.ndarray
        Year for each time step, shape (ntime,)
    baseline : tuple, optional
        Baseline period for climatology and normalization (default: 1990-2020)
    smooth_months : int, optional
        Running mean window in months (default: 5)
    threshold : float, optional
        Normalized threshold for phase labeling (default: 0.4)
    min_length : int, optional
        Minimum consecutive months for an ENSO event (default: 6)
    remove_forced : bool, optional
        Whether to subtract the ensemble mean before computing (default: True)

    Returns
    -------
    nino34 : np.ndarray
        Normalized Niño3.4 index, shape (nens, nyear, nmon=12)
    nino34_flat : np.ndarray
        Normalized Niño3.4 index, shape (nens, ntime)
    labels : np.ndarray
        Phase labels (+1, 0, -1), shape (nens, nyear, nmon=12)
    labels_flat : np.ndarray
        Phase labels, shape (nens, ntime)

    Examples
    --------
    >>> nino34, nino34_flat, labels, labels_flat = compute_nino34_index(
    ...     sst, lat, lon, years)
    >>> nino34_jja    = nino34[:, :, 5:8]   # June–August subset
    >>> labels_jja    = labels[:, :, 5:8]
    """
    nens  = sst.shape[0]
    ntime = sst.shape[1]
    nyear = ntime // 12

    # Remove forced signal (ensemble mean)
    sst_int = _remove_forced_signal(sst) if remove_forced else sst

    # Area mean over Niño3.4 box (5S–5N, 170W–120W = 190–240°E)
    ts = _area_mean_ensemble(sst_int, lat, lon,
                             latmin=-5, latmax=5, lonmin=190, lonmax=240)

    # Monthly anomalies (per member, baseline climatology)
    anoms = _monthly_anoms_ensemble(ts, years, baseline=baseline)

    # 5-month running mean
    anoms = uniform_filter1d(anoms, size=smooth_months, axis=1, mode='nearest')

    # Normalize by baseline standard deviation
    nino34_flat = _normalize_by_baseline_std_ensemble(anoms, years, baseline=baseline)

    # Phase labels
    labels_flat = enso_phase_labels_ensemble(
        nino34_flat, threshold=threshold, min_length=min_length
    )

    # Reshape to (nens, nyear, nmon=12)
    nino34 = nino34_flat.reshape(nens, nyear, 12)
    labels = labels_flat.reshape(nens, nyear, 12)

    return nino34, nino34_flat, labels, labels_flat


# ─────────────────────────────────────────────────────────────────────────────
# ENSO CP/TP: Niño3, Niño4, N_CT, N_WP
# ─────────────────────────────────────────────────────────────────────────────

def compute_enso_cp_tp_indices(
    sst: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    years: np.ndarray,
    baseline: Tuple[int, int] = (1990, 2020),
    smooth_months: int = 5,
    threshold: float = 0.4,
    remove_forced: bool = True
) -> dict:
    """
    Compute Niño3, Niño4, Niño3.4, cold-tongue (N_CT), and warm-pool (N_WP)
    indices for the CESM2-LE ensemble.

    The CT/WP split follows Takahashi et al.:
        N_CT = N3 - α*N4
        N_WP = N4 - α*N3
        α = 2/5 if N3*N4 > 0, else 0

    Parameters
    ----------
    sst : np.ndarray
        SST data, shape (nens, ntime, nlat, nlon)
    lat, lon : np.ndarray
        1D coordinate arrays (lon in 0-360°E)
    years : np.ndarray
        Year for each time step, shape (ntime,)
    baseline : tuple, optional
        Baseline period (default: 1990-2020)
    smooth_months : int, optional
        Running mean window (default: 5)
    threshold : float, optional
        Threshold for phase labeling (default: 0.4)
    remove_forced : bool, optional
        Whether to subtract the ensemble mean (default: True)

    Returns
    -------
    dict with keys (reshaped arrays, shape (nens, nyear, nmon=12)):
        'n34', 'n3', 'n4', 'n_ct', 'n_wp'
        'labels_n34', 'labels_n3', 'labels_n4', 'labels_n_ct', 'labels_n_wp'
    And corresponding flat arrays (shape (nens, ntime)) with '_flat' suffix:
        'n34_flat', 'n3_flat', ..., 'labels_n34_flat', ...

    Examples
    --------
    >>> result = compute_enso_cp_tp_indices(sst, lat, lon, years)
    >>> nino_ct = result['n_ct']          # (nens, nyear, 12)
    >>> labels  = result['labels_n_ct']
    """
    nens  = sst.shape[0]
    ntime = sst.shape[1]
    nyear = ntime // 12

    sst_int = _remove_forced_signal(sst) if remove_forced else sst

    # Area means for three Niño boxes
    n34_ts = _area_mean_ensemble(sst_int, lat, lon,
                                 latmin=-5, latmax=5, lonmin=190, lonmax=240)
    n3_ts  = _area_mean_ensemble(sst_int, lat, lon,
                                 latmin=-5, latmax=5, lonmin=210, lonmax=270)
    n4_ts  = _area_mean_ensemble(sst_int, lat, lon,
                                 latmin=-5, latmax=5, lonmin=160, lonmax=210)

    # Monthly anomalies
    n34 = _monthly_anoms_ensemble(n34_ts, years, baseline=baseline)
    n3  = _monthly_anoms_ensemble(n3_ts,  years, baseline=baseline)
    n4  = _monthly_anoms_ensemble(n4_ts,  years, baseline=baseline)

    # CT / WP split: α = 2/5 when N3 and N4 have the same sign
    alpha = np.where(n3 * n4 > 0, 2.0 / 5.0, 0.0)
    n_ct = n3 - alpha * n4
    n_wp = n4 - alpha * n3

    # Smooth all five
    n34, n3, n4, n_ct, n_wp = [
        uniform_filter1d(x, size=smooth_months, axis=1, mode='nearest')
        for x in (n34, n3, n4, n_ct, n_wp)
    ]

    # Normalize by baseline std
    n34, n3, n4, n_ct, n_wp = [
        _normalize_by_baseline_std_ensemble(x, years, baseline=baseline)
        for x in (n34, n3, n4, n_ct, n_wp)
    ]

    # Phase labels (simple threshold, no min_length for CT/WP)
    def _labels(x):
        return enso_phase_labels_ensemble(x, threshold=threshold, min_length=1)

    out = {}
    for key, flat in [('n34', n34), ('n3', n3), ('n4', n4),
                      ('n_ct', n_ct), ('n_wp', n_wp)]:
        lbl = _labels(flat)
        out[f'{key}_flat']        = flat
        out[f'labels_{key}_flat'] = lbl
        out[key]                  = flat.reshape(nens, nyear, 12)
        out[f'labels_{key}']      = lbl.reshape(nens, nyear, 12)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# IPO: Interdecadal Pacific Oscillation
# ─────────────────────────────────────────────────────────────────────────────

def compute_ipo_index(
    sst: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    years: np.ndarray,
    baseline: Tuple[int, int] = (1990, 2020),
    filter_cutoff_years: float = 13.0,
    remove_forced: bool = True
) -> Tuple[np.ndarray, np.ndarray,
           np.ndarray, np.ndarray,
           np.ndarray, np.ndarray,
           np.ndarray, np.ndarray]:
    """
    Compute the Interdecadal Pacific Oscillation (IPO) index for CESM2-LE.

    IPO = Tropical Pacific SSTa - 0.5 * (N.Pacific SSTa + S.Pacific SSTa)

    Regions (Henley et al. 2015):
        Tropical:  10S–10N,  170E–90W  (170–270°E)
        N.Pacific: 25N–45N,  140E–145W (140–215°E)
        S.Pacific: 50S–15S,  150E–160W (150–200°E)

    Parameters
    ----------
    sst : np.ndarray
        SST data, shape (nens, ntime, nlat, nlon)
    lat, lon : np.ndarray
        1D coordinate arrays (lon in 0-360°E)
    years : np.ndarray
        Year for each time step, shape (ntime,)
    baseline : tuple, optional
        Baseline period for monthly climatology removal (default: 1990-2020)
    filter_cutoff_years : float, optional
        Chebyshev filter cutoff in years (default: 13)
    remove_forced : bool, optional
        Whether to subtract the ensemble mean (default: True)

    Returns
    -------
    ipo : np.ndarray
        Unfiltered IPO index, shape (nens, nyear, nmon=12)
    ipo_flat : np.ndarray
        Unfiltered IPO index, shape (nens, ntime)
    ipo_filtered : np.ndarray
        Low-pass filtered IPO index, shape (nens, nyear, nmon=12)
    ipo_filtered_flat : np.ndarray
        Low-pass filtered IPO index, shape (nens, ntime)
    labels : np.ndarray
        Phase labels for unfiltered IPO (+1, 0, -1), shape (nens, nyear, nmon=12)
    labels_flat : np.ndarray
        Phase labels for unfiltered IPO, shape (nens, ntime)
    labels_filtered : np.ndarray
        Phase labels for filtered IPO (+1, 0, -1), shape (nens, nyear, nmon=12)
    labels_filtered_flat : np.ndarray
        Phase labels for filtered IPO, shape (nens, ntime)

    Examples
    --------
    >>> (ipo, ipo_f,
    ...  ipo_filt, ipo_filt_f,
    ...  lbl, lbl_f,
    ...  lbl_filt, lbl_filt_f) = compute_ipo_index(sst, lat, lon, years)
    """
    nens  = sst.shape[0]
    ntime = sst.shape[1]
    nyear = ntime // 12

    sst_int = _remove_forced_signal(sst) if remove_forced else sst

    # Area-weighted box means
    trop_ts = _area_mean_ensemble(sst_int, lat, lon, -10,  10,  170, 270)
    npac_ts = _area_mean_ensemble(sst_int, lat, lon,  25,  45,  140, 215)
    spac_ts = _area_mean_ensemble(sst_int, lat, lon, -50, -15,  150, 200)

    # Monthly anomalies per member (removes residual seasonal cycle)
    trop = _monthly_anoms_ensemble(trop_ts, years, baseline=baseline)
    npac = _monthly_anoms_ensemble(npac_ts, years, baseline=baseline)
    spac = _monthly_anoms_ensemble(spac_ts, years, baseline=baseline)

    # IPO = trop - 0.5*(N.Pac + S.Pac)
    ipo_flat = trop - 0.5 * (npac + spac)   # (nens, ntime)

    # 13-year Chebyshev low-pass filter per ensemble member (Henley et al. 2015)
    ipo_filtered_flat = chebyshev_lowpass(ipo_flat, cutoff_years=filter_cutoff_years)

    # Phase labels (sign-based, consistent with Henley et al. 2015)
    labels_flat          = np.sign(ipo_flat).astype(int)
    labels_filtered_flat = np.sign(ipo_filtered_flat).astype(int)

    # Reshape to (nens, nyear, nmon=12)
    ipo             = ipo_flat.reshape(nens, nyear, 12)
    ipo_filtered    = ipo_filtered_flat.reshape(nens, nyear, 12)
    labels          = labels_flat.reshape(nens, nyear, 12)
    labels_filtered = labels_filtered_flat.reshape(nens, nyear, 12)

    return (ipo, ipo_flat,
            ipo_filtered, ipo_filtered_flat,
            labels, labels_flat,
            labels_filtered, labels_filtered_flat)


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


# ─────────────────────────────────────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Compute CESM2-LE climate indices (Niño3.4, ENSO CP/TP, IPO)'
    )
    parser.add_argument('data_dir',   help='Directory containing monthly SST files')
    parser.add_argument('output_dir', help='Directory for output NetCDF files')
    parser.add_argument('--start-year', type=int, default=1850)
    parser.add_argument('--end-year',   type=int, default=2100)
    parser.add_argument('--baseline',   type=int, nargs=2, default=[1990, 2020],
                        metavar=('BASELINE_START', 'BASELINE_END'))
    args = parser.parse_args()

    baseline = tuple(args.baseline)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SST monthly files...")
    sst, lat, lon, years = load_sst_monthly_files(
        args.data_dir,
        start_year=args.start_year,
        end_year=args.end_year
    )
    print(f"  SST shape: {sst.shape}  (nens, ntime, nlat, nlon)")

    unique_years = np.arange(args.start_year, args.end_year + 1)

    # --- Niño3.4 ---
    print("\nComputing Niño3.4 index...")
    nino34, nino34_flat, labels, labels_flat = compute_nino34_index(
        sst, lat, lon, years, baseline=baseline
    )
    nino34_jja = nino34[:, :, 5:8]    # June–July–August
    labels_jja = labels[:, :, 5:8]
    save_nino34(
        nino34, labels, unique_years,
        str(output_dir / 'cesm2le_nino34_index_labels.nc'),
        nino34_jja=nino34_jja, labels_jja=labels_jja
    )

    # --- ENSO CP/TP ---
    print("\nComputing ENSO CP/TP indices...")
    cp_tp = compute_enso_cp_tp_indices(sst, lat, lon, years, baseline=baseline)
    save_enso_cp_tp(
        cp_tp, unique_years,
        str(output_dir / 'cesm2le_enso_cptp_indices.nc')
    )

    # --- IPO ---
    print("\nComputing IPO index...")
    (ipo, ipo_f,
     ipo_filt, ipo_filt_f,
     lbl, lbl_f,
     lbl_filt, lbl_filt_f) = compute_ipo_index(sst, lat, lon, years, baseline=baseline)
    save_ipo(
        ipo, ipo_filt, lbl, lbl_filt, unique_years,
        str(output_dir / 'cesm2le_ipo_index_labels.nc')
    )

    print("\nDone.")
