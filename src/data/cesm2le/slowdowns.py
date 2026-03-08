#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CESM2-LE Slowdown and RILES Detection

Classify 10-year sliding decadal trends in CESM2-LE sea ice extent (SIE) or
sea ice area (SIA) as "slowdown" or "RILES" events, using thresholds derived
from NSIDC observations (output of scripts/01_slowdown_nsidc_sie_sia.py).

Approach
--------
1. Load NSIDC-derived slowdown/riles fraction thresholds  (all 12 months).
2. Load CESM2-LE SIE or SIA for one calendar month.
3. Compute 10-year sliding linear trends for the ensemble mean and each member.
4. Build a time-varying model threshold:

       model_threshold(t) = fraction_nsidc × ensemble_mean_trend(t)

   The fraction encodes *how far above/below the mean* the NSIDC threshold sits.
   Multiplying by the model's own ensemble-mean trend scales it to the model's
   forced sea-ice decline magnitude, which changes over time.

5. Classify each member's trend window as:
   - Slowdown : trend > threshold_slowdown  (anomalously slow ice loss)
   - RILES    : trend < threshold_riles      (anomalously rapid ice loss)
6. Save results to NetCDF.

Reference: x_old/data_processing/D1_MODEL_slowdown_riles_1850-2100.py

Author: Lauren Hoffman
Email: lhoffma2@ucsc.edu
"""

import numpy as np
import xarray as xr
import netCDF4 as nc
from pathlib import Path
from typing import Optional, Tuple, List


MONTH_LABELS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']


# ─────────────────────────────────────────────────────────────────────────────
# Load NSIDC thresholds
# ─────────────────────────────────────────────────────────────────────────────

def load_nsidc_slowdown_thresholds(threshold_file: str) -> dict:
    """
    Load NSIDC-derived slowdown and riles thresholds from NetCDF.

    These files are produced by scripts/01_slowdown_nsidc_sie_sia.py and saved
    at paths.NSIDC_SIE_SLOWDOWN_THRESHOLDS / paths.NSIDC_SIA_SLOWDOWN_THRESHOLDS.

    Parameters
    ----------
    threshold_file : str
        Path to nsidc_sie_slowdown_thresholds.nc or nsidc_sia_slowdown_thresholds.nc.
        Must contain variables: threshold_slowdown, fraction_slowdown,
        threshold_riles, fraction_riles — all shape (12,), indexed by calendar month.

    Returns
    -------
    dict with keys:
        'threshold_slowdown' : np.ndarray, shape (12,)  — absolute threshold, M km² yr⁻¹
        'fraction_slowdown'  : np.ndarray, shape (12,)  — threshold / mean_trend
        'threshold_riles'    : np.ndarray, shape (12,)  — absolute threshold, M km² yr⁻¹
        'fraction_riles'     : np.ndarray, shape (12,)  — threshold / mean_trend

    Examples
    --------
    >>> from configs import paths
    >>> thr = load_nsidc_slowdown_thresholds(str(paths.NSIDC_SIE_SLOWDOWN_THRESHOLDS))
    >>> thr['fraction_slowdown']   # shape (12,)
    """
    ds = xr.open_dataset(threshold_file)
    out = {
        'threshold_slowdown': ds['threshold_slowdown'].values,
        'fraction_slowdown':  ds['fraction_slowdown'].values,
        'threshold_riles':    ds['threshold_riles'].values,
        'fraction_riles':     ds['fraction_riles'].values,
    }
    ds.close()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Load CESM2-LE SIE / SIA per-month files
# ─────────────────────────────────────────────────────────────────────────────

def load_sie_monthly_files(
    data_dir: str,
    month: str,
    member_groups: List[str] = ['first50', 'last50'],
    variable: str = 'sie',
    start_year: int = 1850,
    end_year: int = 2100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load CESM2-LE sea ice extent or area for a single calendar month.

    Reads the per-month files produced by metrics.batch_process_monthly_files
    and concatenates across ensemble member groups.

    File naming  : siextentn_{member_label}_{MONTH}.nc  (SIE)
                   siarean_{member_label}_{MONTH}.nc     (SIA)
    Variable name: siextentn  (SIE) / siarean  (SIA)

    Parameters
    ----------
    data_dir : str
        Directory containing the monthly SIE/SIA files
    month : str
        Calendar month label, e.g. 'SEP'
    member_groups : list of str, optional
        Ensemble groups to concatenate (default: ['first50', 'last50'])
    variable : str, optional
        'sie' for sea ice extent (default) or 'sia' for sea ice area
    start_year : int, optional
        First year in the dataset (default: 1850)
    end_year : int, optional
        Last year in the dataset (default: 2100)

    Returns
    -------
    sie : np.ndarray
        SIE or SIA timeseries, shape (nens, nyear), units: million km²
    years : np.ndarray
        Year labels, shape (nyear,)

    Examples
    --------
    >>> sie, years = load_sie_monthly_files('/data/cesm2le/sie', 'SEP')
    >>> sie.shape   # (100, 251)  for 100 members, 1850-2100
    """
    data_dir = Path(data_dir)
    years = np.arange(start_year, end_year + 1)

    prefix = 'siextentn' if variable == 'sie' else 'siarean'
    group_arrays = []

    for group in member_groups:
        member_label = f'{group}members'
        fpath = data_dir / f'{prefix}_{member_label}_{month}.nc'

        if not fpath.exists():
            raise FileNotFoundError(f"{variable.upper()} file not found: {fpath}")

        ds = nc.Dataset(fpath, 'r')
        ds.set_auto_mask(False)
        chunk = ds.variables[prefix][:].astype(np.float32)   # (nens, nyear)
        ds.close()
        group_arrays.append(chunk)

    sie = np.concatenate(group_arrays, axis=0)   # (nens, nyear)
    return sie, years


# ─────────────────────────────────────────────────────────────────────────────
# Decadal trend computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_decadal_trends_ensemble(
    sie: np.ndarray,
    years: np.ndarray,
    window: int = 10,
    start_year: int = 1990
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute sliding *window*-year linear trends for each ensemble member and
    for the ensemble mean.

    For each starting year-index j, a linear regression is fit over the window
    [j, j+window). The slope (M km² yr⁻¹) is the decadal trend. Only windows
    whose starting year is ≥ start_year are returned.

    Computation is vectorised across ensemble members for efficiency.

    Parameters
    ----------
    sie : np.ndarray
        SIE or SIA, shape (nens, nyear), units: million km²
    years : np.ndarray
        Year labels, shape (nyear,)
    window : int, optional
        Trend window length in years (default: 10)
    start_year : int, optional
        First starting year to include (default: 1990)

    Returns
    -------
    trends_ens : np.ndarray
        Per-member trends, shape (nens, n_trends), M km² yr⁻¹
    trends_mean : np.ndarray
        Ensemble-mean trends, shape (n_trends,), M km² yr⁻¹
    trend_years : np.ndarray
        Starting year of each trend window, shape (n_trends,)

    Examples
    --------
    >>> trends_ens, trends_mean, trend_years = compute_decadal_trends_ensemble(
    ...     sie, years, window=10, start_year=1990)
    >>> trends_ens.shape    # (100, 101)  for 1990-2100
    """
    nens, ny = sie.shape
    n_windows = ny - window

    start_idx = int(np.where(years == start_year)[0][0])

    # Pre-compute OLS coefficients analytically for speed
    # For x = [0, 1, ..., w-1]:  slope = Σ(xi - x̄)(yi - ȳ) / Σ(xi - x̄)²
    dx = np.arange(window, dtype=np.float64)
    dx_bar = dx.mean()
    dx_dev = dx - dx_bar                         # (window,)
    dx_var = (dx_dev ** 2).sum()                 # scalar

    all_slopes_ens  = np.full((nens, n_windows), np.nan)
    all_slopes_mean = np.full(n_windows, np.nan)

    sie_mean = np.nanmean(sie, axis=0)           # (nyear,)

    for j in range(n_windows):
        # --- Ensemble members (vectorised) ---
        dy = sie[:, j : j + window].astype(np.float64)   # (nens, window)
        # Only compute slope where all window values are non-NaN
        valid_rows = ~np.any(np.isnan(dy), axis=1)        # (nens,)
        if valid_rows.any():
            dy_valid = dy[valid_rows, :]                  # (k, window)
            dy_bar   = dy_valid.mean(axis=1, keepdims=True)
            slopes   = (dx_dev[None, :] * (dy_valid - dy_bar)).sum(axis=1) / dx_var
            all_slopes_ens[valid_rows, j] = slopes

        # --- Ensemble mean ---
        dy_m = sie_mean[j : j + window].astype(np.float64)
        valid_m = ~np.isnan(dy_m)
        if valid_m.sum() >= 2:
            # Fall back to polyfit for NaN-aware mean trend
            coeffs = np.polyfit(dx[valid_m], dy_m[valid_m], 1)
            all_slopes_mean[j] = coeffs[0]

    # Trim to start_year
    trends_ens  = all_slopes_ens[:, start_idx:]   # (nens, n_trends)
    trends_mean = all_slopes_mean[start_idx:]      # (n_trends,)
    trend_years = years[start_idx : start_idx + trends_mean.shape[0]]

    return trends_ens, trends_mean, trend_years


# ─────────────────────────────────────────────────────────────────────────────
# Model threshold computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_model_thresholds(
    trends_mean: np.ndarray,
    fraction_slowdown: float,
    fraction_riles: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute time-varying model slowdown and RILES thresholds.

    The threshold is scaled from the NSIDC observation-derived fraction by the
    CESM2-LE ensemble mean trend at each trend window. Because the ensemble mean
    trend changes over time (faster ice loss in later decades), this produces a
    threshold that adapts to the model's own forced trajectory:

        model_threshold(t) = fraction_nsidc × ensemble_mean_trend(t)

    A slowdown is anomalously slow relative to the running forced trend;
    a RILES event is anomalously rapid.

    Parameters
    ----------
    trends_mean : np.ndarray
        Ensemble-mean decadal trends, shape (n_trends,), M km² yr⁻¹.
        Typically negative (ice is declining).
    fraction_slowdown : float
        NSIDC fraction for the slowdown threshold (threshold / mean_trend).
        Expected to be < 1 for negative trends (threshold is less negative).
    fraction_riles : float
        NSIDC fraction for the RILES threshold (threshold / mean_trend).
        Expected to be > 1 for negative trends (threshold is more negative).

    Returns
    -------
    threshold_slowdown : np.ndarray
        Time-varying slowdown threshold, shape (n_trends,), M km² yr⁻¹
    threshold_riles : np.ndarray
        Time-varying RILES threshold, shape (n_trends,), M km² yr⁻¹
    """
    threshold_slowdown = fraction_slowdown * trends_mean
    threshold_riles    = fraction_riles    * trends_mean
    return threshold_slowdown, threshold_riles


# ─────────────────────────────────────────────────────────────────────────────
# Event classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_slowdowns(
    trends_ens: np.ndarray,
    threshold_slowdown: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classify each ensemble member's trend window as a slowdown event.

    A slowdown occurs when the decadal trend exceeds the model threshold —
    sea ice loss is anomalously slow (or reversed) relative to the forced decline.

    Parameters
    ----------
    trends_ens : np.ndarray
        Per-member trends, shape (nens, n_trends)
    threshold_slowdown : np.ndarray
        Time-varying slowdown threshold, shape (n_trends,)

    Returns
    -------
    slowdown : np.ndarray
        Binary mask, shape (nens, n_trends); 1 = slowdown, 0 = normal
    trends_slowdown : np.ndarray
        Trend values at slowdown windows, NaN elsewhere, shape (nens, n_trends)
    """
    thresh          = threshold_slowdown[np.newaxis, :]        # (1, n_trends)
    slowdown        = (trends_ens > thresh).astype(int)
    trends_slowdown = np.where(trends_ens > thresh, trends_ens, np.nan)
    return slowdown, trends_slowdown


def classify_riles(
    trends_ens: np.ndarray,
    threshold_riles: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classify each ensemble member's trend window as a RILES event.

    A RILES (Rapid Ice Loss EventS) occurs when the decadal trend falls below
    the model threshold — sea ice loss is anomalously rapid.

    Parameters
    ----------
    trends_ens : np.ndarray
        Per-member trends, shape (nens, n_trends)
    threshold_riles : np.ndarray
        Time-varying RILES threshold, shape (n_trends,)

    Returns
    -------
    riles : np.ndarray
        Binary mask, shape (nens, n_trends); 1 = RILES, 0 = normal
    trends_riles : np.ndarray
        Trend values at RILES windows, NaN elsewhere, shape (nens, n_trends)
    """
    thresh       = threshold_riles[np.newaxis, :]               # (1, n_trends)
    riles        = (trends_ens < thresh).astype(int)
    trends_riles = np.where(trends_ens < thresh, trends_ens, np.nan)
    return riles, trends_riles


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────

def save_slowdown_events(
    slowdown: np.ndarray,
    riles: np.ndarray,
    trends_ens: np.ndarray,
    trends_mean: np.ndarray,
    threshold_slowdown: np.ndarray,
    threshold_riles: np.ndarray,
    trend_years: np.ndarray,
    output_file: str,
    month_idx: Optional[int] = None
) -> None:
    """
    Save CESM2-LE slowdown and RILES events to NetCDF.

    Parameters
    ----------
    slowdown : np.ndarray
        Slowdown binary mask, shape (nens, n_trends)
    riles : np.ndarray
        RILES binary mask, shape (nens, n_trends)
    trends_ens : np.ndarray
        Per-member trends, shape (nens, n_trends), M km² yr⁻¹
    trends_mean : np.ndarray
        Ensemble-mean trend, shape (n_trends,), M km² yr⁻¹
    threshold_slowdown : np.ndarray
        Time-varying slowdown threshold, shape (n_trends,), M km² yr⁻¹
    threshold_riles : np.ndarray
        Time-varying RILES threshold, shape (n_trends,), M km² yr⁻¹
    trend_years : np.ndarray
        Starting year of each 10-yr trend window, shape (n_trends,)
    output_file : str
        Destination NetCDF path
    month_idx : int, optional
        0-based month index used for the description attribute
    """
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    nens = slowdown.shape[0]

    ds = xr.Dataset(
        {
            "slowdown":           (("nens", "nyr"), slowdown),
            "riles":              (("nens", "nyr"), riles),
            "linear_trends_ens":  (("nens", "nyr"), trends_ens),
            "linear_trends_mean": (("nyr",),         trends_mean),
            "threshold_slowdown": (("nyr",),          threshold_slowdown),
            "threshold_riles":    (("nyr",),          threshold_riles),
        },
        coords={
            "nens": np.arange(nens),
            "nyr":  trend_years,
        },
    )

    month_str = f" month {month_idx + 1}" if month_idx is not None else ""
    ds.attrs["description"] = (
        f"CESM2-LE slowdown and RILES events{month_str}: 10-yr decadal trends "
        f"classified against NSIDC-scaled model thresholds."
    )
    ds.attrs["threshold_method"] = (
        "model_threshold(t) = fraction_nsidc × ensemble_mean_trend(t)"
    )

    ds["slowdown"].attrs["description"]          = "1 = slowdown event, 0 = normal"
    ds["riles"].attrs["description"]             = "1 = RILES event, 0 = normal"
    ds["linear_trends_ens"].attrs["units"]       = "M km2 yr-1"
    ds["linear_trends_mean"].attrs["units"]      = "M km2 yr-1"
    ds["threshold_slowdown"].attrs["units"]      = "M km2 yr-1"
    ds["threshold_riles"].attrs["units"]         = "M km2 yr-1"
    ds["nyr"].attrs["description"]               = "Starting year of each 10-yr trend window"

    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(output_file, format="NETCDF4", encoding=encoding)
    print(f"✓ Saved slowdown/RILES events to: {output_file}")


# ─────────────────────────────────────────────────────────────────────────────
# Top-level pipeline
# ─────────────────────────────────────────────────────────────────────────────

def compute_cesm2le_slowdowns(
    sie_dir: str,
    threshold_file: str,
    output_dir: str,
    months: Optional[List[str]] = None,
    variable: str = 'sie',
    member_groups: List[str] = ['first50', 'last50'],
    start_year: int = 1850,
    end_year: int = 2100,
    trend_start_year: int = 1990,
    window: int = 10,
) -> None:
    """
    Full pipeline: classify CESM2-LE sea ice trends as slowdowns or RILES.

    For each requested calendar month this function:
    1. Reads per-month NSIDC fraction thresholds.
    2. Loads CESM2-LE SIE (or SIA) for that month.
    3. Computes 10-year sliding linear trends for every ensemble member.
    4. Builds the time-varying model threshold from the ensemble mean trend.
    5. Classifies each trend window as slowdown / RILES.
    6. Saves one NetCDF per month.

    Parameters
    ----------
    sie_dir : str
        Directory containing per-month CESM2-LE SIE/SIA files (output of
        metrics.batch_process_monthly_files)
    threshold_file : str
        Path to NSIDC slowdown thresholds NetCDF. Must be the output of
        scripts/01_slowdown_nsidc_sie_sia.py, i.e.
        paths.NSIDC_SIE_SLOWDOWN_THRESHOLDS or paths.NSIDC_SIA_SLOWDOWN_THRESHOLDS.
    output_dir : str
        Directory for output NetCDF files
    months : list of str, optional
        Month labels to process (default: all 12, e.g. ['SEP'] for September only)
    variable : str, optional
        'sie' for sea ice extent (default) or 'sia' for sea ice area
    member_groups : list of str, optional
        Ensemble groups to load and concatenate (default: ['first50', 'last50'])
    start_year : int, optional
        First year in the SIE/SIA data (default: 1850)
    end_year : int, optional
        Last year in the SIE/SIA data (default: 2100)
    trend_start_year : int, optional
        First year to include in the trend output (default: 1990)
    window : int, optional
        Trend window length in years (default: 10)

    Examples
    --------
    >>> from configs import paths
    >>> compute_cesm2le_slowdowns(
    ...     sie_dir=str(paths.CESM2LE_DIR / 'sie'),
    ...     threshold_file=str(paths.NSIDC_SIE_SLOWDOWN_THRESHOLDS),
    ...     output_dir=str(paths.CESM2LE_DIR / 'slowdowns'),
    ...     months=['SEP'],
    ...     variable='sie',
    ... )
    """
    if months is None:
        months = MONTH_LABELS

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load NSIDC thresholds once (all 12 months in one file)
    print(f"Loading NSIDC thresholds from:\n  {threshold_file}")
    thresholds  = load_nsidc_slowdown_thresholds(threshold_file)
    frac_slow   = thresholds['fraction_slowdown']    # (12,)
    frac_riles  = thresholds['fraction_riles']       # (12,)

    for month in months:
        month_idx = MONTH_LABELS.index(month)
        print(f"\n{'='*60}")
        print(f"  Month: {month}  (index {month_idx})")
        print(f"  Fractions — slowdown: {frac_slow[month_idx]:.4f}  "
              f"riles: {frac_riles[month_idx]:.4f}")
        print(f"{'='*60}")

        # Load SIE/SIA for this month
        print(f"  Loading CESM2-LE {variable.upper()} ...")
        sie, years = load_sie_monthly_files(
            data_dir=sie_dir,
            month=month,
            member_groups=member_groups,
            variable=variable,
            start_year=start_year,
            end_year=end_year,
        )
        print(f"  Shape: {sie.shape}  (nens, nyear={end_year - start_year + 1})")

        # Decadal trends
        print(f"  Computing {window}-yr sliding trends (start {trend_start_year}) ...")
        trends_ens, trends_mean, trend_years = compute_decadal_trends_ensemble(
            sie, years, window=window, start_year=trend_start_year
        )
        print(f"  Trends shape: {trends_ens.shape}  "
              f"({trend_years[0]}–{trend_years[-1]})")

        # Model thresholds
        threshold_slow, threshold_ril = compute_model_thresholds(
            trends_mean,
            fraction_slowdown=frac_slow[month_idx],
            fraction_riles=frac_riles[month_idx],
        )

        # Classify
        slowdown, _ = classify_slowdowns(trends_ens, threshold_slow)
        riles, _    = classify_riles(trends_ens, threshold_ril)

        n_slow  = int(slowdown.sum())
        n_riles = int(riles.sum())
        total   = slowdown.size
        print(f"  Slowdowns : {n_slow:5d} / {total}  ({100 * n_slow / total:.1f}%)")
        print(f"  RILES     : {n_riles:5d} / {total}  ({100 * n_riles / total:.1f}%)")

        # Save
        out_fname = (f'cesm2le_{variable}_slowdown_riles_{month}_'
                     f'{trend_start_year}-{end_year}.nc')
        save_slowdown_events(
            slowdown=slowdown,
            riles=riles,
            trends_ens=trends_ens,
            trends_mean=trends_mean,
            threshold_slowdown=threshold_slow,
            threshold_riles=threshold_ril,
            trend_years=trend_years,
            output_file=str(output_dir / out_fname),
            month_idx=month_idx,
        )

    print(f"\nAll done. Outputs written to: {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Command-line entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(PROJECT_ROOT))
    from configs import paths

    parser = argparse.ArgumentParser(
        description='Classify CESM2-LE sea ice decadal trends as slowdowns or RILES.'
    )
    parser.add_argument(
        '--variable', choices=['sie', 'sia'], default='sie',
        help='Sea ice variable (default: sie)'
    )
    parser.add_argument(
        '--months', nargs='+', default=None,
        metavar='MON',
        help='Month labels to process, e.g. SEP or JAN MAR SEP (default: all 12)'
    )
    parser.add_argument(
        '--sie-dir', default=None,
        help='Directory containing per-month SIE/SIA files '
             '(default: DATA_ROOT/cesm2le/sie or .../sia)'
    )
    parser.add_argument(
        '--output-dir', default=None,
        help='Output directory (default: DATA_ROOT/cesm2le/slowdowns)'
    )
    parser.add_argument(
        '--start-year',       type=int, default=1850,
        help='First year in the SIE data (default: 1850)'
    )
    parser.add_argument(
        '--end-year',         type=int, default=2100,
        help='Last year in the SIE data (default: 2100)'
    )
    parser.add_argument(
        '--trend-start-year', type=int, default=1990,
        help='First year to include in trend output (default: 1990)'
    )
    parser.add_argument(
        '--window',           type=int, default=10,
        help='Trend window length in years (default: 10)'
    )
    args = parser.parse_args()

    # Resolve paths
    variable = args.variable
    sie_dir = (Path(args.sie_dir) if args.sie_dir
               else paths.CESM2LE_DIR / variable)
    output_dir = (Path(args.output_dir) if args.output_dir
                  else paths.CESM2LE_DIR / 'slowdowns')
    threshold_file = (paths.NSIDC_SIE_SLOWDOWN_THRESHOLDS if variable == 'sie'
                      else paths.NSIDC_SIA_SLOWDOWN_THRESHOLDS)

    print(f"Data root      : {paths.DATA_ROOT}")
    print(f"Variable       : {variable.upper()}")
    print(f"SIE dir        : {sie_dir}")
    print(f"Threshold file : {threshold_file}")
    print(f"Output dir     : {output_dir}")

    compute_cesm2le_slowdowns(
        sie_dir=str(sie_dir),
        threshold_file=str(threshold_file),
        output_dir=str(output_dir),
        months=args.months,
        variable=variable,
        start_year=args.start_year,
        end_year=args.end_year,
        trend_start_year=args.trend_start_year,
        window=args.window,
    )
