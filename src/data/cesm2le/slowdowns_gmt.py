#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CESM2-LE GMT Slowdown Detection
================================

Classify 10-year sliding decadal trends in CESM2-LE yearly mean global mean
temperature (GMT) as "slowdown" events, using thresholds derived analogously
to the NSIDC-based SIE slowdown thresholds.

Sign convention
---------------
GMT is generally *increasing* over time, so a slowdown is defined as a
decadal trend that is **lower** than the slowdown threshold — the warming
rate is anomalously slow (or reversed) relative to the forced warming trend.
This is the opposite of SIE, where the forced trend is negative (declining)
and a slowdown is a trend that is *higher* (less negative) than the threshold.

Approach
--------
1. Load yearly mean GMT from the output of scripts/01_cesm2le_preprocessing.py.
2. Compute 10-year sliding linear trends (reuses
   ``slowdowns.compute_decadal_trends_ensemble``).
3. Define a slowdown threshold as:

       threshold(t) = fraction × ensemble_mean_trend(t)

   where ``fraction`` is derived from the GMT trends themselves using
   mean − 1σ logic (analogous to how the NSIDC SIE fraction is derived from
   mean + 1σ, but with flipped sign).

4. Classify each member's trend window as a slowdown where
   ``trend < threshold`` (warming anomalously slow).
5. Save results to NetCDF in the same directory/style as SIE slowdown files.

Reference: src/data/cesm2le/slowdowns.py  (SIE version)

Author: Lauren Hoffman
Email:  lhoffma2@ucsc.edu
"""

import numpy as np
import xarray as xr
from pathlib import Path
from typing import Optional, Tuple, List

from .slowdowns import compute_decadal_trends_ensemble


# ─────────────────────────────────────────────────────────────────────────────
# Load yearly GMT from preprocessing output
# ─────────────────────────────────────────────────────────────────────────────

def load_gmt_yearly(
    gmt_dir: str,
    member_groups: List[str] = ['first50', 'last50'],
    start_year: int = 1990,
    end_year: int = 2100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load CESM2-LE yearly mean GMT and concatenate across member groups.

    Reads the annual-mean GMT files produced by
    ``scripts/01_cesm2le_preprocessing.py`` (which calls
    ``combine.calculate_annual_mean`` for TREFHT).

    File naming:  trefht_gmt_cesmle_{member_label}_{start}-{end}.nc
    Variable:     trefht_gmt  — shape (nem, nyr)

    Parameters
    ----------
    gmt_dir : str
        Directory containing the yearly GMT NetCDF files.
    member_groups : list of str, optional
        Ensemble groups to concatenate (default: ['first50', 'last50']).
    start_year : int, optional
        First year in the data (default: 1990).
    end_year : int, optional
        Last year in the data (default: 2100).

    Returns
    -------
    gmt : np.ndarray
        GMT time series, shape (nens, nyear), units: K.
    years : np.ndarray
        Year labels, shape (nyear,).
    """
    gmt_dir = Path(gmt_dir)
    group_arrays = []

    for group in member_groups:
        member_label = f'{group}members'
        fpath = gmt_dir / (
            f'trefht_gmt_cesmle_{member_label}_{start_year}-{end_year}.nc'
        )
        if not fpath.exists():
            raise FileNotFoundError(
                f"GMT yearly file not found: {fpath}\n"
                f"Run scripts/01_cesm2le_preprocessing.py first."
            )

        ds = xr.open_dataset(fpath)
        chunk = ds['trefht_gmt'].values.astype(np.float32)  # (nem, nyr)
        years = ds['years'].values
        ds.close()
        group_arrays.append(chunk)

    gmt = np.concatenate(group_arrays, axis=0)  # (nens, nyear)
    return gmt, years


# ─────────────────────────────────────────────────────────────────────────────
# GMT slowdown threshold (from model trends themselves)
# ─────────────────────────────────────────────────────────────────────────────

def compute_gmt_slowdown_fraction(
    trends_ens: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Derive the GMT slowdown fraction from the ensemble distribution of trends.

    For GMT, the forced trend is positive (warming), so a slowdown is
    anomalously *low* warming.  The threshold is defined as:

        threshold = mean_trend - 1σ

    and the fraction is:

        fraction = threshold / mean_trend

    This fraction (< 1) is then used to scale the time-varying ensemble-mean
    trend to produce a time-varying threshold, exactly analogous to the SIE
    approach but with the sign convention flipped.

    Parameters
    ----------
    trends_ens : np.ndarray
        Per-member trends, shape (nens, n_trends).

    Returns
    -------
    fraction : float
        threshold / mean_trend (scalar, averaged over all time windows).
    mean_trend : float
        Grand mean of all trends.
    std_trend : float
        Standard deviation of all trends.
    """
    all_trends = trends_ens.ravel()
    valid = ~np.isnan(all_trends)
    mean_trend = float(np.mean(all_trends[valid]))
    std_trend = float(np.std(all_trends[valid]))

    threshold = mean_trend - std_trend
    fraction = threshold / mean_trend if mean_trend != 0 else 1.0

    return fraction, mean_trend, std_trend


def compute_gmt_model_threshold(
    trends_mean: np.ndarray,
    fraction_slowdown: float,
) -> np.ndarray:
    """
    Compute time-varying GMT slowdown threshold.

        model_threshold(t) = fraction × ensemble_mean_trend(t)

    Parameters
    ----------
    trends_mean : np.ndarray
        Ensemble-mean decadal trends, shape (n_trends,), K yr⁻¹.
    fraction_slowdown : float
        Slowdown fraction (threshold / mean_trend), typically < 1.

    Returns
    -------
    threshold_slowdown : np.ndarray
        Time-varying slowdown threshold, shape (n_trends,), K yr⁻¹.
    """
    return fraction_slowdown * trends_mean


# ─────────────────────────────────────────────────────────────────────────────
# Event classification  (GMT: trend < threshold → slowdown)
# ─────────────────────────────────────────────────────────────────────────────

def classify_gmt_slowdowns(
    trends_ens: np.ndarray,
    threshold_slowdown: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classify each ensemble member's trend window as a GMT slowdown event.

    A GMT slowdown occurs when the decadal warming trend falls *below* the
    model threshold — warming is anomalously slow relative to the forced trend.

    **Sign convention**: opposite to SIE.
    - SIE:  trend > threshold → slowdown  (less negative = slower ice loss)
    - GMT:  trend < threshold → slowdown  (less positive = slower warming)

    Parameters
    ----------
    trends_ens : np.ndarray
        Per-member trends, shape (nens, n_trends), K yr⁻¹.
    threshold_slowdown : np.ndarray
        Time-varying slowdown threshold, shape (n_trends,), K yr⁻¹.

    Returns
    -------
    slowdown : np.ndarray
        Binary mask, shape (nens, n_trends); 1 = slowdown, 0 = normal.
    trends_slowdown : np.ndarray
        Trend values at slowdown windows, NaN elsewhere.
    """
    thresh = threshold_slowdown[np.newaxis, :]  # (1, n_trends)
    slowdown = (trends_ens < thresh).astype(int)
    trends_slowdown = np.where(trends_ens < thresh, trends_ens, np.nan)
    return slowdown, trends_slowdown


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────

def save_gmt_slowdown_events(
    slowdown: np.ndarray,
    trends_ens: np.ndarray,
    trends_mean: np.ndarray,
    threshold_slowdown: np.ndarray,
    trend_years: np.ndarray,
    fraction_slowdown: float,
    output_file: str,
) -> None:
    """
    Save CESM2-LE GMT slowdown events to NetCDF.

    Parameters
    ----------
    slowdown : np.ndarray
        Slowdown binary mask, shape (nens, n_trends).
    trends_ens : np.ndarray
        Per-member trends, shape (nens, n_trends), K yr⁻¹.
    trends_mean : np.ndarray
        Ensemble-mean trend, shape (n_trends,), K yr⁻¹.
    threshold_slowdown : np.ndarray
        Time-varying slowdown threshold, shape (n_trends,), K yr⁻¹.
    trend_years : np.ndarray
        Starting year of each 10-yr trend window, shape (n_trends,).
    fraction_slowdown : float
        The fraction used: threshold / mean_trend.
    output_file : str
        Destination NetCDF path.
    """
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    nens = slowdown.shape[0]

    ds = xr.Dataset(
        {
            "slowdown":              (("nens", "nyr"), slowdown),
            "gmt_trends_ens":        (("nens", "nyr"), trends_ens),
            "gmt_trends_mean":       (("nyr",),        trends_mean),
            "gmt_threshold_slowdown": (("nyr",),        threshold_slowdown),
        },
        coords={
            "nens": np.arange(nens),
            "nyr":  trend_years,
        },
    )

    ds.attrs["description"] = (
        "CESM2-LE GMT slowdown events: 10-yr decadal trends "
        "classified against fraction-scaled ensemble-mean thresholds."
    )
    ds.attrs["threshold_method"] = (
        "model_threshold(t) = fraction × ensemble_mean_trend(t); "
        "slowdown where trend < threshold (anomalously slow warming)"
    )
    ds.attrs["fraction_slowdown"] = fraction_slowdown

    ds["slowdown"].attrs["description"]             = "1 = slowdown event, 0 = normal"
    ds["gmt_trends_ens"].attrs["units"]             = "K yr-1"
    ds["gmt_trends_ens"].attrs["description"]       = "Per-member 10-yr linear trend in yearly GMT"
    ds["gmt_trends_mean"].attrs["units"]            = "K yr-1"
    ds["gmt_trends_mean"].attrs["description"]      = "Ensemble-mean 10-yr linear trend in yearly GMT"
    ds["gmt_threshold_slowdown"].attrs["units"]     = "K yr-1"
    ds["gmt_threshold_slowdown"].attrs["description"] = "Time-varying GMT slowdown threshold"
    ds["nyr"].attrs["description"]                  = "Starting year of each 10-yr trend window"

    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(output_file, format="NETCDF4", encoding=encoding)
    print(f"  Saved GMT slowdown events to: {output_file}")


# ─────────────────────────────────────────────────────────────────────────────
# Top-level pipeline
# ─────────────────────────────────────────────────────────────────────────────

def compute_cesm2le_gmt_slowdowns(
    gmt_dir: str,
    output_dir: str,
    member_groups: List[str] = ['first50', 'last50'],
    start_year: int = 1990,
    end_year: int = 2100,
    trend_start_year: int = 1990,
    window: int = 10,
) -> None:
    """
    Full pipeline: classify CESM2-LE GMT decadal trends as slowdowns.

    Steps:
    1. Load yearly mean GMT from preprocessing output.
    2. Compute 10-year sliding linear trends (reuses SIE trend code).
    3. Derive the slowdown fraction from the ensemble trend distribution.
    4. Build a time-varying threshold and classify slowdowns.
    5. Save to NetCDF.

    Parameters
    ----------
    gmt_dir : str
        Directory containing yearly GMT files (output of
        scripts/01_cesm2le_preprocessing.py).
    output_dir : str
        Directory for output NetCDF files.
    member_groups : list of str, optional
        Ensemble groups to load (default: ['first50', 'last50']).
    start_year : int, optional
        First year in the GMT data (default: 1990).
    end_year : int, optional
        Last year in the GMT data (default: 2100).
    trend_start_year : int, optional
        First year to include in the trend output (default: 1990).
    window : int, optional
        Trend window length in years (default: 10).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load yearly GMT
    print(f"Loading yearly GMT from: {gmt_dir}")
    gmt, years = load_gmt_yearly(
        gmt_dir=gmt_dir,
        member_groups=member_groups,
        start_year=start_year,
        end_year=end_year,
    )
    print(f"  GMT shape: {gmt.shape}  (nens, nyear)")

    # 2. Compute decadal trends (reuses SIE trend helper)
    print(f"  Computing {window}-yr sliding trends (start {trend_start_year}) ...")
    trends_ens, trends_mean, trend_years = compute_decadal_trends_ensemble(
        gmt, years, window=window, start_year=trend_start_year
    )
    print(f"  Trends shape: {trends_ens.shape}  "
          f"({trend_years[0]}-{trend_years[-1]})")

    # 3. Derive slowdown fraction from model trends
    fraction, grand_mean, grand_std = compute_gmt_slowdown_fraction(trends_ens)
    print(f"  Grand mean trend : {grand_mean:.6f} K/yr")
    print(f"  Grand std  trend : {grand_std:.6f} K/yr")
    print(f"  Slowdown fraction: {fraction:.4f}")

    # 4. Time-varying threshold and classification
    threshold_slow = compute_gmt_model_threshold(trends_mean, fraction)
    slowdown, _ = classify_gmt_slowdowns(trends_ens, threshold_slow)

    n_slow = int(slowdown.sum())
    total = slowdown.size
    print(f"  GMT slowdowns: {n_slow:5d} / {total}  ({100 * n_slow / total:.1f}%)")

    # 5. Save
    out_fname = f'cesm2le_gmt_slowdown_{trend_start_year}-{end_year}.nc'
    save_gmt_slowdown_events(
        slowdown=slowdown,
        trends_ens=trends_ens,
        trends_mean=trends_mean,
        threshold_slowdown=threshold_slow,
        trend_years=trend_years,
        fraction_slowdown=fraction,
        output_file=str(output_dir / out_fname),
    )

    print(f"\nGMT slowdown pipeline complete. Output: {output_dir / out_fname}")
