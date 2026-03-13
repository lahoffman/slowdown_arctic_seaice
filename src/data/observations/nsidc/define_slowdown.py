"""
define_slowdown.py — decadal trend analysis and event detection
===============================================================
Computes 10-year sliding linear trends from monthly SIE, then defines
two types of anomalous trend events:

  slowdown — trend is anomalously *positive* (mean + 1σ), i.e. sea ice
             loss is decelerating or reversing relative to the long-run mean.
  riles    — trend is anomalously *negative* (mean − 1σ), i.e. sea ice
             loss is accelerating beyond the long-run mean.

Reference: x_old/data_processing/D0_SLOWDOWN_sie_nsidc_cesmle.py, lines 194+.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr


# ---------------------------------------------------------------------------
# Trend computation
# ---------------------------------------------------------------------------

def compute_decadal_trends(
    data: np.ndarray,
    yearmon: np.ndarray,
    window: int = 10,
    start_year: int = 1990,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute sliding *window*-year linear trends for each calendar month.

    For each month ``i`` and each starting year-index ``j``, a linear
    regression is fit over ``data[i, j : j+window]`` using non-NaN points.

    Parameters
    ----------
    data : np.ndarray, shape (12, n_years)
        Cleaned monthly SIE from ``preprocess_nsidc_sie()``.
    yearmon : np.ndarray, shape (12, n_years)
        Year labels (output of ``preprocess_nsidc_sie()``).
    window : int
        Trend window length in years (default 10).
    start_year : int
        Discard trend windows whose starting year is earlier than this.
        Defaults to 1990 (matches the original analysis).

    Returns
    -------
    linear_trends : np.ndarray, shape (12, n_trends)
        Trend slopes in M km² yr⁻¹ for each month and trend window.
    trend_years : np.ndarray, shape (12, n_trends)
        Starting year of each trend window (same layout as linear_trends).
    """
    nn = data.shape[1]

    # Index in yearmon where start_year first appears (same for every row)
    start_idx = int(np.where(yearmon[0, :] == start_year)[0][0])

    # Compute all windows [0 … nn-window-1] then trim to start_year
    all_slopes = []
    for i in range(12):
        month_slopes = []
        for j in range(nn - window):
            dx = np.arange(window, dtype=float)
            dy = data[i, j : j + window].astype(float)

            valid = ~np.isnan(dy)
            if valid.sum() >= 2:
                coeffs = np.polyfit(dx[valid], dy[valid], 1)
                month_slopes.append(coeffs[0])
            else:
                month_slopes.append(np.nan)
        all_slopes.append(month_slopes)

    all_trends = np.array(all_slopes)               # (12, nn-window)
    linear_trends = all_trends[:, start_idx:]        # (12, n_trends)
    trend_years   = yearmon[:, start_idx : start_idx + linear_trends.shape[1]]

    return linear_trends, trend_years


# ---------------------------------------------------------------------------
# Threshold definition — slowdowns
# ---------------------------------------------------------------------------

def define_slowdown_threshold(
    linear_trends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Define the slowdown threshold as mean + 1σ of the decadal trends.

    A 10-year window is labelled a *slowdown* when its trend exceeds this
    threshold — sea ice loss is anomalously slow (or reversed).

    Parameters
    ----------
    linear_trends : np.ndarray, shape (12, n_trends)
        Output of ``compute_decadal_trends()``.

    Returns
    -------
    threshold : np.ndarray, shape (12,)
        Per-month slowdown threshold (M km² yr⁻¹).
    mask : np.ndarray of bool, shape (12, n_trends)
        True where the window qualifies as a slowdown.
    fraction : np.ndarray, shape (12,)
        Threshold expressed as a fraction of the mean trend
        (threshold / mean_trend).
    """
    mean_trend = np.nanmean(linear_trends, axis=1)   # (12,)
    std_trend  = np.nanstd(linear_trends,  axis=1)   # (12,)

    threshold = mean_trend + std_trend
    mask      = linear_trends > threshold[:, np.newaxis]
    fraction  = threshold / mean_trend

    return threshold, mask, fraction


# ---------------------------------------------------------------------------
# Threshold definition — riles
# ---------------------------------------------------------------------------

def define_riles_threshold(
    linear_trends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Define the riles threshold as mean − 1σ of the decadal trends.

    A 10-year window is labelled a *riles* event when its trend falls below
    this threshold — sea ice loss is anomalously rapid.

    Parameters
    ----------
    linear_trends : np.ndarray, shape (12, n_trends)
        Output of ``compute_decadal_trends()``.

    Returns
    -------
    threshold : np.ndarray, shape (12,)
        Per-month riles threshold (M km² yr⁻¹).
    mask : np.ndarray of bool, shape (12, n_trends)
        True where the window qualifies as a riles event.
    fraction : np.ndarray, shape (12,)
        Threshold expressed as a fraction of the mean trend
        (threshold / mean_trend).
    """
    mean_trend = np.nanmean(linear_trends, axis=1)   # (12,)
    std_trend  = np.nanstd(linear_trends,  axis=1)   # (12,)

    threshold = mean_trend - std_trend
    mask      = linear_trends < threshold[:, np.newaxis]
    fraction  = threshold / mean_trend

    return threshold, mask, fraction


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_slowdown_thresholds(
    threshold_slowdown: np.ndarray,
    fraction_slowdown: np.ndarray,
    threshold_riles: np.ndarray,
    fraction_riles: np.ndarray,
    output_file: str,
) -> None:
    """
    Save per-month slowdown and riles thresholds to NetCDF.

    Parameters
    ----------
    threshold_slowdown, fraction_slowdown : np.ndarray, shape (12,)
        Outputs of ``define_slowdown_threshold()``.
    threshold_riles, fraction_riles : np.ndarray, shape (12,)
        Outputs of ``define_riles_threshold()``.
    output_file : str
        Destination path (e.g. ``'.../sie_nsidc_thresholds.nc'``).
    """
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    ds = xr.Dataset(
        {
            "threshold_slowdown": (("month",), threshold_slowdown),
            "fraction_slowdown":  (("month",), fraction_slowdown),
            "threshold_riles":    (("month",), threshold_riles),
            "fraction_riles":     (("month",), fraction_riles),
        },
        coords={"month": np.arange(1, 13)},
    )
    ds.attrs["description"] = (
        "NSIDC SIE per-month slowdown and riles thresholds "
        "(10-yr decadal trend ± 1σ)"
    )
    ds["threshold_slowdown"].attrs["units"] = "M km2 yr-1"
    ds["threshold_riles"].attrs["units"]    = "M km2 yr-1"

    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(output_file, encoding=encoding)
    print(f"✓ Saved thresholds to: {output_file}")


def save_slowdown_events(
    mask: np.ndarray,
    ice: np.darray,
    yearmon: np.ndarray,
    linear_trends: np.ndarray,
    trend_years: np.ndarray,
    month_idx: int,
    output_file: str,
    start_year: int = 1990,
) -> None:
    """
    Save the slowdown event mask and trend values for one calendar month.

    Parameters
    ----------
    mask : np.ndarray of bool/int, shape (12, n_trends)
        Full slowdown mask from ``define_slowdown_threshold()``.
    linear_trends : np.ndarray, shape (12, n_trends)
        Full trend array from ``compute_decadal_trends()``.
    trend_years : np.ndarray, shape (12, n_trends)
        Starting years for each trend window.
    month_idx : int
        0-based month index (0 = January … 8 = September).
    output_file : str
        Destination path.
    """
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    start_idx = int(np.where(yearmon[0, :] == start_year)[0][0])
    yearsice = yearmon[month_idx, start_idx:].astype(int)
    years  = trend_years[month_idx].astype(int)
    events = mask[month_idx].astype(int)
    trends = linear_trends[month_idx]
    seaice = ice[month_idx,start_idx:] 

    ds = xr.Dataset(
        {
            "slowdown":      (("year",), events),
            "linear_trend":  (("year",), trends),
            "seaice":        (("yearsice",), seaice),
        },
        coords={"year": years, "yearice": yearsice},
    )
    ds.attrs["description"] = (
        f"NSIDC SIE slowdown events — month {month_idx + 1} "
        f"(10-yr decadal trend > mean + 1σ)"
    )
    ds["slowdown"].attrs["description"]     = "1 = slowdown event, 0 = normal"
    ds["linear_trend"].attrs["units"]       = "M km2 yr-1"
    ds["seaice"].attrs["units"]            = "M km2"

    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(output_file, encoding=encoding)
    print(f"✓ Saved slowdown events (month {month_idx + 1}) to: {output_file}")


def save_riles_events(
    mask: np.ndarray,
    linear_trends: np.ndarray,
    trend_years: np.ndarray,
    month_idx: int,
    output_file: str,
) -> None:
    """
    Save the riles event mask and trend values for one calendar month.

    Parameters
    ----------
    mask : np.ndarray of bool/int, shape (12, n_trends)
        Full riles mask from ``define_riles_threshold()``.
    linear_trends : np.ndarray, shape (12, n_trends)
        Full trend array from ``compute_decadal_trends()``.
    trend_years : np.ndarray, shape (12, n_trends)
        Starting years for each trend window.
    month_idx : int
        0-based month index (0 = January … 8 = September).
    output_file : str
        Destination path.
    """
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    years  = trend_years[month_idx].astype(int)
    events = mask[month_idx].astype(int)
    trends = linear_trends[month_idx]

    ds = xr.Dataset(
        {
            "riles":         (("year",), events),
            "linear_trend":  (("year",), trends),
        },
        coords={"year": years},
    )
    ds.attrs["description"] = (
        f"NSIDC SIE riles events — month {month_idx + 1} "
        f"(10-yr decadal trend < mean − 1σ)"
    )
    ds["riles"].attrs["description"]    = "1 = riles event, 0 = normal"
    ds["linear_trend"].attrs["units"]   = "M km2 yr-1"

    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(output_file, encoding=encoding)
    print(f"✓ Saved riles events (month {month_idx + 1}) to: {output_file}")
