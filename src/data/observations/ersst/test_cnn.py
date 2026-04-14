"""
Observation-based CNN test input preparation.

Converts regridded ERSSTv5 output (from scripts/01_ersst_preprocessing.py)
into the exact format expected by the trained JJA SST CNN for inference
on observational data (1990–2024).

Pipeline
--------
1. load_regridded_ersst     — read the regridded ERSST NetCDF
2. select_years             — subset to 1990–2024 monthly data
3. compute_jja_mean         — reshape to (nyear, 12, nx, ny) and average Jun/Jul/Aug
4. apply_land_ocean_mask    — replace land cells with NaN (using CESM2-LE land mask)
5. remove_linear_trend      — per-pixel linear detrend (removes forced signal)
6. standardize_obs          — global ocean-only mean/std normalisation
7. fill_land_sentinel       — replace land NaN with -10.0 (CNN sentinel value)
8. format_for_cnn           — add trailing channel dimension → (nyear, 192, 288, 1)
9. prepare_obs_for_cnn      — convenience wrapper running the full pipeline

Replicates the logic of:
    x_old/data_processing/D2_TVT_OBS_sst_jja_forced_linear.py
while matching the new project structure and CNN interface.

Authors: Lauren Hoffman  <lhoffma2@ucsc.edu>
"""

import numpy as np
import netCDF4 as nc
from pathlib import Path
from typing import Optional, Tuple, Dict


# ============================================================================
# 1. Load regridded ERSST
# ============================================================================

def load_regridded_ersst(filepath: str) -> np.ndarray:
    """
    Load the regridded ERSSTv5 SST array from NetCDF.

    Expects the output of ``scripts/01_ersst_preprocessing.py``, which
    stores variable ``sst_obs`` with dimensions ``(nte, nx, ny)`` on the
    CESM2-LE atmospheric grid (192 x 288).

    Parameters
    ----------
    filepath : str or Path
        Path to ``sst_regrid_cesm2le.nc``.

    Returns
    -------
    sst_obs : np.ndarray, shape (ntime, 192, 288)
        Monthly SST on the CESM2-LE grid.  Land cells are NaN (inherited
        from the bilinear interpolation of ocean-only ERSSTv5).
    """
    filepath = str(filepath)
    with nc.Dataset(filepath, 'r') as ds:
        sst_obs = np.array(ds.variables['sst_obs'][:], dtype=np.float64)
    return sst_obs


# ============================================================================
# 2. Select years
# ============================================================================

def select_years(
    sst_monthly: np.ndarray,
    data_start_year: int,
    start_year: int = 1990,
    end_year: int = 2024,
) -> np.ndarray:
    """
    Subset monthly SST to the requested year range.

    The old script builds a date index starting at 1854-01 and slices from
    the first January of ``start_year``.  Here we compute the offset in
    months from ``data_start_year`` (the first year present in the file).

    Parameters
    ----------
    sst_monthly : np.ndarray, shape (ntime, nx, ny)
        Full monthly time series.
    data_start_year : int
        Calendar year of the first time step in *sst_monthly* (e.g. 1854).
    start_year : int
        First year to include (default 1990).
    end_year : int
        Last year to include (default 2024).

    Returns
    -------
    sst_subset : np.ndarray, shape (n_months_subset, nx, ny)
        Monthly data for [start_year-01 … end_year-12].
    """
    idx_start = (start_year - data_start_year) * 12
    idx_end = (end_year - data_start_year + 1) * 12  # exclusive, through Dec

    ntime = sst_monthly.shape[0]
    if idx_end > ntime:
        # Partial final year — take whatever is available
        idx_end = ntime

    return sst_monthly[idx_start:idx_end]


# ============================================================================
# 3. Compute JJA mean
# ============================================================================

def compute_jja_mean(sst_monthly: np.ndarray) -> np.ndarray:
    """
    Compute the June-July-August seasonal mean from monthly data.

    Reshapes the monthly array into ``(nyear, 12, nx, ny)`` and averages
    over months 5, 6, 7 (0-indexed), i.e. June, July, August.  

    Parameters
    ----------
    sst_monthly : np.ndarray, shape (n_months, nx, ny)
        Monthly SST.  ``n_months`` must be divisible by 12 (complete years).

    Returns
    -------
    sst_jja : np.ndarray, shape (nyear, nx, ny)
        JJA seasonal-mean SST.
    """
    ntime, nx, ny = sst_monthly.shape
    if ntime % 12 != 0:
        # Truncate to complete years (same implicit assumption in old script)
        n_complete = (ntime // 12) * 12
        sst_monthly = sst_monthly[:n_complete]
        ntime = n_complete

    nyear = ntime // 12
    sst_reshape = sst_monthly.reshape(nyear, 12, nx, ny)

    # Months 5:8 → June (idx 5), July (idx 6), August (idx 7)
    sst_jja = np.nanmean(sst_reshape[:, 5:8, :, :], axis=1)
    return sst_jja


# ============================================================================
# 4. Apply land/ocean mask
# ============================================================================

def apply_land_ocean_mask(
    sst: np.ndarray,
    landmask: np.ndarray,
) -> np.ndarray:
    """
    Set land grid cells to NaN.

    Parameters
    ----------
    sst : np.ndarray, shape (nyear, nx, ny)
    landmask : np.ndarray, shape (nx, ny)
        0 = ocean, 1 = land.

    Returns
    -------
    sst_masked : np.ndarray, shape (nyear, nx, ny)
        SST with land cells set to NaN.
    """
    lm = landmask[np.newaxis, :, :]  # (1, nx, ny) for broadcasting
    return np.where(lm == 0, sst, np.nan)


# ============================================================================
# 5. Remove linear forced trend
# ============================================================================

def remove_linear_trend(
    sst_jja: np.ndarray,
    start_year: int = 1990,
    end_year: int = 2024,
) -> np.ndarray:
    """
    Remove the forced signal by subtracting a per-pixel linear trend.

    Parameters
    ----------
    sst_jja : np.ndarray, shape (nyear, nx, ny)
        JJA-mean SST (may contain NaN on land).
    start_year : int
        First year (used to build the regressor axis).
    end_year : int
        Last year (inclusive).

    Returns
    -------
    sst_residual : np.ndarray, shape (nyear, nx, ny)
        Linearly detrended JJA SST.
    """
    nyear, nx, ny = sst_jja.shape
    years = np.arange(start_year, end_year + 1, dtype=np.float64)

    if len(years) != nyear:
        raise ValueError(
            f"Year range {start_year}–{end_year} implies {len(years)} years, "
            f"but sst_jja has {nyear} time steps."
        )

    sst_residual = np.empty_like(sst_jja)

    for i in range(nx):
        for j in range(ny):
            ts = sst_jja[:, i, j]
            if np.all(np.isnan(ts)):
                # Pure land — leave as NaN
                sst_residual[:, i, j] = np.nan
                continue
            # Suppress polyfit RankWarning for all-NaN or near-constant columns
            with np.errstate(invalid='ignore'):
                coeff = np.polyfit(years, ts, 1)
            y_fit = np.poly1d(coeff)(years)
            sst_residual[:, i, j] = ts - y_fit

    return sst_residual


# ============================================================================
# 6. Standardise (global ocean-only mean/std)
# ============================================================================

def standardize_obs(
    sst_residual: np.ndarray,
    mu: Optional[float] = None,
    sigma: Optional[float] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Standardise the detrended SST field.

    By default, computes the global mean and standard deviation from all
    non-NaN values in ``sst_residual`` (i.e., ocean pixels only, since
    land is NaN). 

    If external ``mu`` / ``sigma`` are provided (e.g., training-set
    statistics from a saved TVT split), those are used instead — this is
    the correct approach when running inference with a model trained on
    CESM2-LE data, so that observation and model inputs share the same
    normalisation reference.

    Parameters
    ----------
    sst_residual : np.ndarray, shape (nyear, nx, ny)
        Detrended JJA SST (NaN on land).
    mu : float, optional
        Pre-computed mean.  If None, estimated from *sst_residual*.
    sigma : float, optional
        Pre-computed std.  If None, estimated from *sst_residual*.

    Returns
    -------
    sst_std : np.ndarray, shape (nyear, nx, ny)
        Standardised field.
    mu : float
        Mean used for standardisation.
    sigma : float
        Std used for standardisation.
    """
    if mu is None:
        # nanmean over all axes → scalar; land is NaN so excluded automatically
        mu = float(np.nanmean(sst_residual))
    if sigma is None:
        sigma = float(np.nanstd(sst_residual))

    if sigma == 0:
        raise ValueError("Standard deviation is zero — cannot standardise.")

    sst_std = (sst_residual - mu) / sigma
    return sst_std, mu, sigma


# ============================================================================
# 7. Fill land with sentinel value
# ============================================================================

def fill_land_sentinel(
    sst: np.ndarray,
    landmask: np.ndarray,
    fill_value: float = -10.0,
) -> np.ndarray:
    """
    Replace land pixels with a fixed sentinel value for the CNN.

    The trained CNN expects land cells filled with ``-10.0`` (set during
    training by ``src.cnn.splits.apply_landmask``).  This must be applied
    *after* standardisation so the sentinel sits well outside the range of
    physically meaningful standardised SST values.

    Parameters
    ----------
    sst : np.ndarray, shape (nyear, nx, ny)
        Standardised SST (may have NaN on land).
    landmask : np.ndarray, shape (nx, ny)
        0 = ocean, 1 = land.
    fill_value : float
        Sentinel value for land (default -10.0).

    Returns
    -------
    sst_filled : np.ndarray, shape (nyear, nx, ny)
    """
    lm = landmask[np.newaxis, :, :]
    return np.where(lm == 1, fill_value, sst)


# ============================================================================
# 8. Format for CNN
# ============================================================================

def format_for_cnn(sst: np.ndarray) -> np.ndarray:
    """
    Add trailing channel dimension for the CNN.

    The model expects input shape ``(n_samples, nx, ny, 1)`` (a single
    SST channel).

    Parameters
    ----------
    sst : np.ndarray, shape (n_samples, nx, ny)

    Returns
    -------
    np.ndarray, shape (n_samples, nx, ny, 1)
    """
    return sst[..., np.newaxis].astype(np.float32)


# ============================================================================
# 9. Full pipeline wrapper
# ============================================================================

def prepare_obs_for_cnn(
    regridded_ersst_path: str,
    landmask_path: str,
    data_start_year: int = 1854,
    start_year: int = 1990,
    end_year: int = 2024,
    mu: Optional[float] = None,
    sigma: Optional[float] = None,
    land_fill_value: float = -10.0,
) -> Dict:
    """
    End-to-end pipeline: regridded ERSST → CNN-ready observation tensor.

    Runs steps 1–8 in sequence and returns a dict with the final array
    plus intermediate products useful for diagnostics.

    Parameters
    ----------
    regridded_ersst_path : str or Path
        Path to ``sst_regrid_cesm2le.nc`` (output of 01_ersst_preprocessing.py).
    landmask_path : str or Path
        Path to the CESM2-LE land mask NetCDF (variable ``landmask``,
        shape (192, 288), 0 = ocean, 1 = land).
    data_start_year : int
        Calendar year of the first month in the regridded file (default 1854).
    start_year : int
        First year for the observation window (default 1990).
    end_year : int
        Last year for the observation window (default 2024).
    mu : float, optional
        External mean for standardisation (e.g., from a TVT training split).
        If None, computed from the observation residuals themselves — this
        matches the old script's behaviour.
    sigma : float, optional
        External std for standardisation.  Same convention as *mu*.
    land_fill_value : float
        Sentinel for land pixels (default -10.0).

    Returns
    -------
    dict with keys
        ``X``            : np.ndarray (nyear, 192, 288, 1) — CNN input
        ``sst_jja``      : np.ndarray (nyear, 192, 288)    — raw JJA means
        ``sst_residual`` : np.ndarray (nyear, 192, 288)    — after detrending
        ``sst_std``      : np.ndarray (nyear, 192, 288)    — after standardising
        ``mu``           : float — mean used
        ``sigma``        : float — std used
        ``years``        : np.ndarray (nyear,)
    """
    # --- 0. Load land mask ------------------------------------------------
    with nc.Dataset(str(landmask_path), 'r') as ds:
        landmask = np.array(ds.variables['landmask'][:])

    # --- 1. Load regridded ERSST ------------------------------------------
    sst_all = load_regridded_ersst(regridded_ersst_path)
    print(f"Loaded regridded ERSST: {sst_all.shape}")

    # --- 2. Select years --------------------------------------------------
    sst_sub = select_years(sst_all, data_start_year, start_year, end_year)
    print(f"Selected {start_year}–{end_year}: {sst_sub.shape}")

    # --- 3. JJA mean ------------------------------------------------------
    sst_jja = compute_jja_mean(sst_sub)
    years = np.arange(start_year, start_year + sst_jja.shape[0])
    print(f"JJA mean: {sst_jja.shape}, years {years[0]}–{years[-1]}")

    # --- 4. Apply land mask (NaN on land) ---------------------------------
    sst_jja_masked = apply_land_ocean_mask(sst_jja, landmask)

    # --- 5. Remove linear trend -------------------------------------------
    sst_residual = remove_linear_trend(
        sst_jja_masked, start_year=int(years[0]), end_year=int(years[-1])
    )
    print(f"Linear trend removed: {sst_residual.shape}")

    # --- 6. Standardise ---------------------------------------------------
    sst_std, mu_used, sigma_used = standardize_obs(sst_residual, mu=mu, sigma=sigma)
    print(f"Standardised (mu={mu_used:.4f}, sigma={sigma_used:.4f})")

    # --- 7. Fill land sentinel --------------------------------------------
    sst_filled = fill_land_sentinel(sst_std, landmask, fill_value=land_fill_value)

    # --- 8. Format for CNN ------------------------------------------------
    X = format_for_cnn(sst_filled)
    print(f"CNN input ready: {X.shape}  (n_samples, nx, ny, nch)")

    return {
        'X': X,
        'sst_jja': sst_jja,
        'sst_residual': sst_residual,
        'sst_std': sst_std,
        'mu': mu_used,
        'sigma': sigma_used,
        'years': years,
    }
