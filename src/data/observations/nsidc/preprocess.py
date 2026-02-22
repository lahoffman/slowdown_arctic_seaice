"""
preprocess.py — NSIDC Sea Ice Index preprocessing
==================================================
Loads the NSIDC Sea Ice Index monthly Excel file, fills a known data gap
(Dec 1987 / Jan 1988), and returns a clean (12 × n_years) monthly array
for either sea ice extent (SIE) or sea ice area (SIA).

Reference: x_old/data_processing/D0_SLOWDOWN_sie_nsidc_cesmle.py, lines 1–194.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MONTHS = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December',
]

# NH sheet suffix used in the Excel file
_SHEET_TAIL = '-NH'

# 1978 boundary: Jan–Oct have no SMMR data (NSIDC starts Nov 1978)
# True  → set first year to NaN
_MASK_1978 = [True, True, True, True, True, True, True, True, True, True, False, False]

# Current-year boundary: Mar–Dec are incomplete at time of download
# True  → set last year to NaN
_MASK_CUR  = [False, False, True, True, True, True, True, True, True, True, True, True]

# Excel column mapping for each variable (pandas unnamed-column convention)
_VARIABLE_COL = {
    'extent': 'Unnamed: 5',
    'area':   'Unnamed: 6',
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_nsidc_sie(
    file_path: str,
    current_year: Optional[int] = None,
    variable: str = 'extent',
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load monthly NH sea ice extent *or* area from the NSIDC Sea Ice Index
    Excel file.

    Each of the 12 monthly sheets is read and values placed into a
    (12, n_years) array aligned to the common year range 1978–current_year.

    Parameters
    ----------
    file_path : str
        Path to ``Sea_Ice_Index_Monthly_Data_with_Statistics_G02135_v3.0.xlsx``.
    current_year : int, optional
        Override the reference "current year" (defaults to today's year).
    variable : {'extent', 'area'}
        Which variable to load.  ``'extent'`` reads the Extent column
        (M km²); ``'area'`` reads the Area column (M km²).

    Returns
    -------
    data : np.ndarray, shape (12, n_years)
        Raw monthly SIE or SIA in M km², NaN where data are unavailable.
    yearrange : np.ndarray, shape (n_years,)
        Year labels 1978 … current_year.
    """
    if variable not in _VARIABLE_COL:
        raise ValueError(f"variable must be one of {list(_VARIABLE_COL)}; got '{variable}'")
    col = _VARIABLE_COL[variable]

    if current_year is None:
        current_year = datetime.now().year

    yearrange = np.arange(1978, current_year + 1)
    nn = len(yearrange)
    monthly = []

    for i, month in enumerate(MONTHS):
        values = np.full(nn, np.nan)
        sheet = month + _SHEET_TAIL
        df = pd.read_excel(file_path, sheet_name=sheet)

        # Data starts at row index 9 in the raw sheet
        timei = np.array(df[['Unnamed: 1']])[9:].ravel()
        vali  = np.array(df[[col]])[9:].ravel()

        # Jan–Oct start at 1979 (first full year of SSM/I); Nov–Dec at 1978
        start = 1979 if i < 10 else 1978
        # Jan–Feb may include the current year; Mar–Dec stop at the previous year
        end   = current_year if i < 2 else current_year - 1

        si = np.where(yearrange == start)[0][0]
        ei = np.where(yearrange == end)[0][0]
        values[si : ei + 1] = vali[: ei - si + 1]
        monthly.append(values)

    return np.array(monthly), yearrange  # (12, nn), (nn,)


def preprocess_nsidc_sie(
    file_path: str,
    current_year: Optional[int] = None,
    variable: str = 'extent',
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Full preprocessing pipeline for NSIDC monthly sea ice extent or area.

    Steps
    -----
    1. Load all 12 monthly Excel sheets → raw (12, n_years) array.
    2. Interpolate the Dec 1987 / Jan 1988 data gap (quadratic fit).
    3. Mask incomplete boundary months (1978 partial start; current-year
       partial end).

    Parameters
    ----------
    file_path : str
        Path to the NSIDC Sea Ice Index Excel file.
    current_year : int, optional
        Override the reference "current year" (defaults to today's year).
    variable : {'extent', 'area'}
        Which variable to load and preprocess (default ``'extent'``).

    Returns
    -------
    data : np.ndarray, shape (12, n_years)
        Cleaned monthly SIE or SIA (M km²), ready for trend analysis.
        Rows = months (0 = Jan … 11 = Dec), columns = years.
    yearmon : np.ndarray, shape (12, n_years)
        Year label for every entry in *data*.
    time : list of datetime
        Monthly time axis, length 12 × n_years, ordered year-major
        (Jan–Dec of 1978, then Jan–Dec of 1979, …).
    """
    if current_year is None:
        current_year = datetime.now().year

    # 1. Load
    siextent, yearrange = load_nsidc_sie(file_path, current_year, variable=variable)
    nn = len(yearrange)

    # 2. Flatten year-major (all months of each year together), interpolate,
    #    then reshape back to (12, nn)
    sie_flat = siextent.flatten('F')            # (12*nn,)  year-major
    sie_flat = _interpolate_dec87_jan88(sie_flat)
    sie_monthly = sie_flat.reshape(nn, 12).T    # (12, nn)

    # 3. Build yearmon (12, nn) and time list
    years_rep  = np.repeat(yearrange, 12).astype(int)   # 1978×12, 1979×12, …
    months_rep = np.tile(np.arange(1, 13), nn).astype(int)
    time = [datetime(y, m, 1) for y, m in zip(years_rep, months_rep)]
    yearmon = years_rep.reshape(nn, 12).T   # (12, nn)

    # 4. Mask incomplete boundary months
    data = _mask_boundary_nans(sie_monthly)

    return data, yearmon, time


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _interpolate_dec87_jan88(sie: np.ndarray) -> np.ndarray:
    """
    Fill the Dec 1987 / Jan 1988 gap using a quadratic fit over the
    surrounding 7-month window (Sep 1987 – Mar 1988, flat indices 116–122).

    The two missing values are at flat indices 119 (Dec 1987) and 120 (Jan 1988)
    within a year-major (12 × n_years) flattened array starting at 1978.
    """
    sie = sie.copy()
    window_idx  = np.arange(1, 8)          # 1-based position within window
    window_vals = sie[116:123]             # 7 surrounding values

    valid = ~np.isnan(window_vals)
    x = window_idx[valid]
    y = window_vals[valid]

    if len(x) >= 3:
        coeffs = np.polyfit(x, y, 2)
        poly   = np.poly1d(coeffs)
        sie[119] = poly(window_idx[3])     # Dec 1987
        sie[120] = poly(window_idx[4])     # Jan 1988

    return sie


def _mask_boundary_nans(sie_monthly: np.ndarray) -> np.ndarray:
    """
    Set incomplete months to NaN:
      - 1978 (column 0): Jan–Oct have no SMMR data
      - current year (column -1): Mar–Dec are not yet available
    """
    data = sie_monthly.copy()
    for m in range(12):
        if _MASK_1978[m]:
            data[m, 0] = np.nan
        if _MASK_CUR[m]:
            data[m, -1] = np.nan
    return data
