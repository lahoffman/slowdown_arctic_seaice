#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 1: Sea Ice Extent Slowdown Analysis (REFACTORED)

This is a refactored version of F1_FS1_FS2_siextent_slowdown.py
demonstrating the new modular structure.

Key improvements:
- Uses centralized config for paths
- Imports functions from modular package
- Cleaner organization
- Easier to maintain and test
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

# Import from modular package
from src import config
from src.utils import load_netcdf, movmean
from src.figures import (
    setup_figure_style,
    moving_decadal_trend,
    classify_slowdown,
    plot_colored_decadal_segments,
    add_panel_label,
    create_multi_panel_figure,
    save_publication_figure,
    COLORS
)


def load_nsidc_monthly_sie(file_path, current_year=2025):
    """
    Load NSIDC monthly sea ice extent data.

    Parameters
    ----------
    file_path : str or Path
        Path to NSIDC Excel file
    current_year : int, optional
        Current year for data extraction

    Returns
    -------
    tuple
        (sie_monthly, yearmon) arrays
    """
    months_sheet = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    string_tail = "-NH"

    yearrange = np.arange(1978, current_year + 1)
    nn = current_year - 1978 + 1

    siextenti = []
    timei_all = []

    for i in range(12):
        sie = np.full((nn, 1), np.nan)
        years = np.full((nn, 1), np.nan)

        sheet = months_sheet[i] + string_tail
        df = pd.read_excel(file_path, sheet_name=sheet)

        timei = np.array(df[["Unnamed: 1"]])[9:]
        siei = np.array(df[["Unnamed: 5"]])[9:]

        start = 1979 if i < 10 else 1978
        end = current_year - 1 if i >= 2 else current_year

        start_idx = np.where(yearrange == start)[0][0]
        end_idx = np.where(yearrange == end)[0][0]

        sie[start_idx : end_idx + 1] = siei
        years[start_idx : end_idx + 1] = timei

        siextenti.append(sie)
        timei_all.append(years)

    siextent = np.reshape(np.array(siextenti), (12, nn))
    timeall = np.reshape(np.array(timei_all), (12, nn))

    yearsf = timeall.flatten("F")
    monthsf = np.transpose(np.tile(np.arange(1, 13), (1, nn)))
    ny = yearsf.shape[0]

    years = np.reshape(yearsf.astype(int), (ny,))
    years[:10] = 1978
    years[ny - 12:] = current_year
    months = np.reshape(monthsf.astype(int), (ny,))

    # Monthly series per calendar month
    sie_flat = siextent.flatten("F")
    monthly = []
    for k in range(1, 13):
        monthly.append(sie_flat[months == k])
    sie_monthly = np.array(monthly)

    # Year grid
    yearmon = np.reshape(years, (nn, 12)).T

    # Interpolation patch for Dec 1987 & Jan 1988
    sie = sie_flat.copy()
    if sie.size > 121:
        tint = np.arange(1, 8)
        sieint = sie[116:123]
        dx = tint.copy().astype(float)
        dy = sieint.copy().astype(float)
        dx2 = tint.copy().astype(float)

        msk = np.isfinite(dy) & np.isfinite(dx)
        if msk.sum() >= 3:
            coeff = np.polyfit(dx[msk], dy[msk], 2)
            poly = np.poly1d(coeff)
            y_fit = poly(dx2)
            sie[119] = y_fit[3]
            sie[120] = y_fit[4]

        monthly = []
        for k in range(1, 13):
            monthly.append(sie[months == k])
        sie_monthly = np.array(monthly)

    return sie_monthly, yearmon


def compute_nsidc_trends_and_thresholds(sie_monthly, yearmon):
    """
    Compute moving decadal trends and slowdown thresholds.

    Parameters
    ----------
    sie_monthly : ndarray
        Monthly SIE data (12, n_years)
    yearmon : ndarray
        Year array (12, n_years)

    Returns
    -------
    tuple
        (linear_trends, years_for_trends, threshold_slowdown, ...)
    """
    data = sie_monthly.copy()

    # NaN handling
    i98 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    i25 = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    for j in range(12):
        if i98[j] == 1:
            data[j, 0] = np.nan
        if i25[j] == 1:
            data[j, -1] = np.nan

    # Compute trends per month
    slopes = []
    for m in range(12):
        slopes.append(moving_decadal_trend(data[m, :], window=10))
    linear_trends = np.array(slopes)

    # Slice to start at year offset
    linear_trends = linear_trends[:, 12:]
    nt = linear_trends.shape[1]

    mean_trend_obs = np.nanmean(linear_trends, axis=1)
    std_trend_obs = np.nanstd(linear_trends, axis=1)

    trend_obs_threshold_slowdown = mean_trend_obs + std_trend_obs
    fraction_obs_threshold_slowdown = trend_obs_threshold_slowdown / mean_trend_obs

    years_for_trends = yearmon[:, 12:-10]

    return (
        linear_trends,
        years_for_trends,
        trend_obs_threshold_slowdown,
        mean_trend_obs,
        std_trend_obs
    )


def main():
    """Main function to generate Figure 1."""

    print("=" * 70)
    print("Generating Figure 1: Sea Ice Extent Slowdown")
    print("=" * 70)

    # Set up styling
    setup_figure_style()

    # =========================================================================
    # 1. LOAD NSIDC DATA
    # =========================================================================
    print("\n[1/4] Loading NSIDC data...")

    nsidc_file = config.ROOT_PATH / "Sea_Ice_Index_Monthly_Data_by_Year_G02135_v3.0.xlsx"
    sie_monthly, yearmon = load_nsidc_monthly_sie(nsidc_file, current_year=2024)

    print(f"  ✓ Loaded NSIDC data: {sie_monthly.shape}")

    # =========================================================================
    # 2. COMPUTE TRENDS
    # =========================================================================
    print("\n[2/4] Computing trends and thresholds...")

    (linear_trends, years_for_trends, threshold_slowdown,
     mean_trend, std_trend) = compute_nsidc_trends_and_thresholds(sie_monthly, yearmon)

    print(f"  ✓ Computed trends for {linear_trends.shape[0]} months")

    # =========================================================================
    # 3. CLASSIFY SLOWDOWNS
    # =========================================================================
    print("\n[3/4] Classifying slowdowns...")

    # September (month index 8)
    sep_trends = linear_trends[8, :]
    sep_threshold = threshold_slowdown[8]
    sep_slowdown_mask = classify_slowdown(sep_trends, sep_threshold)

    n_slowdown = np.sum(sep_slowdown_mask)
    print(f"  ✓ Found {n_slowdown} slowdown periods in September")

    # =========================================================================
    # 4. CREATE FIGURE
    # =========================================================================
    print("\n[4/4] Creating figure...")

    fig, axes = create_multi_panel_figure(nrows=2, ncols=3, figsize=(18, 12))

    # Panel (a): NSIDC September SIE with trends
    ax = axes[0, 0]
    sep_sie = sie_monthly[8, :]
    sep_years = yearmon[8, :]

    ax.plot(sep_years, sep_sie, 'o-', color=COLORS['mean'], label='Sept SIE')

    # Plot colored trend segments
    plot_colored_decadal_segments(
        ax,
        years_for_trends[8, :],
        sep_sie[12:],  # Align with trends
        sep_trends,
        sep_slowdown_mask,
        window=10
    )

    ax.set_xlabel('Year')
    ax.set_ylabel('Sea Ice Extent (million km²)')
    ax.legend()
    add_panel_label(ax, '(a)')

    # Add more panels here following similar pattern...
    # (This is a template - add remaining panels as needed)

    plt.tight_layout()

    # Save figure
    output_file = config.get_figure_file('F1_siextent_slowdown_refactored.png')
    save_publication_figure(fig, output_file, dpi=300)

    print("\n" + "=" * 70)
    print("✓ Complete!")
    print("=" * 70)

    plt.show()


if __name__ == "__main__":
    main()
