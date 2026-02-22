#!/usr/bin/env python3
"""
Example: Calculate Sea Ice Metrics and Regrid AICE

This script demonstrates how to:
1. Calculate sea ice extent (15% concentration threshold)
2. Calculate sea ice area (concentration-weighted)
3. Regrid AICE to SST grid for comparison
4. Batch process all monthly files

Author: Lauren Hoffman
Email: lhoffma2@ucsc.edu
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cesm2le import (
    calculate_sea_ice_extent,
    calculate_sea_ice_area,
    regrid_aice_to_sst,
    batch_process_monthly_files
)


def example_calculate_extent():
    """
    Example: Calculate sea ice extent for a single month.

    Sea ice extent uses 15% concentration threshold.
    """
    print("\n" + "="*70)
    print("Example 1: Calculate sea ice extent for January")
    print("="*70 + "\n")

    extent = calculate_sea_ice_extent(
        aice_file='/cofast/lhoffman/cesmle/aice/mon/aice_cesmle_first50members_mon_JAN_199001-210012.nc',
        tarea_file='/cofast/lhoffman/cesmle/grid/grid_file.nc',
        save_output='/cofast/lhoffman/cesmle/metrics/siextentn_first50_JAN.nc'
    )

    print(f"January sea ice extent statistics:")
    print(f"  Mean: {extent.mean():.2f} million km²")
    print(f"  Std:  {extent.std():.2f} million km²")
    print(f"  Min:  {extent.min():.2f} million km²")
    print(f"  Max:  {extent.max():.2f} million km²")
    print(f"\n✓ Complete! Extent saved.")


def example_calculate_area():
    """
    Example: Calculate sea ice area for a single month.

    Sea ice area is concentration-weighted (no threshold).
    """
    print("\n" + "="*70)
    print("Example 2: Calculate sea ice area for January")
    print("="*70 + "\n")

    area = calculate_sea_ice_area(
        aice_file='/cofast/lhoffman/cesmle/aice/mon/aice_cesmle_first50members_mon_JAN_199001-210012.nc',
        tarea_file='/cofast/lhoffman/cesmle/grid/grid_file.nc',
        save_output='/cofast/lhoffman/cesmle/metrics/siarean_first50_JAN.nc'
    )

    print(f"January sea ice area statistics:")
    print(f"  Mean: {area.mean():.2f} million km²")
    print(f"  Std:  {area.std():.2f} million km²")
    print(f"  Min:  {area.min():.2f} million km²")
    print(f"  Max:  {area.max():.2f} million km²")
    print(f"\n✓ Complete! Area saved.")


def example_compare_extent_vs_area():
    """
    Example: Compare extent vs area for the same month.

    Extent is typically larger than area because it counts all grid cells
    with >15% ice as fully ice-covered.
    """
    print("\n" + "="*70)
    print("Example 3: Compare extent vs area")
    print("="*70 + "\n")

    aice_file = '/cofast/lhoffman/cesmle/aice/mon/aice_cesmle_first50members_mon_JAN_199001-210012.nc'
    tarea_file = '/cofast/lhoffman/cesmle/grid/grid_file.nc'

    extent = calculate_sea_ice_extent(aice_file, tarea_file)
    area = calculate_sea_ice_area(aice_file, tarea_file)

    print("January statistics:")
    print(f"  Extent mean: {extent.mean():.2f} million km²")
    print(f"  Area mean:   {area.mean():.2f} million km²")
    print(f"  Difference:  {(extent.mean() - area.mean()):.2f} million km²")
    print(f"  Ratio:       {(extent.mean() / area.mean()):.2f}")
    print(f"\n✓ Extent is typically 10-20% larger than area")


def example_regrid_aice():
    """
    Example: Regrid AICE to SST grid for direct comparison.

    AICE is on CICE grid, SST is on POP ocean grid.
    This allows pixel-by-pixel comparison.
    """
    print("\n" + "="*70)
    print("Example 4: Regrid AICE to SST grid")
    print("="*70 + "\n")

    regrid_aice_to_sst(
        aice_file='/cofast/lhoffman/cesmle/aice/mon/aice_cesmle_first50members_mon_JAN_199001-210012.nc',
        sst_grid_file='/cofast/lhoffman/cesmle/sst/mon/sst_cesmle_first50members_mon_JAN_199001-210012.nc',
        output_file='/cofast/lhoffman/cesmle/aice/mon_regridded/aice_regridded_first50_JAN_199001-210012.nc',
        method='linear'  # or 'nearest'
    )

    print("\n✓ Complete! Regridded AICE can now be compared with SST on same grid")


def example_batch_process():
    """
    Example: Process all 12 months at once.

    This calculates extent and area for all monthly files.
    """
    print("\n" + "="*70)
    print("Example 5: Batch process all months")
    print("="*70 + "\n")

    batch_process_monthly_files(
        aice_dir='/cofast/lhoffman/cesmle/aice/mon',
        tarea_file='/cofast/lhoffman/cesmle/grid/grid_file.nc',
        output_dir='/cofast/lhoffman/cesmle/metrics',
        member_label='first50members',
        calculate_extent=True,
        calculate_area=True
    )

    print("\n✓ Complete! All 12 months processed")


def example_seasonal_analysis():
    """
    Example: Calculate metrics for specific seasons only.

    E.g., just winter months (DJF) or summer months (JJA).
    """
    print("\n" + "="*70)
    print("Example 6: Seasonal analysis (winter months)")
    print("="*70 + "\n")

    winter_months = ['DEC', 'JAN', 'FEB']

    batch_process_monthly_files(
        aice_dir='/cofast/lhoffman/cesmle/aice/mon',
        tarea_file='/cofast/lhoffman/cesmle/grid/grid_file.nc',
        output_dir='/cofast/lhoffman/cesmle/metrics/winter',
        member_label='first50members',
        months=winter_months,
        calculate_extent=True,
        calculate_area=True
    )

    print("\n✓ Complete! Winter months processed")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("Sea Ice Metrics Examples")
    print("="*70)
    print("\nThis script demonstrates sea ice extent, area, and regridding")
    print("\nTo run an example, uncomment one of the function calls below:")
    print()

    # Uncomment one of these to run:
    # example_calculate_extent()        # Calculate extent for one month
    # example_calculate_area()          # Calculate area for one month
    # example_compare_extent_vs_area()  # Compare extent vs area
    # example_regrid_aice()             # Regrid AICE to SST grid
    # example_batch_process()           # Process all 12 months
    # example_seasonal_analysis()       # Process specific seasons

    print("\n" + "="*70)
    print("Quick Usage:")
    print("="*70)
    print("""
# Calculate sea ice extent
from slowdown.data.cesm2le import calculate_sea_ice_extent

extent = calculate_sea_ice_extent(
    aice_file='/path/to/aice_JAN.nc',
    tarea_file='/path/to/grid.nc',
    save_output='/path/to/extent_JAN.nc'
)

# Calculate sea ice area
from slowdown.data.cesm2le import calculate_sea_ice_area

area = calculate_sea_ice_area(
    aice_file='/path/to/aice_JAN.nc',
    tarea_file='/path/to/grid.nc',
    save_output='/path/to/area_JAN.nc'
)

# Regrid AICE to SST grid
from slowdown.data.cesm2le import regrid_aice_to_sst

regrid_aice_to_sst(
    aice_file='/path/to/aice_JAN.nc',
    sst_grid_file='/path/to/sst_JAN.nc',
    output_file='/path/to/aice_regridded_JAN.nc'
)

# Batch process all months
from slowdown.data.cesm2le import batch_process_monthly_files

batch_process_monthly_files(
    aice_dir='/path/to/aice/mon',
    tarea_file='/path/to/grid.nc',
    output_dir='/path/to/metrics'
)
    """)
    print("="*70 + "\n")
