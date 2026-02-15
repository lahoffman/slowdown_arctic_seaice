#!/usr/bin/env python3
"""
Example: Download and Process CESM2-LE Data

This script demonstrates how to use the download_cesmle module to:
1. Combine raw CESM2-LE data chunks into full timeseries
2. Separate combined data by month for analysis

Author: Lauren Hoffman
Email: lhoffma2@ucsc.edu
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cesm2le import (
    download_raw_data,
    combine_ensemble_members,
    separate_by_month,
    process_cesmle_variable,
    add_variable_config
)


def example_download_sst():
    """
    Example: Download raw SST data from UCAR.

    This is the first step - downloading the raw data files.
    """
    print("\n" + "="*70)
    print("Example 0: Download raw SST data from UCAR")
    print("="*70 + "\n")

    # Download SST data for first 50 members
    download_raw_data(
        variable='SST',
        output_dir='/cofast/lhoffman/cesmle/sst/raw',
        member_groups=['first50']  # Start with just first50 to test
    )

    print("\n✓ Complete! Raw SST data downloaded")


def example_download_and_process():
    """
    Example: Complete workflow - download and process in one go.

    This is the recommended approach for a complete analysis.
    """
    print("\n" + "="*70)
    print("Example 0b: Complete download and processing workflow")
    print("="*70 + "\n")

    # Step 1: Download raw data
    print("Step 1: Downloading raw data from UCAR...")
    download_raw_data(
        variable='SST',
        output_dir='/cofast/lhoffman/cesmle/sst/raw',
        member_groups=['first50', 'last50']
    )

    # Step 2: Process the data
    print("\nStep 2: Processing downloaded data...")
    process_cesmle_variable(
        variable='SST',
        raw_data_path='/cofast/lhoffman/cesmle/sst/raw',
        combined_output_path='/cofast/lhoffman/cesmle/sst/mon_combined/sst_cesmle_{group}members_mon_199001-210012.nc',
        monthly_output_dir='/cofast/lhoffman/cesmle/sst/mon',
        member_groups=['first50', 'last50']
    )

    print("\n✓ Complete! SST data downloaded and processed")


def example_sst_processing():
    """
    Example: Process SST data for first 50 ensemble members.

    This demonstrates the step-by-step approach.
    """
    print("\n" + "="*70)
    print("Example 1: Step-by-step SST processing")
    print("="*70 + "\n")

    # Define paths
    raw_data_path = '/cofast/lhoffman/cesmle/sst/raw'
    combined_output = '/cofast/lhoffman/cesmle/sst/mon_combined/sst_cesmle_first50members_mon_199001-210012.nc'
    monthly_output_dir = '/cofast/lhoffman/cesmle/sst/mon'

    # Step 1: Combine ensemble members
    print("Step 1: Combining raw data chunks...")
    combine_ensemble_members(
        variable='SST',
        raw_data_path=raw_data_path,
        output_path=combined_output,
        member_group='first50'
    )

    # Step 2: Separate by month
    print("\nStep 2: Separating by month...")
    separate_by_month(
        combined_file=combined_output,
        output_dir=monthly_output_dir,
        variable='SST',
        member_label='first50members'
    )

    print("\n✓ Complete! SST data processed for first 50 members")


def example_complete_pipeline():
    """
    Example: Use the complete pipeline function for both member groups.

    This is the recommended approach - it handles everything in one call.
    """
    print("\n" + "="*70)
    print("Example 2: Complete pipeline for SST (both member groups)")
    print("="*70 + "\n")

    process_cesmle_variable(
        variable='SST',
        raw_data_path='/cofast/lhoffman/cesmle/sst/raw',
        combined_output_path='/cofast/lhoffman/cesmle/sst/mon_combined/sst_cesmle_{group}members_mon_199001-210012.nc',
        monthly_output_dir='/cofast/lhoffman/cesmle/sst/mon',
        member_groups=['first50', 'last50']
    )

    print("\n✓ Complete! SST data processed for all 100 members")


def example_aice_processing():
    """
    Example: Process sea ice concentration (AICE) data.

    The module works with any CESM2 variable.
    """
    print("\n" + "="*70)
    print("Example 3: Process AICE (sea ice concentration)")
    print("="*70 + "\n")

    process_cesmle_variable(
        variable='AICE',
        raw_data_path='/cofast/lhoffman/cesmle/aice/raw',
        combined_output_path='/cofast/lhoffman/cesmle/aice/mon_combined/aice_cesmle_{group}members_mon_199001-210012.nc',
        monthly_output_dir='/cofast/lhoffman/cesmle/aice/mon',
        member_groups=['first50', 'last50']
    )

    print("\n✓ Complete! AICE data processed for all 100 members")


def example_custom_time_range():
    """
    Example: Process data with custom time range.

    You can customize the time periods if your data has different ranges.
    """
    print("\n" + "="*70)
    print("Example 4: Custom time range")
    print("="*70 + "\n")

    # Custom time periods (example: only historical period)
    start_years = ['1990', '2000', '2010']
    end_years = ['1999', '2009', '2014']

    combine_ensemble_members(
        variable='SST',
        raw_data_path='/cofast/lhoffman/cesmle/sst/raw',
        output_path='/cofast/lhoffman/cesmle/sst/mon_combined/sst_first50_199001-201412.nc',
        start_years=start_years,
        end_years=end_years,
        hist_cutoff_idx=3,  # All chunks are historical
        member_group='first50'
    )

    print("\n✓ Complete! Custom time range processed")


def example_custom_variable():
    """
    Example: Download and process a custom variable not in the default config.

    This shows how to work with any CESM2 variable.
    """
    print("\n" + "="*70)
    print("Example 5: Custom variable (UBOT - zonal wind)")
    print("="*70 + "\n")

    # Add configuration for custom variable
    add_variable_config(
        variable='UBOT',
        component='cam.h0',
        url_path='atm/proc/tseries/month_1/UBOT'
    )

    # Download the data
    download_raw_data(
        variable='UBOT',
        output_dir='/cofast/lhoffman/cesmle/ubot/raw',
        member_groups=['first50']
    )

    print("\n✓ Complete! Custom variable downloaded")


def example_dry_run():
    """
    Example: Use dry_run to see what would be downloaded without actually downloading.

    Useful for checking URLs and file availability before starting a large download.
    """
    print("\n" + "="*70)
    print("Example 6: Dry run (see what would be downloaded)")
    print("="*70 + "\n")

    # Dry run - shows commands without executing
    download_raw_data(
        variable='SST',
        output_dir='/cofast/lhoffman/cesmle/sst/raw',
        member_groups=['first50'],
        dry_run=True  # Only print, don't download
    )

    print("\n✓ Dry run complete! No files were downloaded.")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("CESM2-LE Data Download and Processing Examples")
    print("="*70)
    print("\nThis script contains several examples of how to use download_cesmle.py")
    print("\nTo run an example, uncomment one of the function calls below:")
    print()

    # Uncomment one of these to run:
    # example_download_sst()             # Download raw data from UCAR
    # example_download_and_process()     # Complete workflow (recommended)
    # example_sst_processing()           # Step-by-step processing only
    # example_complete_pipeline()        # Pipeline for processing existing data
    # example_aice_processing()          # Different variable
    # example_custom_time_range()        # Custom time periods
    # example_custom_variable()          # Add and download custom variables
    # example_dry_run()                  # Test without downloading

    print("\n" + "="*70)
    print("Quick Usage:")
    print("="*70)
    print("""
# Complete workflow - download and process
from src.data.download_cesmle import download_raw_data, process_cesmle_variable

# Step 1: Download raw data from UCAR
download_raw_data(
    variable='SST',
    output_dir='/path/to/raw/data',
    member_groups=['first50', 'last50']
)

# Step 2: Process the downloaded data
process_cesmle_variable(
    variable='SST',
    raw_data_path='/path/to/raw/data',
    combined_output_path='/path/to/combined/output_{group}_199001-210012.nc',
    monthly_output_dir='/path/to/monthly/output',
    member_groups=['first50', 'last50']
)
    """)
    print("="*70 + "\n")
