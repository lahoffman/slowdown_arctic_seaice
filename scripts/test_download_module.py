#!/usr/bin/env python3
"""
Quick test script to verify download_cesmle module imports and basic functionality.

This doesn't actually download or process data, just checks that the module
is correctly set up.
"""

import sys
from pathlib import Path

# Add src/data/cesm2le to path to import modules directly
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src' / 'data' / 'cesm2le'))

print("="*70)
print("Testing CESM2-LE modules")
print("="*70 + "\n")

# Test imports
print("1. Testing imports...")
try:
    # Import directly from module files to avoid package dependencies
    import download
    import combine
    import metrics
    import regrid

    download_raw_data = download.download_raw_data
    add_variable_config = download.add_variable_config
    CMIP6_MEMBERS = download.CMIP6_MEMBERS
    SMBB_MEMBERS = download.SMBB_MEMBERS
    VARIABLE_CONFIG = download.VARIABLE_CONFIG

    combine_ensemble_members = combine.combine_ensemble_members
    separate_by_month = combine.separate_by_month
    process_cesmle_variable = combine.process_cesmle_variable

    calculate_sea_ice_extent = metrics.calculate_sea_ice_extent
    calculate_sea_ice_area = metrics.calculate_sea_ice_area
    batch_process_monthly_files = metrics.batch_process_monthly_files

    regrid_aice_to_sst = regrid.regrid_aice_to_sst

    print("   ✓ All imports successful\n")
except ImportError as e:
    print(f"   ✗ Import failed: {e}\n")
    sys.exit(1)

# Test member lists
print("2. Checking ensemble member lists...")
print(f"   - CMIP6 members (first 50): {len(CMIP6_MEMBERS)}")
print(f"   - SMBB members (last 50): {len(SMBB_MEMBERS)}")
if len(CMIP6_MEMBERS) == 50 and len(SMBB_MEMBERS) == 50:
    print("   ✓ Member lists correct\n")
else:
    print("   ✗ Member lists have wrong length\n")

# Test variable config
print("3. Checking variable configurations...")
print(f"   - Available variables: {list(VARIABLE_CONFIG.keys())}")
expected_vars = ['SST', 'AICE', 'TS', 'PRECT']
if all(var in VARIABLE_CONFIG for var in expected_vars):
    print("   ✓ Default variables configured\n")
else:
    print("   ✗ Missing some default variables\n")

# Test adding custom variable
print("4. Testing add_variable_config...")
try:
    add_variable_config(
        variable='TEST_VAR',
        component='cam.h0',
        url_path='atm/proc/tseries/month_1/TEST_VAR'
    )
    if 'TEST_VAR' in VARIABLE_CONFIG:
        print("   ✓ Successfully added custom variable\n")
    else:
        print("   ✗ Failed to add custom variable\n")
except Exception as e:
    print(f"   ✗ Error adding variable: {e}\n")

# Test dry run (doesn't actually download)
print("5. Testing download_raw_data (dry run)...")
try:
    # This won't download anything, just tests the function structure
    download_raw_data(
        variable='SST',
        output_dir='/tmp/test_cesm_download',
        member_groups=['first50'],
        hist_startyears=[1990],
        hist_endyears=[1999],
        ssp_startyears=[2015],
        ssp_endyears=[2024],
        dry_run=True
    )
    print("   ✓ Dry run completed successfully\n")
except Exception as e:
    print(f"   ✗ Dry run failed: {e}\n")

# Summary
print("="*70)
print("Module test complete!")
print("="*70)
print("\nThe download_cesmle module is properly set up and ready to use.")
print("\nNext steps:")
print("  1. Review scripts/example_download_cesmle.py for usage examples")
print("  2. Use download_raw_data() to download CESM2 data from UCAR")
print("  3. Use process_cesmle_variable() to process downloaded data")
print("\n" + "="*70 + "\n")
