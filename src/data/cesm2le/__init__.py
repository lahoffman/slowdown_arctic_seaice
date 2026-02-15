"""
CESM2 Large Ensemble (CESM2-LE) Data Processing Package

This package provides tools for downloading, processing, and analyzing CESM2-LE data.

Modules:
- download: Download raw data from UCAR repository
- combine: Combine raw chunks and separate by month
- metrics: Calculate sea ice extent and area
- regrid: Regrid AICE to SST grid

Author: Lauren Hoffman
Email: lhoffma2@ucsc.edu
"""

from .download import (
    download_raw_data,
    add_variable_config,
    CMIP6_MEMBERS,
    SMBB_MEMBERS,
    VARIABLE_CONFIG,
    BASE_URL
)

from .combine import (
    combine_ensemble_members,
    separate_by_month,
    process_cesmle_variable
)

from .metrics import (
    calculate_sea_ice_extent,
    calculate_sea_ice_area,
    batch_process_monthly_files
)

from .regrid import (
    regrid_aice_to_sst
)

__all__ = [
    # Download functions
    'download_raw_data',
    'add_variable_config',
    # Combine functions
    'combine_ensemble_members',
    'separate_by_month',
    'process_cesmle_variable',
    # Metrics functions
    'calculate_sea_ice_extent',
    'calculate_sea_ice_area',
    'batch_process_monthly_files',
    # Regrid functions
    'regrid_aice_to_sst',
    # Constants
    'CMIP6_MEMBERS',
    'SMBB_MEMBERS',
    'VARIABLE_CONFIG',
    'BASE_URL'
]
