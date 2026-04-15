"""
CESM2 Large Ensemble (CESM2-LE) Data Processing Package

This package provides tools for downloading, processing, and analyzing CESM2-LE data.

Modules:
- download:        Download raw data from UCAR repository
- combine:         Combine raw chunks and separate by month
- metrics:         Calculate sea ice extent and area
- regrid:          Regrid AICE to SST grid
- climate_indices: Compute ENSO (Niño3.4, CP/TP) and IPO indices
- slowdowns:       Classify decadal trends as slowdowns or RILES events

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
    process_cesmle_variable,
    calculate_annual_mean,
)

from .metrics import (
    calculate_sea_ice_extent,
    calculate_sea_ice_area,
    batch_process_monthly_files
)

from .regrid import (
    regrid_aice_to_sst
)

from .climate_indices import (
    load_grid_latlon,
    compute_nino34_index,
    compute_enso_cp_tp_indices,
    compute_ipo_index,
    compute_arctic_sst_index,
    compute_arctic_sst_forced_em,
    chebyshev_lowpass,
    enso_phase_labels_ensemble,
    save_nino34,
    save_enso_cp_tp,
    save_ipo,
    save_arctic_sst,
    save_arctic_sst_forced_em,
)

from .slowdowns import (
    load_nsidc_slowdown_thresholds,
    load_sie_monthly_files,
    compute_decadal_trends_ensemble,
    compute_model_thresholds,
    classify_slowdowns,
    classify_riles,
    save_slowdown_events,
    compute_cesm2le_slowdowns,
)

__all__ = [
    # Download functions
    'download_raw_data',
    'add_variable_config',
    # Combine functions
    'combine_ensemble_members',
    'separate_by_month',
    'process_cesmle_variable',
    'calculate_annual_mean',
    # Metrics functions
    'calculate_sea_ice_extent',
    'calculate_sea_ice_area',
    'batch_process_monthly_files',
    # Regrid functions
    'regrid_aice_to_sst',
    # Climate index functions
    'load_grid_latlon',
    'compute_nino34_index',
    'compute_enso_cp_tp_indices',
    'compute_ipo_index',
    'chebyshev_lowpass',
    'enso_phase_labels_ensemble',
    'save_nino34',
    'save_enso_cp_tp',
    'save_ipo',
    'compute_arctic_sst_index',
    'save_arctic_sst',
    'compute_arctic_sst_forced_em',
    'save_arctic_sst_forced_em',    
    # Slowdown / RILES functions
    'load_nsidc_slowdown_thresholds',
    'load_sie_monthly_files',
    'compute_decadal_trends_ensemble',
    'compute_model_thresholds',
    'classify_slowdowns',
    'classify_riles',
    'save_slowdown_events',
    'compute_cesm2le_slowdowns',
    # Constants
    'CMIP6_MEMBERS',
    'SMBB_MEMBERS',
    'VARIABLE_CONFIG',
    'BASE_URL'
]
