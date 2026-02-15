"""
Analysis functions for climate indices and statistical operations.
"""

from .climate_indices import (
    compute_nino34_index,
    compute_ipo_index,
    label_enso_phases
)

from .statistics import (
    compute_anomalies,
    compute_climatology,
    normalize_by_std
)

from .trends import (
    compute_linear_trend,
    remove_linear_trend
)

# CESM2-LE data processing (organized in subdirectory)
from .cesm2le import (
    download_raw_data,
    combine_ensemble_members,
    separate_by_month,
    process_cesmle_variable,
    add_variable_config,
    calculate_sea_ice_extent,
    calculate_sea_ice_area,
    regrid_aice_to_sst,
    batch_process_monthly_files,
    CMIP6_MEMBERS,
    SMBB_MEMBERS,
    VARIABLE_CONFIG
)

__all__ = [
    'compute_nino34_index',
    'compute_ipo_index',
    'label_enso_phases',
    'compute_anomalies',
    'compute_climatology',
    'normalize_by_std',
    'compute_linear_trend',
    'remove_linear_trend',
    'download_raw_data',
    'combine_ensemble_members',
    'separate_by_month',
    'process_cesmle_variable',
    'add_variable_config',
    'calculate_sea_ice_extent',
    'calculate_sea_ice_area',
    'regrid_aice_to_sst',
    'batch_process_monthly_files',
    'CMIP6_MEMBERS',
    'SMBB_MEMBERS',
    'VARIABLE_CONFIG'
]
