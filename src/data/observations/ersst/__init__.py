"""
NOAA ERSSTv5 Data Package

Modules:
- download:          Download raw ERSSTv5 data and load into arrays
- regrid_to_cesm2le: Regrid ERSSTv5 to CESM2-LE atmospheric grid
- climate_indices:   Compute ENSO (Niño3.4, CP/TP) and IPO indices
"""

from .download import (
    download_ersst,
    load_ersst,
    ERSST_URL,
    ERSST_FILENAME
)

from .regrid_to_cesm2le import (
    load_cesm2le_grid,
    regrid_ersst_to_cesm2le,
    save_regridded_ersst,
    process_ersst_regrid
)

from .climate_indices import (
    compute_nino34_index,
    compute_enso_cp_tp_indices,
    compute_ipo_index,
    compute_arctic_sst_index,
    chebyshev_lowpass,
    enso_phase_labels,
    save_nino34,
    save_enso_cp_tp,
    save_ipo,
    save_arctic_sst,
)

__all__ = [
    # Download
    'download_ersst',
    'load_ersst',
    # Regrid
    'load_cesm2le_grid',
    'regrid_ersst_to_cesm2le',
    'save_regridded_ersst',
    'process_ersst_regrid',
    # Climate indices
    'compute_nino34_index',
    'compute_enso_cp_tp_indices',
    'compute_ipo_index',
    'chebyshev_lowpass',
    'enso_phase_labels',
    'save_nino34',
    'save_enso_cp_tp',
    'save_ipo',
    'compute_arctic_sst_index',
    'save_arctic_sst',
]
