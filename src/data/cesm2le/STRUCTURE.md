# CESM2-LE Package Structure

This subdirectory contains all CESM2-LE specific data processing code, organized by functionality.

## Module Organization

```
src/data/cesm2le/
├── __init__.py          # Package initialization, exports all functions
├── download.py          # Download raw data from UCAR
├── combine.py           # Combine chunks and separate by month
├── metrics.py           # Calculate sea ice extent and area
├── regrid.py            # Regrid AICE to SST grid
├── README.md            # Complete documentation
└── STRUCTURE.md         # This file
```

## Modules

### `download.py`
**Purpose**: Download raw CESM2-LE data from UCAR repository

**Functions**:
- `download_raw_data()` - Download files using wget
- `add_variable_config()` - Add support for new variables
- `_download_single_file()` - Helper for single file download

**Constants**:
- `CMIP6_MEMBERS` - First 50 ensemble members
- `SMBB_MEMBERS` - Last 50 ensemble members
- `VARIABLE_CONFIG` - Variable metadata (component, url_path)
- `BASE_URL` - UCAR data repository base URL

### `combine.py`
**Purpose**: Combine raw data chunks and organize by month

**Functions**:
- `combine_ensemble_members()` - Combine time chunks into [nens, ntime, nlat, nlon]
- `separate_by_month()` - Split into 12 monthly files [nens, nyears, nlat, nlon]
- `process_cesmle_variable()` - Complete pipeline (combine + separate)

### `metrics.py`
**Purpose**: Calculate sea ice metrics

**Functions**:
- `calculate_sea_ice_extent()` - NH extent with 15% threshold
- `calculate_sea_ice_area()` - NH area (concentration-weighted)
- `batch_process_monthly_files()` - Process all 12 months
- `_save_metric()` - Helper to save timeseries to NetCDF

### `regrid.py`
**Purpose**: Regrid AICE to match SST grid

**Functions**:
- `regrid_aice_to_sst()` - Interpolate AICE from CICE to POP grid
- `_regrid_2d()` - Helper for 2D interpolation

## Usage

### Import from package (recommended)
```python
from src.data.cesm2le import (
    download_raw_data,
    combine_ensemble_members,
    calculate_sea_ice_extent,
    regrid_aice_to_sst
)
```

### Import from specific modules
```python
from src.data.cesm2le.download import download_raw_data, CMIP6_MEMBERS
from src.data.cesm2le.combine import combine_ensemble_members
from src.data.cesm2le.metrics import calculate_sea_ice_extent
from src.data.cesm2le.regrid import regrid_aice_to_sst
```

### Backward compatibility
All functions are also available from `src.data`:
```python
from src.data import download_raw_data, calculate_sea_ice_extent
```

## Design Principles

1. **Focused modules**: Each module has a single, clear purpose
2. **No circular dependencies**: Clean import structure
3. **Dataset-specific**: All CESM2-LE code isolated from other datasets
4. **Scalable**: Easy to add new modules as needed
5. **Well-documented**: Each module has clear docstrings

## Adding New Functionality

To add new CESM2-LE-specific functions:

1. Determine which module it belongs in (or create a new one)
2. Add the function to the appropriate module
3. Export it in `__init__.py`
4. Add to `__all__` list
5. Update this STRUCTURE.md file
6. Add examples to README.md

## Future Structure

As you add observational datasets, the structure will grow:

```
src/data/
├── cesm2le/              # CESM2-LE specific (done!)
│   ├── download.py
│   ├── combine.py
│   ├── metrics.py
│   └── regrid.py
├── observations/         # Observational data (future)
│   ├── ersst/
│   │   ├── download.py
│   │   └── preprocess.py
│   ├── nsidc/
│   │   ├── download.py
│   │   └── preprocess.py
│   └── hadisst/
│       ├── download.py
│       └── preprocess.py
├── climate_indices.py    # Generic utilities (cross-dataset)
├── statistics.py
└── trends.py
```

This structure keeps dataset-specific code organized while maintaining generic utilities at the top level.
