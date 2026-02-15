# CESM2-LE Data Download Module

This module provides functions to download and process CESM2 Large Ensemble (CESM2-LE) data.

## Overview

The CESM2-LE dataset comes in chunks organized by:
- **Ensemble member**: 100 total (first 50 are cmip6, last 50 are smbb)
- **Time periods**: Data spanning 1990-2100 with different year ranges
- **Variables**: SST, AICE, TS, etc.

This module handles:
0. **Downloading** raw data from UCAR repository
1. **Combining** raw data chunks into full timeseries `[nens, ntime, nlat, nlon]`
2. **Separating** combined data by month for easier analysis `[nens, nyears, nlat, nlon]`

## Quick Start

### Complete Workflow (Download + Process)

```python
from src.data.download_cesmle import download_raw_data, process_cesmle_variable

# Step 1: Download raw data from UCAR
download_raw_data(
    variable='SST',
    output_dir='/cofast/lhoffman/cesmle/sst/raw',
    member_groups=['first50', 'last50']
)

# Step 2: Process the downloaded data
process_cesmle_variable(
    variable='SST',
    raw_data_path='/cofast/lhoffman/cesmle/sst/raw',
    combined_output_path='/cofast/lhoffman/cesmle/sst/mon_combined/sst_{group}_199001-210012.nc',
    monthly_output_dir='/cofast/lhoffman/cesmle/sst/mon',
    member_groups=['first50', 'last50']
)
```

## Functions

### `download_raw_data()`

Downloads raw CESM2-LE data directly from the UCAR repository using wget.

**Parameters:**
- `variable` (str): Variable name (e.g., 'SST', 'AICE', 'TS')
- `output_dir` (str): Directory where raw files will be saved
- `member_groups` (List[str], optional): Which groups to download (default: ['first50', 'last50'])
- `hist_startyears`, `hist_endyears` (List[int], optional): Custom historical time periods
- `ssp_startyears`, `ssp_endyears` (List[int], optional): Custom future time periods
- `component`, `url_path`, `var_name` (str, optional): Override defaults for custom variables
- `dry_run` (bool, optional): Print commands without executing (default: False)

**Output:**
- Downloads raw NetCDF files from UCAR to output_dir
- Handles file renaming if wget adds .1 suffix
- Skips files that already exist

**Example:**
```python
from src.data.download_cesmle import download_raw_data

# Download SST data for first 50 members
download_raw_data(
    variable='SST',
    output_dir='/cofast/lhoffman/cesmle/sst/raw',
    member_groups=['first50']
)

# Download AICE data for all 100 members
download_raw_data(
    variable='AICE',
    output_dir='/cofast/lhoffman/cesmle/aice/raw',
    member_groups=['first50', 'last50']
)

# Dry run to see what would be downloaded
download_raw_data(
    variable='TS',
    output_dir='/cofast/lhoffman/cesmle/ts/raw',
    dry_run=True
)
```

### `combine_ensemble_members()`

Combines raw CESM2-LE data chunks into a single file.

**Parameters:**
- `variable` (str): Variable name (e.g., 'SST', 'AICE', 'TS')
- `raw_data_path` (str): Base directory where raw CESM2 data is stored
- `output_path` (str): Full path for the output NetCDF file
- `member_group` (str): 'first50', 'last50', or 'all'
- `start_years`, `end_years` (List[str], optional): Custom time periods
- Other optional parameters for customization

**Output:**
- NetCDF file with shape `[nens, ntime, nlat, nlon]`
- `ntime = nyears * 12` (monthly data)

**Example:**
```python
from src.data.download_cesmle import combine_ensemble_members

combine_ensemble_members(
    variable='SST',
    raw_data_path='/cofast/lhoffman/cesmle/sst/raw',
    output_path='/cofast/lhoffman/cesmle/sst/mon_combined/sst_first50_199001-210012.nc',
    member_group='first50'
)
```

### `separate_by_month()`

Separates combined monthly data into individual files for each month.

**Parameters:**
- `combined_file` (str): Path to combined NetCDF file
- `output_dir` (str): Directory where monthly files will be saved
- `variable` (str): Variable name
- `start_year`, `end_year` (int, optional): Time range
- `member_label` (str, optional): Label for output filenames

**Output:**
- 12 NetCDF files (JAN, FEB, ..., DEC) with shape `[nens, nyears, nlat, nlon]`

**Example:**
```python
from src.data.download_cesmle import separate_by_month

separate_by_month(
    combined_file='/cofast/lhoffman/cesmle/sst/mon_combined/sst_first50_199001-210012.nc',
    output_dir='/cofast/lhoffman/cesmle/sst/mon',
    variable='SST',
    member_label='first50members'
)
```

### `process_cesmle_variable()`

Complete pipeline that runs both combine and separate operations.

**Parameters:**
- `variable` (str): Variable name
- `raw_data_path` (str): Base directory for raw data
- `combined_output_path` (str): Path template with `{group}` placeholder
- `monthly_output_dir` (str): Directory for monthly outputs
- `member_groups` (List[str], optional): Which groups to process

**Example:**
```python
from src.data.download_cesmle import process_cesmle_variable

# Process both first50 and last50 members in one call
process_cesmle_variable(
    variable='AICE',
    raw_data_path='/cofast/lhoffman/cesmle/aice/raw',
    combined_output_path='/cofast/lhoffman/cesmle/aice/mon_combined/aice_{group}_199001-210012.nc',
    monthly_output_dir='/cofast/lhoffman/cesmle/aice/mon',
    member_groups=['first50', 'last50']
)
```

## Data Structure

### Input (Raw CESM2 Data)

Raw data comes in files named:
```
b.e21.BHISTcmip6.f09_g17.LE2-{member}.cam.h0.{VAR}.{start_year}01-{end_year}12.nc
b.e21.BSSP370cmip6.f09_g17.LE2-{member}.cam.h0.{VAR}.{start_year}01-{end_year}12.nc
```

Time periods:
- Historical (BHIST): 1990-1999, 2000-2009, 2010-2014
- Future (BSSP370): 2015-2024, 2025-2034, ..., 2095-2100

### Output Structure

**Combined files:**
```
variable_name/
  mon_combined/
    var_first50members_mon_199001-210012.nc    # [50, 1332, nlat, nlon]
    var_last50members_mon_199001-210012.nc     # [50, 1332, nlat, nlon]
```

**Monthly files:**
```
variable_name/
  mon/
    var_cesmle_first50members_mon_JAN_199001-210012.nc   # [50, 111, nlat, nlon]
    var_cesmle_first50members_mon_FEB_199001-210012.nc   # [50, 111, nlat, nlon]
    ...
    var_cesmle_last50members_mon_JAN_199001-210012.nc
    ...
```

## Supported Variables

The module includes pre-configured support for common variables:

```python
from src.data.download_cesmle import VARIABLE_CONFIG

# Pre-configured variables:
# - SST: Sea surface temperature (ocean model: pop.h)
# - AICE: Sea ice concentration (sea ice model: cice.h)
# - TS: Surface temperature (atmosphere model: cam.h0)
# - PRECT: Precipitation rate (atmosphere model: cam.h0)

print(VARIABLE_CONFIG.keys())
```

### Adding Custom Variables

To add support for additional CESM2 variables:

```python
from src.data.download_cesmle import add_variable_config, download_raw_data

# Add configuration for wind component
add_variable_config(
    variable='UBOT',
    component='cam.h0',
    url_path='atm/proc/tseries/month_1/UBOT'
)

# Now you can download it
download_raw_data(
    variable='UBOT',
    output_dir='/cofast/lhoffman/cesmle/ubot/raw'
)
```

## Ensemble Members

The module includes pre-defined member lists:

```python
from src.data.download_cesmle import CMIP6_MEMBERS, SMBB_MEMBERS

print(f"First 50 members: {len(CMIP6_MEMBERS)}")
print(f"Last 50 members: {len(SMBB_MEMBERS)}")
```

## More Examples

See `scripts/example_download_cesmle.py` for additional examples including:
- Step-by-step processing
- Custom time ranges
- Processing different variables
- Working with specific ensemble member groups

## Original Scripts

This module generalizes the functionality from:
- `x_old/data_download/D0.0_sst_download_1990-2100` - Raw data download (bash script)
- `x_old/data_download/D0.2_sst_combine_mon_1990-2100.py` - Combine time chunks
- `x_old/data_download/D0.3_sst_monthly.py` - Separate by month

The modular version:
- Works with any CESM2 variable (not just SST/AICE)
- Implements downloads in Python (no separate bash scripts needed)
- Provides more flexibility and easier maintenance
- Includes dry-run mode and better error handling
- Skips already-downloaded files automatically
