# Arctic Sea Ice Slowdown - Modular Structure

## 🎯 Overview

This repository has been refactored to use a **modular structure** that makes the code:
- ✅ **Maintainable**: Change paths in one place, not everywhere
- ✅ **Reusable**: Functions can be used across multiple scripts
- ✅ **Readable**: Clear organization and documentation
- ✅ **Scalable**: Easy to add new functionality

## 📁 New Directory Structure

```
slowdown_arctic_seaice/
├── src/                          # Main package
│   ├── __init__.py
│   ├── config.py                 # ⭐ Centralized configuration (paths, parameters)
│   ├── utils/                    # Utility functions
│   │   ├── __init__.py
│   │   ├── data_io.py           # Load/save NetCDF files
│   │   ├── grid_ops.py          # Regridding and spatial operations
│   │   └── general.py           # General utilities (movmean, ncdisp, etc.)
│   ├── analysis/                 # Analysis functions
│   │   ├── __init__.py
│   │   ├── climate_indices.py   # ENSO, IPO index calculations
│   │   ├── statistics.py        # Statistical functions (anomalies, trends)
│   │   └── trends.py            # Trend analysis
│   └── plotting/                 # Plotting functions
│       ├── __init__.py
│       ├── maps.py              # Map plotting
│       └── timeseries.py        # Time series plots
├── examples/                     # Example scripts showing new structure
│   └── example_refactored_enso_index.py
├── data_download/                # Original scripts (to be refactored)
├── data_processing/              # Original scripts (to be refactored)
├── figures/                      # Original scripts (to be refactored)
├── setup.py                      # Package installation
└── README.md                     # This file
```

## 🚀 Getting Started

### 1. Install the Package

Install in development mode (recommended for active development):

```bash
cd /path/to/slowdown_arctic_seaice
pip install -e .
```

This allows you to edit the source code and have changes immediately available.

### 2. Configure Paths

**THIS IS THE KEY IMPROVEMENT!** Edit `src/config.py` to set your paths:

```python
# In src/config.py, change this line:
ROOT_PATH = Path('/cofast/lhoffman/slowdown/')

# To your local path:
ROOT_PATH = Path('/Users/lahoffma/projects/arcticWATCH/slowdown_arctic_seaice')
```

That's it! No more hunting through dozens of scripts to update paths.

### 3. Use the New Structure

#### Example: Before (Old Way)

```python
# OLD: D1_OBS_SST_regridded_ENSO_index.py
# 270 lines of code with hardcoded paths and inline functions

rootpath = '/cofast/lhoffman/slowdown/'  # ❌ Hardcoded

# 50+ lines of imports...
import sys
import os
import numpy as np
# ... many more ...

# Define function inline
def compute_nino34_index(sst_obs, lat, lon, years, baseline=(1990,2020)):
    # 50+ lines of code...
    pass

# 100+ lines of procedural code to load data, compute index, save...
```

#### Example: After (New Way)

```python
# NEW: examples/example_refactored_enso_index.py
# 50 lines of clean, readable code

from src import config  # ✅ Centralized config
from src.utils import load_cesm2_grid, save_netcdf
from src.analysis import compute_nino34_index, label_enso_phases
from src.plotting import plot_enso_phases

# Load data
lat, lon = load_cesm2_grid()
sst_obs = load_netcdf(config.ERSST_REGRIDDED, variables=['sst_obs'])

# Compute index (one line!)
nino34 = compute_nino34_index(sst_obs, lat, lon, years)

# Label phases
labels = label_enso_phases(nino34)

# Save
save_netcdf(data_dict, coords_dict, config.get_output_file('nino34.nc'))

# Plot
plot_enso_phases(dates, nino34, labels, save_path='figure.png')
```

## 📦 Key Modules

### `src/config.py` - Configuration

**Most important file!** Contains all paths and parameters:

```python
from src import config

# Paths
config.ROOT_PATH           # Base directory
config.DATA_PATH           # Data directory
config.CESM2_GRID_FILE     # Common data files
config.ERSST_REGRIDDED

# Parameters
config.BASELINE_PERIOD     # (1990, 2020)
config.NINO34_THRESHOLD    # 0.4
config.NINO34_REGION       # {'lat_min': -5, 'lat_max': 5, ...}

# Helper functions
config.get_output_file('myfile.nc')   # Returns ROOT_PATH / 'output' / 'myfile.nc'
config.get_figure_file('fig.png')     # Returns ROOT_PATH / 'figures' / 'fig.png'
```

### `src/utils/` - Utilities

**Data I/O:**
```python
from src.utils import load_netcdf, save_netcdf, load_cesm2_grid, load_landmask

# Load specific variables
data = load_netcdf('file.nc', variables=['sst', 'lat', 'lon'])

# Save with compression
save_netcdf(data_dict, coords_dict, 'output.nc')

# Load common datasets
lat, lon = load_cesm2_grid()
landmask = load_landmask()
```

**Grid Operations:**
```python
from src.utils import regrid_to_cesm2, subset_region

# Regrid to CESM2 grid
sst_regridded = regrid_to_cesm2(sst, lat_src, lon_src, lat_cesm, lon_cesm)

# Subset region
data_subset, lat_sub, lon_sub, _, _ = subset_region(
    data, lat, lon, lat_min=-5, lat_max=5, lon_min=190, lon_max=240
)
```

**General Utilities:**
```python
from src.utils import movmean, ncdisp

# Moving average
smoothed = movmean(data, window=5, axis=0)

# Display NetCDF info
ncdisp(dataset)
```

### `src/analysis/` - Analysis

**Climate Indices:**
```python
from src.analysis import compute_nino34_index, label_enso_phases

# Compute Niño 3.4 index
nino34 = compute_nino34_index(sst, lat, lon, years)

# Label ENSO phases
labels = label_enso_phases(nino34)  # Returns 1 (El Niño), -1 (La Niña), 0 (Neutral)
```

**Statistics:**
```python
from src.analysis import compute_anomalies, compute_climatology, normalize_by_std

# Compute monthly climatology
climatology = compute_climatology(data, years, months, baseline=(1990, 2020))

# Compute anomalies
anomalies = compute_anomalies(data, years, months)

# Normalize
normalized = normalize_by_std(data)
```

**Trends:**
```python
from src.analysis import compute_linear_trend, remove_linear_trend

# Compute trend
slope, intercept = compute_linear_trend(data, axis=0)

# Remove trend
detrended, trend = remove_linear_trend(data, axis=0)
```

### `src/plotting/` - Plotting

**Maps:**
```python
from src.plotting import plot_map, plot_comparison_maps

# Single map
fig, ax = plot_map(data, lat, lon, title='SST', save_path='map.png')

# Comparison maps
datasets = [
    {'data': data1, 'lat': lat1, 'lon': lon1},
    {'data': data2, 'lat': lat2, 'lon': lon2}
]
plot_comparison_maps(datasets, titles=['Original', 'Regridded'])
```

**Time Series:**
```python
from src.plotting import plot_timeseries, plot_enso_phases

# Simple time series
plot_timeseries(time, data, title='My Data')

# ENSO phases with shading
plot_enso_phases(dates, nino34, labels, save_path='enso.png')
```

## 🔄 Migration Guide

To refactor your existing scripts:

### Step 1: Replace Path Definitions

**Before:**
```python
rootpath = '/cofast/lhoffman/slowdown/'
filepath = rootpath + 'data/myfile.nc'
```

**After:**
```python
from src import config
filepath = config.get_data_file('myfile.nc')
```

### Step 2: Replace Inline Functions

**Before:**
```python
def compute_nino34_index(sst, lat, lon, years):
    # 50 lines of code...
    pass

nino34 = compute_nino34_index(sst, lat, lon, years)
```

**After:**
```python
from src.analysis import compute_nino34_index

nino34 = compute_nino34_index(sst, lat, lon, years)
```

### Step 3: Use Utility Functions

**Before:**
```python
dataset = nc.Dataset(filepath, 'r')
lat = np.array(dataset.variables['lat'])
lon = np.array(dataset.variables['lon'])
sst = np.array(dataset.variables['sst'])
```

**After:**
```python
from src.utils import load_netcdf

data = load_netcdf(filepath, variables=['lat', 'lon', 'sst'])
lat, lon, sst = data['lat'], data['lon'], data['sst']
```

## 🧪 Testing

Run the example script to test the new structure:

```bash
python examples/example_refactored_enso_index.py
```

## 💡 Benefits

### Before Refactoring
- ❌ 270 lines per script
- ❌ Paths hardcoded everywhere
- ❌ Functions duplicated across files
- ❌ Hard to maintain
- ❌ Difficult to test

### After Refactoring
- ✅ ~50 lines per script
- ✅ One config file for all paths
- ✅ Reusable function library
- ✅ Easy to maintain
- ✅ Easy to test and extend

## 📝 Next Steps

1. **Update `src/config.py`** with your paths
2. **Test the example script** to make sure everything works
3. **Refactor your scripts one by one**:
   - Start with the simplest scripts
   - Replace hardcoded paths with config
   - Replace inline code with package functions
   - Test each refactored script

4. **Add new functions** as needed to the package modules

## 🤝 Contributing

When adding new functionality:

1. **Add utility functions** to `src/utils/`
2. **Add analysis functions** to `src/analysis/`
3. **Add plotting functions** to `src/plotting/`
4. **Update `__init__.py`** files to export new functions
5. **Document your functions** with docstrings

## 📚 Resources

- [NumPy](https://numpy.org/doc/)
- [xarray](https://docs.xarray.dev/)
- [Cartopy](https://scitools.org.uk/cartopy/)
- [Python Packaging Guide](https://packaging.python.org/)

---

**Questions?** Contact Lauren Hoffman (lhoffma2@ucsc.edu)
