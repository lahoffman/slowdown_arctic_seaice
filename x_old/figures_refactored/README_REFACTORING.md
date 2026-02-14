# Refactoring Guide for Figure Scripts

This guide explains how to refactor your figure scripts to use the new modular structure.

## 📋 Overview

You have **11 figure scripts** to refactor:
- **4 simple scripts** (no ML): F1, FSX, FSX_lots, FSX_EI_performance
- **7 ML scripts** (TensorFlow/innvestigate): F2-3, F4 (3 variants), M1

## 🎯 Refactoring Strategy

### Step 1: Set Up Conda Environments

First, create both conda environments:

```bash
# Standard environment (for simple scripts)
conda env create -f environment.yml
conda activate slowdown-standard

# ML environment (for CNN/XAI scripts)
conda env create -f environment-ml.yml
conda activate slowdown-ml
```

### Step 2: Update src/config.py

Edit `src/config.py` to set your ROOT_PATH:

```python
ROOT_PATH = Path('/Users/lahoffma/projects/arcticWATCH/slowdown_arctic_seaice')
```

### Step 3: Refactor Scripts One by One

Follow this template for each script:

## 📝 Refactoring Template

### Before (Old Script):

```python
#------------------------------------------------------
# ROOT PATH
rootpath = '/cofast/lhoffman/slowdown/'
#------------------------------------------------------

# 50+ lines of imports
import sys
import os
import numpy as np
# ...

sys.path.append(rootpath+'functions/')
from functions_general import ncdisp, movmean

# Inline function definitions
def bootstrap_mean_ci(vals, B=1000, alpha=0.05):
    # 20 lines of code...
    pass

def moving_decadal_trend(y, window=10):
    # 15 lines of code...
    pass

# 200+ lines of procedural code
filepath = rootpath + 'data/myfile.nc'
dataset = nc.Dataset(filepath, 'r')
# ...
```

### After (Refactored):

```python
#!/usr/bin/env python3
"""
Figure X: Description (REFACTORED)

Brief description of what this figure shows.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

# Import from modular package
from src import config
from src.utils import load_netcdf, load_cesm2_grid
from src.figures import setup_figure_style, moving_decadal_trend
from src.ml import bootstrap_mean_ci  # If using ML functions

def main():
    """Main function."""

    print("Generating Figure X...")

    # 1. Load data using config paths
    data_file = config.ROOT_PATH / 'data' / 'myfile.nc'
    data = load_netcdf(data_file, variables=['sst', 'lat', 'lon'])

    # 2. Process data using package functions
    trends = moving_decadal_trend(data['sst'], window=10)

    # 3. Create figure
    setup_figure_style()
    fig, ax = plt.subplots(figsize=(12, 8))

    # 4. Plot
    ax.plot(data['time'], trends)

    # 5. Save
    output_file = config.get_figure_file('figureX.png')
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_file}")

    plt.show()

if __name__ == "__main__":
    main()
```

## 🔧 Common Refactoring Tasks

### 1. Replace Hardcoded Paths

**Before:**
```python
rootpath = '/cofast/lhoffman/slowdown/'
filepath = rootpath + 'D1_ersstv5_sst_regridded_to_cesm2_1854-2024.nc'
```

**After:**
```python
from src import config
filepath = config.ROOT_PATH / 'D1_ersstv5_sst_regridded_to_cesm2_1854-2024.nc'
# Or use the predefined path:
filepath = config.ERSST_REGRIDDED
```

### 2. Replace Inline Functions

**Before:**
```python
def bootstrap_mean_ci(vals, B=1000, alpha=0.05):
    # 20 lines...
    pass

mean, lower, upper = bootstrap_mean_ci(my_data)
```

**After:**
```python
from src.ml import bootstrap_mean_ci

mean, lower, upper = bootstrap_mean_ci(my_data)
```

### 3. Replace Data Loading

**Before:**
```python
dataset = nc.Dataset(filepath, 'r')
lat = np.array(dataset.variables['lat'])
lon = np.array(dataset.variables['lon'])
sst = np.array(dataset.variables['sst'])
dataset.close()
```

**After:**
```python
from src.utils import load_netcdf

data = load_netcdf(filepath, variables=['lat', 'lon', 'sst'])
lat, lon, sst = data['lat'], data['lon'], data['sst']
```

### 4. Use Figure Utilities

**Before:**
```python
import matplotlib as mpl
mpl.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 20,
    # ... 15 more lines
})
```

**After:**
```python
from src.figures import setup_figure_style

setup_figure_style()
```

## 🧪 ML/XAI Scripts

For scripts using TensorFlow/innvestigate (F2-3, F4 variants, M1):

### Environment

```bash
conda activate slowdown-ml  # Use ML environment!
```

### Import Pattern

```python
from src.ml import (
    load_model,
    compute_lrp_analysis,
    bootstrap_mean_ci,
    bootstrap_p_value_diff_means
)
```

### Example: Loading Model and Computing LRP

**Before:**
```python
import tensorflow as tf
from tensorflow import keras
import innvestigate

model = keras.models.load_model(rootpath + 'model_seasonal.h5')
analyzer = innvestigate.create_analyzer('lrp.z', model)
attributions = analyzer.analyze(X_test)
```

**After:**
```python
from src.ml import load_model, compute_lrp_analysis

model_path = config.ROOT_PATH / 'model_seasonal.h5'
model = load_model(model_path)
attributions = compute_lrp_analysis(model, X_test, method='lrp.z')
```

## 📊 Script-Specific Notes

### F1_FS1_FS2_siextent_slowdown.py (Simple)
- **Lines**: ~1000
- **Key functions**: `moving_decadal_trend`, `classify_slowdown`, `plot_colored_decadal_segments`
- **Refactored version**: `figures_refactored/F1_siextent_slowdown_REFACTORED.py` (template provided)
- **Status**: ✅ Utility functions created in `src/figures/`

### F2-3_EI_cnn_lrpz_seasonal_global.py (ML)
- **Lines**: ~800
- **Key functions**: `bootstrap_mean_ci`, `bootstrap_p_value_diff_means`, LRP analysis
- **Environment**: `slowdown-ml`
- **Key imports**: `tensorflow`, `innvestigate`, `sklearn`
- **Status**: ✅ ML utilities created in `src/ml/`

### F4_EI_*_indexing_pdf.py (3 variants - ML)
- **Variants**: ENSO, IPO, SSTArctic
- **Common structure**: Similar patterns, can refactor together
- **Environment**: `slowdown-ml`

### FSX_siextent_slowdown*.py (2 variants - Simple)
- **Simpler scripts**: Easier refactoring
- **Environment**: `slowdown-standard`

### M1_cnn_prediction_obs_seasonal_global.py (ML)
- **Model predictions on observations**
- **Environment**: `slowdown-ml`

## 🚀 Recommended Refactoring Order

1. **Start simple**: FSX scripts (no ML, shorter)
2. **Practice on F1**: Use the template provided
3. **Move to ML**: F2-3 (well-structured ML script)
4. **Group similar**: F4 variants (similar structure)
5. **Finish with M1**: Model predictions

## ✅ Refactoring Checklist

For each script, check:

- [ ] Replaced `rootpath` with `config` imports
- [ ] Replaced inline functions with package imports
- [ ] Used `load_netcdf()` instead of raw `nc.Dataset()`
- [ ] Used `setup_figure_style()` for matplotlib settings
- [ ] Created `main()` function
- [ ] Added docstrings
- [ ] Tested script runs successfully
- [ ] Output figures match originals
- [ ] Saved to `figures_refactored/` folder

## 🐛 Common Issues

### Issue 1: Import Errors

**Error**: `ModuleNotFoundError: No module named 'src'`

**Solution**: Add repo root to path at top of script:
```python
import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
```

### Issue 2: TensorFlow Version Conflicts

**Error**: `innvestigate requires tensorflow < 2.13`

**Solution**: Use the ML conda environment:
```bash
conda activate slowdown-ml
```

### Issue 3: Path Not Found

**Error**: `FileNotFoundError: /cofast/lhoffman/slowdown/...`

**Solution**: Update `src/config.py` with correct `ROOT_PATH`

## 📚 Additional Resources

- **Example refactored script**: `figures_refactored/F1_siextent_slowdown_REFACTORED.py`
- **Example ENSO script**: `examples/example_refactored_enso_index.py`
- **Main README**: `README_MODULAR.md`
- **Package documentation**: Docstrings in all `src/` modules

## 💡 Tips

1. **Keep originals**: The original `figures/` folder is unchanged
2. **Work incrementally**: Refactor one section at a time
3. **Test frequently**: Run script after each major change
4. **Compare outputs**: Verify figures match originals
5. **Ask for help**: If stuck, refer to template or examples

## 🎓 Learning Examples

Two complete examples provided:

1. **Simple analysis**: `examples/example_refactored_enso_index.py`
   - Shows: Data loading, analysis, plotting
   - Environment: `slowdown-standard`
   - Lines: ~130 (vs ~270 original)

2. **Figure script**: `figures_refactored/F1_siextent_slowdown_REFACTORED.py`
   - Shows: Complex figure generation
   - Environment: `slowdown-standard`
   - Lines: ~200 (vs ~1000 original)

---

**Questions?** Refer to `README_MODULAR.md` or examine the example scripts!
