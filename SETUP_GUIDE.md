# Setup Guide: Modular Arctic Sea Ice Analysis

## 🚀 Quick Start

### 1. Create Conda Environments

You need TWO conda environments:

**Standard environment** (for data processing & simple figures):
```bash
cd /Users/lahoffma/projects/arcticWATCH/slowdown_arctic_seaice
conda env create -f environment.yml
conda activate slowdown-standard
```

**ML environment** (for CNN/XAI figures):
```bash
conda env create -f environment-ml.yml
conda activate slowdown-ml
```

### 2. Configure Paths

Edit `src/config.py` and update the ROOT_PATH:

```python
ROOT_PATH = Path('/Users/lahoffma/projects/arcticWATCH/slowdown_arctic_seaice')
```

### 3. Test the Setup

**Test standard environment:**
```bash
conda activate slowdown-standard
python examples/example_refactored_enso_index.py
```

**Test ML environment:**
```bash
conda activate slowdown-ml
python -c "import tensorflow; print(tensorflow.__version__)"
python -c "import innvestigate; print('XAI tools ready!')"
```

## 📦 What Was Created

### New Directory Structure

```
slowdown_arctic_seaice/
├── src/                              # Modular package
│   ├── config.py                     # ⭐ PATHS & SETTINGS
│   ├── utils/                        # Data I/O, grid operations
│   ├── analysis/                     # Climate indices, statistics
│   ├── plotting/                     # Plotting utilities
│   ├── figures/                      # 🆕 Figure-specific utilities
│   └── ml/                           # 🆕 ML/XAI utilities
│
├── figures/                          # Original scripts (KEPT)
├── figures_refactored/              # 🆕 New refactored scripts
│   ├── F1_siextent_slowdown_REFACTORED.py (template)
│   └── README_REFACTORING.md (guide)
│
├── examples/                         # Example scripts
│   └── example_refactored_enso_index.py
│
├── environment.yml                   # 🆕 Standard conda env
├── environment-ml.yml               # 🆕 ML conda env
├── requirements.txt                 # Standard pip requirements
├── requirements-ml.txt              # ML pip requirements
├── setup.py                         # Package installation
├── README_MODULAR.md               # Main documentation
└── SETUP_GUIDE.md                  # This file!
```

### New Modules Created

#### `src/config.py` ⭐
Centralized configuration - **change paths here**!

#### `src/utils/`
- `data_io.py`: Load/save NetCDF files
- `grid_ops.py`: Regridding, subsetting
- `general.py`: Moving averages, utilities

#### `src/analysis/`
- `climate_indices.py`: ENSO, IPO calculations
- `statistics.py`: Anomalies, climatology
- `trends.py`: Trend analysis

#### `src/plotting/`
- `maps.py`: Map plotting
- `timeseries.py`: Time series plots

#### `src/figures/` 🆕
- `trends.py`: `moving_decadal_trend()`, `classify_slowdown()`
- `plot_utils.py`: `plot_colored_decadal_segments()`, figure styling

#### `src/ml/` 🆕
- `model_utils.py`: Load models, prepare data
- `xai_utils.py`: LRP analysis, attribution maps
- `statistics.py`: `bootstrap_mean_ci()`, permutation tests

## 🔧 Environment Details

### Standard Environment (Python 3.11)
For data processing and non-ML figures:
- Modern NumPy (1.24+)
- Modern SciPy, pandas, xarray
- Cartopy, matplotlib, cmocean

**Use for:**
- Data processing scripts (`data_processing/`, `data_download/`)
- Simple figures (`F1`, `FSX_*`)
- General analysis

### ML Environment (Python 3.9)
For CNN models and XAI analysis:
- TensorFlow 2.12 (last stable for Python 3.9)
- innvestigate 2.0+ (for LRP analysis)
- Compatible NumPy 1.23, SciPy 1.10
- scikit-learn 1.2

**Use for:**
- ML figure scripts (`F2-3`, `F4_*`, `M1`)
- Any script importing `tensorflow` or `innvestigate`

## 📝 Working with the Code

### Switching Environments

```bash
# For data processing or simple figures
conda activate slowdown-standard

# For ML/XAI figures
conda activate slowdown-ml

# Check current environment
conda info --envs
```

### Running Original Scripts

Original scripts still work (they're in `figures/`):

```bash
cd figures
# They still reference /cofast/lhoffman/slowdown/ paths
# You may need to update paths or run on cluster
```

### Running Refactored Scripts

Refactored scripts use the new modular structure:

```bash
# Make sure you updated src/config.py first!

conda activate slowdown-standard
python figures_refactored/F1_siextent_slowdown_REFACTORED.py

# Or for ML scripts:
conda activate slowdown-ml
python figures_refactored/F2-3_REFACTORED.py  # (create this next)
```

## 🎓 Next Steps

### For You to Do:

1. ✅ **Done**: Modular structure created
2. ✅ **Done**: Conda environments configured
3. ✅ **Done**: Utility modules created
4. ✅ **Done**: Example scripts created
5. ⏳ **TODO**: Update `src/config.py` with your paths
6. ⏳ **TODO**: Create conda environments
7. ⏳ **TODO**: Test example script
8. ⏳ **TODO**: Refactor remaining figure scripts (use template!)

### Refactoring Priority:

1. **Start**: `FSX_siextent_slowdown.py` (simple, short)
2. **Practice**: `F1` (use template provided)
3. **ML scripts**: `F2-3`, `F4_*`, `M1` (use ML environment)

## 📚 Documentation

- **Main guide**: `README_MODULAR.md` - Complete package documentation
- **Refactoring guide**: `figures_refactored/README_REFACTORING.md` - How to refactor scripts
- **This file**: Quick setup reference
- **Examples**:
  - `examples/example_refactored_enso_index.py` - Data analysis example
  - `figures_refactored/F1_siextent_slowdown_REFACTORED.py` - Figure script template

## ⚠️ Important Notes

### Path Management
- **NEVER hardcode paths** in scripts anymore
- **ALWAYS use** `config.ROOT_PATH` or `config.get_data_file()`
- Change paths in ONE place: `src/config.py`

### Environment Selection
- Check imports at top of script
- See `tensorflow` or `innvestigate`? → Use `slowdown-ml`
- No ML imports? → Use `slowdown-standard`

### Git Tracking
- ✅ **Commit**: New `src/` modules, conda env files, refactored scripts
- ✅ **Commit**: Documentation files
- ⚠️ **Don't commit**: Data files (`.nc`, `.h5`), model files
- ⚠️ **Don't commit**: Output figures (`.png`, `.pdf`)

## 🆘 Troubleshooting

### "Module 'src' not found"
```python
# Add at top of script:
import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
```

### "TensorFlow version incompatible with innvestigate"
```bash
# Use ML environment:
conda activate slowdown-ml
```

### "File not found: /cofast/lhoffman/..."
```python
# Update src/config.py with correct ROOT_PATH
ROOT_PATH = Path('/Users/lahoffma/projects/arcticWATCH/slowdown_arctic_seaice')
```

### Conda environment creation fails
```bash
# Try creating with explicit channel priority:
conda env create -f environment-ml.yml --channel-priority flexible

# Or create manually:
conda create -n slowdown-ml python=3.9
conda activate slowdown-ml
pip install -r requirements-ml.txt
```

## 🎉 Benefits Summary

**Before modularization:**
- ❌ 270+ lines per script
- ❌ Paths hardcoded everywhere
- ❌ Functions copied across files
- ❌ Difficult to maintain

**After modularization:**
- ✅ ~50-100 lines per script
- ✅ Paths in ONE file
- ✅ Reusable function library
- ✅ Easy to maintain and test
- ✅ Proper environment management

---

**Ready to start?** Update `src/config.py`, create your conda environments, and test the example script!

**Questions?** Check `README_MODULAR.md` or `figures_refactored/README_REFACTORING.md`
