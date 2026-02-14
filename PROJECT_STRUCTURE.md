# Project Structure Overview

## ✅ Clean Structure Implemented

Your repository now follows standard Python project organization:

## 📁 New Clean Structure

```
slowdown_arctic_seaice/
├── configs/                    # ⭐ All configuration
│   ├── paths.py               #    Change paths HERE
│   ├── model.py               #    CNN architecture
│   └── training.py            #    Hyperparameters
│
├── src/                        # Library code (functions only)
│   ├── data/                  #    Data I/O, preprocessing, indices
│   ├── models/                #    Architecture, training, pipeline
│   ├── xai/                   #    LRP and XAI functions
│   └── visualization/         #    Plotting functions
│
├── scripts/                    # Workflow scripts (executable)
│   ├── 01_download_data.py    #    Download/check data
│   ├── 02_preprocess_data.py  #    Create TVT splits
│   ├── 03_train_models.py     #    Train CNN models
│   ├── 04_compute_xai.py      #    Compute LRP attributions
│   └── 05_generate_figures.py #    Generate plots
│
├── docs/                       # Documentation
│   ├── setup.md               #    Setup instructions
│   └── workflow.md            #    Workflow guide
│
├── notebooks/                  # Jupyter notebooks
├── tests/                      # Unit tests
│
├── results/                    # All outputs (gitignored)
│   ├── data/                  #    Processed TVT splits
│   ├── models/                #    Trained models
│   ├── attributions/          #    LRP attributions
│   └── figures/               #    Publication figures
│
├── .gitignore
├── README.md                   # Main readme
├── setup.py
├── environment.yml
├── environment-ml.yml
│
└── [OLD - Kept for Reference]
    ├── data_download/         # Original scripts
    ├── data_processing/       # Original scripts
    ├── figures/               # Original scripts
    ├── figures_refactored/    # Previous refactoring
    ├── run_ml_xai/            # Previous ML scripts
    └── examples/              # Previous examples
```

## 🎯 Key Principles

### 1. Configuration (`configs/`)
**All settings in ONE place**

```python
# configs/paths.py
ROOT_DATA_PATH = Path('/your/path')  # ⭐ Change here

# configs/model.py
CNN_CONFIG = {...}  # Model architecture

# configs/training.py
TRAINING_CONFIG = {...}  # Hyperparameters
```

### 2. Library (`src/`)
**Reusable functions - import these**

```python
# In your scripts:
from src.data import load_netcdf, compute_nino34_index
from src.models import build_cnn_model, train_model
from src.xai import compute_lrp_analysis
```

### 3. Workflow (`scripts/`)
**Executable scripts - run these**

```bash
python scripts/01_download_data.py
python scripts/02_preprocess_data.py
python scripts/03_train_models.py
python scripts/04_compute_xai.py
python scripts/05_generate_figures.py
```

## 🚀 Quick Start

```bash
# 1. Edit paths
vim configs/paths.py  # Change ROOT_DATA_PATH

# 2. Setup environment
conda env create -f environment-ml.yml
conda activate slowdown-ml
pip install -e .

# 3. Run workflow
python scripts/01_download_data.py     # Check data
python scripts/02_preprocess_data.py   # Create splits
python scripts/03_train_models.py      # Train models
python scripts/04_compute_xai.py       # Compute XAI
python scripts/05_generate_figures.py  # Generate plots
```

## 📝 What You Need to Do

The structure is complete! Now implement your workflow:

### In `scripts/`:

1. **`02_preprocess_data.py`**:
   - Implement `load_cesm2_sst_seasonal()`
   - Load your actual CESM2-LE data

2. **`03_train_models.py`**:
   - Load TVT splits
   - Extract X_train, y_train
   - Call training functions

3. **`04_compute_xai.py`**:
   - Load trained models
   - Compute LRP on test data
   - Save attributions

4. **`05_generate_figures.py`**:
   - Load results
   - Create publication figures

## 🔄 Migrating Old Code

Your old code is preserved in:
- `data_download/`
- `data_processing/`
- `figures/`
- `run_ml_xai/`

To migrate:

**Step 1**: Extract functions → Put in `src/`
```python
# Old: run_ml_xai/M1_*.py (inline function)
def compute_nino34_index(...):
    # code here

# New: src/data/climate_indices.py (imported)
from src.data import compute_nino34_index
```

**Step 2**: Extract workflow → Put in `scripts/`
```python
# Old: Long procedural script
# 400 lines of sequential code

# New: Clean workflow script
# 80 lines calling library functions
```

**Step 3**: Extract configuration → Put in `configs/`
```python
# Old: Hardcoded everywhere
rootpath = '/cofast/lhoffman/slowdown/'

# New: Centralized
from configs import paths
data_file = paths.ROOT_DATA_PATH / 'myfile.nc'
```

## 📚 Documentation

- **`README.md`** - Project overview
- **`docs/setup.md`** - Setup instructions
- **`docs/workflow.md`** - Detailed workflow guide
- **`PROJECT_STRUCTURE.md`** - This file

## ✨ Benefits

**Organized**: Clear separation of concerns

**Maintainable**: Change paths in ONE place

**Reusable**: Functions in `src/` used across scripts

**Standard**: Follows Python best practices

**Clean**: Workflow is clear and linear

---

**Start here**: `docs/setup.md` → `docs/workflow.md` → `scripts/`
