# Setup Guide

## Project Structure

Your repository is now organized as a standard Python project:

```
slowdown_arctic_seaice/
├── configs/                    # ⭐ Configuration (CHANGE PATHS HERE)
│   ├── paths.py               # Data paths
│   ├── model.py               # Model architecture
│   └── training.py            # Training hyperparameters
│
├── src/                        # Library code (functions only - import these)
│   ├── data/                  # Data loading, preprocessing, climate indices
│   ├── models/                # Model building, training
│   ├── xai/                   # Explainable AI (LRP)
│   └── visualization/         # Plotting functions
│
├── scripts/                    # Workflow scripts (run these)
│   ├── 01_download_data.py    # Check/download data
│   ├── 02_preprocess_data.py  # Create TVT splits
│   ├── 03_train_models.py     # Train CNNs
│   ├── 04_compute_xai.py      # Compute LRP
│   └── 05_generate_figures.py # Create plots
│
├── notebooks/                  # Jupyter notebooks (for exploration)
├── docs/                       # Documentation
├── tests/                      # Unit tests
│
├── results/                    # All outputs (gitignored)
│   ├── data/                  # Processed TVT splits
│   ├── models/                # Trained models (.h5)
│   ├── attributions/          # LRP maps (.nc)
│   └── figures/               # Publication figures
│
└── [OLD - kept for reference]
    ├── data_download/         # Original scripts
    ├── data_processing/       # Original scripts
    ├── figures/               # Original scripts
    └── run_ml_xai/            # Original scripts
```

## Key Concepts

### 1. Configuration (`configs/`)
**All settings in one place**

- `paths.py` - **⭐ Change your data paths here!**
- `model.py` - CNN architecture
- `training.py` - Hyperparameters

### 2. Library (`src/`)
**Reusable functions - import these in your scripts**

```python
# In your scripts:
from src.data import load_netcdf, compute_nino34_index
from src.models import build_cnn_model, train_model
from src.xai import compute_lrp_analysis
from src.visualization import plot_map
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

## Setup Steps

### 1. Create Environment

```bash
conda env create -f environment-ml.yml
conda activate slowdown-ml
```

### 2. Install Package

```bash
pip install -e .
```

This makes `src/` importable from anywhere.

### 3. Configure Paths

**MOST IMPORTANT STEP!**

Edit `configs/paths.py`:

```python
# Change this line to your data location:
if os.path.exists('/your/local/path'):
    ROOT_DATA_PATH = Path('/your/local/path')
```

### 4. Verify Setup

```bash
python scripts/01_download_data.py
```

This checks if required data files exist.

## Workflow

### Standard Workflow

```bash
# 1. Preprocess (create 9 TVT splits)
python scripts/02_preprocess_data.py

# 2. Train models on all splits
python scripts/03_train_models.py

# 3. Compute XAI attributions
python scripts/04_compute_xai.py

# 4. Generate figures
python scripts/05_generate_figures.py
```

### Test on Single Split

```bash
# Train just split 0
python scripts/03_train_models.py --split 0

# Compute XAI for split 0
python scripts/04_compute_xai.py --split 0
```

## Implementation

The workflow scripts are templates. You need to:

1. **Implement data loading**:
   - In `scripts/02_preprocess_data.py`
   - Load your CESM2-LE SST data
   - Load SIE labels

2. **Implement model training**:
   - In `scripts/03_train_models.py`
   - Extract X_train, y_train from loaded data
   - Call training functions

3. **Implement XAI**:
   - In `scripts/04_compute_xai.py`
   - Load test data
   - Compute and save LRP

4. **Implement plotting**:
   - In `scripts/05_generate_figures.py`
   - Load results
   - Create publication figures

## Migrating Old Code

Your old code is in:
- `data_download/`
- `data_processing/`
- `figures/`
- `run_ml_xai/`

To migrate:

1. **Extract functions** → Move to appropriate `src/` module
2. **Extract scripts** → Adapt into workflow scripts
3. **Extract paths** → Move to `configs/paths.py`
4. **Extract hyperparameters** → Move to `configs/model.py` or `configs/training.py`

## Benefits

**Before**:
```
run_ml_xai/M1_EI_cnn_sst_seasonal_global.py  (400 lines, hardcoded paths)
run_ml_xai/M2_EI_lrpz_sst_seasonal_global.py (300 lines, hardcoded paths)
```

**After**:
```
configs/paths.py           (centralized paths)
configs/training.py        (centralized hyperparameters)
src/models/training.py     (reusable functions)
scripts/03_train_models.py (workflow script - 80 lines)
scripts/04_compute_xai.py  (workflow script - 70 lines)
```

## Questions?

See:
- `README.md` - Project overview
- `docs/workflow.md` - Detailed workflow guide
- `configs/` - All configuration files

---

**Get started**: Edit `configs/paths.py`, then run `python scripts/01_download_data.py`
