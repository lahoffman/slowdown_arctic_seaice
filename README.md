# Arctic Sea Ice Slowdown Analysis

Analysis of Arctic sea ice extent slowdown using CNN models and explainable AI.

## 📁 Project Structure

```
slowdown_arctic_seaice/
├── configs/                # Configuration (paths, hyperparameters)
│   ├── paths.py           # ⭐ CHANGE PATHS HERE
│   ├── model.py           # Model architecture config
│   └── training.py        # Training hyperparameters
│
├── src/                    # Library code (functions only)
│   ├── data/              # Data loading, preprocessing
│   ├── models/            # Model architecture, training
│   ├── xai/               # Explainable AI (LRP)
│   └── visualization/     # Plotting functions
│
├── scripts/               # Executable workflow scripts
│   ├── 01_download_data.py
│   ├── 02_preprocess_data.py
│   ├── 03_train_models.py
│   ├── 04_compute_xai.py
│   └── 05_generate_figures.py
│
├── notebooks/             # Jupyter notebooks
├── docs/                  # Documentation
├── tests/                 # Unit tests
│
├── results/               # All outputs (gitignored)
│   ├── data/             # Processed data
│   ├── models/           # Trained models
│   ├── attributions/     # XAI outputs
│   └── figures/          # Plots
│
└── [Old directories kept for reference]
    ├── data_download/
    ├── data_processing/
    ├── figures/
    └── run_ml_xai/
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda env create -f environment-ml.yml
conda activate slowdown-ml

# Install package
pip install -e .
```

### 2. Configure Paths

Edit `configs/paths.py` to set your data locations:

```python
# Change this line:
ROOT_DATA_PATH = Path('/your/path/to/data')
```

### 3. Run Workflow

```bash
# Step 1: Check data availability
python scripts/01_download_data.py

# Step 2: Preprocess and create TVT splits
python scripts/02_preprocess_data.py

# Step 3: Train models
python scripts/03_train_models.py

# Or train single split:
python scripts/03_train_models.py --split 0

# Step 4: Compute XAI attributions
python scripts/04_compute_xai.py

# Step 5: Generate figures
python scripts/05_generate_figures.py
```

## 📖 Documentation

- **`configs/`** - All configuration in one place
- **`src/`** - Library functions (import these in your scripts)
- **`scripts/`** - Workflow scripts (run these)
- **`docs/`** - Detailed documentation

## 🔧 Development

### Project Organization

- **`src/`** contains reusable functions (library code)
- **`scripts/`** contains executable workflows (run these)
- **`configs/`** contains all configuration (change settings here)

### Adding New Analysis

1. Add functions to `src/` (e.g., `src/data/my_analysis.py`)
2. Create script in `scripts/` that uses those functions
3. Update configs if needed

## 📝 Notes

- **Old directories** (`data_download/`, `data_processing/`, `figures/`) are kept for reference
- **All outputs** go to `results/` (gitignored)
- **All configuration** is in `configs/` (change paths there)

## 📚 Key Files

- `configs/paths.py` - **Change your data paths here**
- `scripts/01_download_data.py` - Start here
- `environment-ml.yml` - ML environment setup

---

**Author**: Lauren Hoffman (lhoffma2@ucsc.edu)
**Version**: 1.0.0
