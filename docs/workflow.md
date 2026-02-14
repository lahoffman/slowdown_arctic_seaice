# Workflow Guide

Complete guide to the analysis workflow.

## Overview

The workflow consists of 5 steps:

1. **Download Data** - Get raw data files
2. **Preprocess** - Clean and create TVT splits
3. **Train Models** - Train CNN models
4. **Compute XAI** - Generate LRP attributions
5. **Generate Figures** - Create publication plots

## Detailed Workflow

### Step 1: Download Data

```bash
python scripts/01_download_data.py
```

**Purpose**: Check data availability and download if needed

**Inputs**: None

**Outputs**: Raw data files in your data directory

**What to implement**:
- Download instructions for each dataset
- Data validation checks

### Step 2: Preprocess Data

```bash
python scripts/02_preprocess_data.py --n-splits 9
```

**Purpose**: Create train-validate-test splits for robustness testing

**Inputs**:
- Raw CESM2-LE SST data
- SIE trend labels

**Outputs**:
- `results/data/tvt_split_{0-8}.nc`

**What to implement**:
- Load CESM2-LE monthly SST
- Compute seasonal averages
- Create 9 different TVT splits

### Step 3: Train Models

```bash
# Train all splits
python scripts/03_train_models.py

# Or train single split
python scripts/03_train_models.py --split 0 --seed 42
```

**Purpose**: Train CNN models on each split

**Inputs**:
- TVT split data from step 2
- Model config from `configs/model.py`
- Training config from `configs/training.py`

**Outputs**:
- `results/models/model_split_{i}_seed_{j}.h5`
- `results/models/model_split_{i}_seed_{j}.json` (metrics)

**What to implement**:
- Load TVT data
- Build CNN from config
- Train with early stopping
- Evaluate and save metrics

### Step 4: Compute XAI

```bash
# Compute attributions for all splits
python scripts/04_compute_xai.py

# Or single split
python scripts/04_compute_xai.py --split 0
```

**Purpose**: Compute LRP attributions for model interpretation

**Inputs**:
- Trained models from step 3
- Test data

**Outputs**:
- `results/attributions/lrp_split_{i}.nc`

**What to implement**:
- Load trained model
- Compute LRP attributions
- Aggregate across ensemble members
- Save as NetCDF

### Step 5: Generate Figures

```bash
# Generate all figures
python scripts/05_generate_figures.py

# Or specific figure
python scripts/05_generate_figures.py --figure performance
```

**Purpose**: Create publication-quality figures

**Inputs**:
- Model metrics
- Attribution maps
- Original data

**Outputs**:
- `results/figures/model_performance.png`
- `results/figures/attribution_maps.png`
- `results/figures/robustness_analysis.png`

**What to implement**:
- Load results from all splits
- Aggregate metrics
- Create multi-panel figures
- Apply publication styling

## Configuration

All configuration is in `configs/`:

- **`paths.py`** - Change data paths here
- **`model.py`** - Modify CNN architecture
- **`training.py`** - Adjust hyperparameters

## Tips

1. **Test on one split first**: Use `--split 0` to test quickly
2. **Check outputs**: Verify each step produces expected files
3. **Use logging**: Scripts print progress to console
4. **Parallel training**: Run different splits simultaneously

## Example Complete Run

```bash
# 1. Check data
python scripts/01_download_data.py

# 2. Preprocess
python scripts/02_preprocess_data.py

# 3. Train (parallel on 9 splits)
for i in {0..8}; do
    python scripts/03_train_models.py --split $i &
done
wait

# 4. Compute XAI
python scripts/04_compute_xai.py

# 5. Generate figures
python scripts/05_generate_figures.py
```

## Troubleshooting

**Import errors**: Make sure conda environment is activated

**Path errors**: Update `configs/paths.py`

**Memory errors**: Reduce batch size in `configs/training.py`
