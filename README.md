# Arctic Sea Ice Slowdown Analysis

Detecting and interpreting decadal *slowdowns* in Arctic sea ice loss with
convolutional neural networks (CNNs) and explainable AI (layer-wise relevance
propagation, LRP).

The pipeline classifies decadal trends in CESM2 Large Ensemble (CESM2-LE) sea
ice as "slowdown" (anomalously slow ice loss) or not, trains CNNs to predict
September slowdowns from summer (JJA) sea surface temperature (SST), explains
the predictions with LRP, and applies the trained models to observed SST
(ERSSTv5).

## Datasets

- **NSIDC** — observed monthly sea ice extent (SIE) and area (SIA); used to
  derive the slowdown threshold.
- **CESM2-LE** — 100-member climate model ensemble (SST, sea ice, surface
  temperature); the source of training data and labels.
- **ERSSTv5** — observed SST, regridded onto the CESM2-LE grid; used for
  out-of-sample prediction.

## Project structure

```
slowdown_arctic_seaice/
├── configs/                 # Configuration
│   ├── paths.py             # Data paths (driven by SLOWDOWN_DATA_ROOT)
│   ├── model.py             # CNN architecture
│   └── training.py          # Training / XAI / analysis hyperparameters
│
├── src/                     # Library code (import these)
│   ├── data/
│   │   ├── io.py            # NetCDF read/write helpers
│   │   ├── cesm2le/         # CESM2-LE download, combine, metrics, regrid,
│   │   │                    #   climate indices, forced response, slowdowns
│   │   └── observations/
│   │       ├── nsidc/       # SIE/SIA preprocessing + slowdown definition
│   │       └── ersst/       # ERSSTv5 download, regrid, indices, CNN prep
│   ├── cnn/                 # splits, model, train
│   └── xai/                 # lrp, k_means
│
├── scripts/                 # Numbered workflow scripts (run these)
├── notebooks/               # Exploratory notebooks
├── figures/                 # Notebooks that build paper figures
├── docs/                    # setup.md, workflow.md
│
└── DATA_ROOT/               # All data + outputs (outside the repo, gitignored)
    ├── nsidc/  ersst/  cesm2le/
    └── results/             # models, attributions, predictions, figures
```

Data and outputs live **outside** the repo, under the directory pointed to by
the `SLOWDOWN_DATA_ROOT` environment variable. Nothing large is committed to
git (see `.gitignore`).

## Quick start

```bash
# 1. Create the environment (conda recommended — TensorFlow 2.14 is pinned)
conda env create -f environment.yml
conda activate slowdown-seaice

# 2. Tell the code where your data lives
export SLOWDOWN_DATA_ROOT=/path/to/slowdowns

# 3. Run the pipeline (see docs/workflow.md for the full sequence)
python scripts/01_cesm2le_preprocessing.py --variable all
python scripts/02_cesm2le_slowdowns.py
python scripts/03_cesm2le_tvt_splits.py
python scripts/04_cesm2le_cnn_train.py
python scripts/05_cesm2le_lrp.py
```

`environment.yml` installs this package in editable mode (`pip install -e .`),
so `configs` and `src` are importable from anywhere. See
[`docs/setup.md`](docs/setup.md) for details.

## Workflow at a glance

The numbered scripts run roughly in order; the number is the dependency stage,
not a strict sequence (several `01_*` and `02_*` scripts are independent).

| Stage | Script | Purpose |
|------|--------|---------|
| 01 | `01_cesm2le_grid.py` | Extract lat/lon grid from a raw SST file |
| 01 | `01_cesm2le_landmask.py` | Build the ocean/land mask |
| 01 | `01_cesm2le_preprocessing.py` | Download CESM2-LE, combine, split by month, compute SIE/SIA |
| 01 | `01_ersst_preprocessing.py` | Download ERSSTv5 and regrid to the CESM2-LE grid |
| 01 | `01_nsidc_slowdown_sie_sia.py` | NSIDC decadal trends + slowdown thresholds/events |
| 02 | `02_cesm2le_climate_indices.py` | Niño3.4, ENSO CP/TP, IPO from CESM2-LE SST |
| 02 | `02_cesm2le_forced.py` | Ensemble-mean JJA SST (the forced response) |
| 02 | `02_cesm2le_slowdowns.py` | Classify CESM2-LE SIE/SIA slowdowns vs NSIDC thresholds |
| 02 | `02_cesm2le_slowdowns_gmt.py` | Classify GMT (warming) slowdowns |
| 02 | `02_ersst_climate_indices.py` | Climate indices from regridded ERSSTv5 |
| 03 | `03_cesm2le_tvt_splits.py` | Build the 9 train/validate/test splits |
| 03 | `03_ersst_test.py` | Prepare CNN-ready ERSSTv5 testing data |
| 04 | `04_cesm2le_cnn_train.py` | Train the CNN for each split × seed |
| 05 | `05_cesm2le_lrp.py` | Compute LRP-z attribution maps |
| 06 | `06_cnn_predict_cesm2le.py` | Precompute CNN predictions on CESM2-LE |
| 06 | `06_cnn_predict_ersst.py` | Precompute CNN predictions on ERSSTv5 |

Paper figures are built in `figures/`:

- `F1_FS1_FS2_slowdowns.ipynb` — NSIDC + CESM2-LE slowdown overview
- `FS3-7_model_performance.ipynb` — CNN performance and composite maps
- `F2-3_FS8-S9_composite_pdf.ipynb` — slowdown probability by climate-index phase
- `F4_predict_obs.ipynb` — predictions on observed (ERSSTv5) SST
- `FS10_sie_gmt.ipynb` — SIE vs GMT slowdown comparison

See [`docs/workflow.md`](docs/workflow.md) for inputs, outputs, and command-line
options for every step.

## Configuration

All settings live in `configs/`:

- `paths.py` — every input and output path, derived from `SLOWDOWN_DATA_ROOT`.
  Add new paths here rather than hard-coding them.
- `model.py` — CNN architecture and ensemble configuration.
- `training.py` — training hyperparameters, the LRP/XAI settings, climate-index
  analysis parameters, season definitions, and plotting defaults.

Scripts read these via `from configs import paths` (and `import configs.training`,
etc.).

## Development

```bash
pip install -e ".[dev]"   # pytest, black, flake8, jupyter
pytest                    # tests live alongside the code (src/data/cesm2le/test_*.py)
```

Add reusable functions to `src/`, then call them from a numbered script in
`scripts/`. Keep paths and hyperparameters in `configs/`.

---

**Author:** Lauren Hoffman (lhoffma2@ucsc.edu) · **Version:** 0.1.0
