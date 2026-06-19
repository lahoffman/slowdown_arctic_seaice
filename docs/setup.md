# Setup Guide

How to install the environment, point the code at your data, and verify the
setup. For the analysis steps themselves, see [`workflow.md`](workflow.md).

## 1. Install the environment

TensorFlow 2.14 and `innvestigate` (the LRP library) pin most of the stack, so
conda is the recommended path:

```bash
conda env create -f environment.yml
conda activate slowdown-seaice
```

`environment.yml` already runs `pip install -e .`, so the package is installed
in editable mode and `configs` / `src` are importable from anywhere.

### Alternative: pip / venv

```bash
python -m venv env && source env/bin/activate
pip install -e .            # core dependencies
pip install -e ".[dev]"     # + pytest, black, flake8, jupyter
```

`requirements.txt` lists the same pinned dependencies if you prefer
`pip install -r requirements.txt`. TensorFlow can be awkward to install via pip —
conda is more reliable.

> **Note on imports.** The package installs as `slowdown` (mapping `src/` →
> `slowdown`), but the workflow scripts add the repo root to `sys.path` and
> import as `from configs import paths` and `from src.data... import ...`. Run
> scripts from the repo root and both styles work.

## 2. Point the code at your data

All paths are derived from a single environment variable, `SLOWDOWN_DATA_ROOT`.
This is the only thing that changes between machines — you should not edit
`configs/paths.py` for routine use.

```bash
# macOS — add to ~/.zshrc to persist
export SLOWDOWN_DATA_ROOT=/Users/you/data/slowdowns

# Linux / remote server — add to ~/.bashrc
export SLOWDOWN_DATA_ROOT=/cofast/you/slowdowns
```

If the variable is unset, `configs/paths.py` falls back to a list of known
local paths; if none exist it raises an error on purpose, so missing data is
caught early rather than written to the wrong place.

### Data directory layout

Everything (inputs and outputs) lives under `SLOWDOWN_DATA_ROOT`, organised by
dataset. `configs/paths.py` creates these directories automatically:

```
DATA_ROOT/
├── nsidc/        raw Sea Ice Index Excel file + processed slowdown outputs
├── ersst/        raw sst.mnmean.nc + regridded SST + climate indices
├── cesm2le/      downloaded ensemble files + monthly SST/AICE/TREF + slowdowns
└── results/      models, attributions, predictions, metrics, figures, logs
```

The only raw input you must supply manually is the NSIDC Sea Ice Index file:

```
DATA_ROOT/nsidc/Sea_Ice_Index_Monthly_Data_with_Statistics_G02135_v3.0.xlsx
```

CESM2-LE and ERSSTv5 are downloaded by the pipeline (see
[`workflow.md`](workflow.md)).

## 3. Configuration files

`configs/` holds everything tunable:

- **`paths.py`** — every input/output path plus helper functions like
  `tvt_split_path(k)`, `model_path(split, run)`, and `attribution_path(...)`.
  Add new paths here rather than hard-coding them.
- **`model.py`** — CNN architecture (`CNN_CONFIG`) and ensemble settings
  (`ENSEMBLE_CONFIG`: 100 members, 9 splits).
- **`training.py`** — training hyperparameters (`TRAINING_CONFIG`), the
  train/val/test split ratios (`SPLIT_CONFIG`), LRP settings (`XAI_CONFIG`),
  climate-index analysis parameters (`ANALYSIS_CONFIG`), season definitions, and
  plotting defaults.

Import them with:

```python
from configs import paths
import configs.training as training
import configs.model as model
```

## 4. Verify the setup

```bash
# Confirm the package imports and paths resolve
python -c "from configs import paths; print('DATA_ROOT =', paths.DATA_ROOT)"

# Run the test suite (model + data tests)
pytest
```

If the first command prints your data root without raising, the environment
and `SLOWDOWN_DATA_ROOT` are configured correctly.

## Next steps

Continue to [`workflow.md`](workflow.md) to run the analysis, starting with the
preprocessing scripts.
