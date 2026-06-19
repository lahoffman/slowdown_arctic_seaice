# Workflow Guide

The full analysis pipeline, step by step. Every step assumes the environment is
installed and `SLOWDOWN_DATA_ROOT` is set (see [`setup.md`](setup.md)). Run
scripts from the repo root.

Scripts are numbered by **dependency stage** (`01_` → `06_`), not as a strict
linear order — several scripts within a stage are independent and can run in any
order or in parallel. All outputs are written under `SLOWDOWN_DATA_ROOT`; exact
paths are defined in `configs/paths.py`.

## Stage 01 — Preprocessing

These prepare the three datasets and can run independently.

### CESM2-LE grid and land mask

```bash
python scripts/01_cesm2le_grid.py <raw_sst_file.nc> -o cesm2le_sst_grid.nc
python scripts/01_cesm2le_landmask.py
```

Extracts the lat/lon grid and builds the ocean (0) / land (1) mask used for
regridding and for masking land pixels during training.

### CESM2-LE download + metrics

```bash
python scripts/01_cesm2le_preprocessing.py --variable all
```

Downloads raw CESM2-LE chunks from UCAR, concatenates them along time, splits
the result into one file per calendar month, and (for AICE) computes sea ice
extent (SIE) and area (SIA). Useful options:

```bash
--variable sst|aice|tref|all   # which variable(s) to process
--skip-download                # data already on disk
--metrics-only                 # only recompute SIE/SIA from existing AICE files
--member-groups first50 last50 # restrict ensemble members
--dry-run                      # print download commands without running them
```

### ERSSTv5 download + regrid

```bash
python scripts/01_ersst_preprocessing.py
```

Downloads ERSSTv5 SST and regrids it onto the CESM2-LE grid
(`DATA_ROOT/ersst/sst_regrid_cesm2le.nc`).

### NSIDC slowdown thresholds

```bash
python scripts/01_nsidc_slowdown_sie_sia.py
# options: --start-year 1990 --window 10
```

Computes 10-year sliding decadal trends in observed SIE/SIA, defines the
slowdown threshold (mean + 1σ), and writes per-month threshold and event files.
These thresholds define what counts as a "slowdown" for the model data.

## Stage 02 — Indices, forced response, slowdown labels

### Climate indices

```bash
python scripts/02_cesm2le_climate_indices.py          # all indices
python scripts/02_cesm2le_climate_indices.py --index nino34
python scripts/02_ersst_climate_indices.py            # from regridded ERSSTv5
```

Computes Niño3.4, ENSO CP/TP, and IPO indices from CESM2-LE and ERSSTv5 SST.

### Forced response

```bash
python scripts/02_cesm2le_forced.py
```

Saves the ensemble-mean JJA SST — the forced component that is subtracted from
each member during training, and reused by the observation pipeline.

### Slowdown classification

```bash
python scripts/02_cesm2le_slowdowns.py                       # SIE, all months
python scripts/02_cesm2le_slowdowns.py --variable sia --months SEP
python scripts/02_cesm2le_slowdowns_gmt.py                   # GMT (warming) slowdowns
```

Classifies CESM2-LE decadal trends as slowdown / RILES events using the NSIDC
thresholds from stage 01. The September SIE classification produces the labels
used for CNN training. The GMT script is the sign-flipped analogue used for the
SIE-vs-GMT comparison figure.

**Depends on:** `01_cesm2le_preprocessing.py` (SIE/SIA or GMT files) and
`01_nsidc_slowdown_sie_sia.py` (thresholds).

## Stage 03 — Build model-ready data

### Training splits

```bash
python scripts/03_cesm2le_tvt_splits.py                    # full pipeline
python scripts/03_cesm2le_tvt_splits.py --climate-indices-only
```

Builds the 9 train / validate / test splits. For each split it loads JJA SST and
September slowdown labels, aligns years, splits the 100 members into
train/val/test blocks, standardises SST with training-set ocean statistics,
applies the land mask, and saves one NetCDF per split
(`results/tvt_splits/cesm2le_sst_jja_slowdown_split{k}.nc`), including the
normalisation statistics needed for LRP.

**Depends on:** `01_cesm2le_preprocessing.py` and `02_cesm2le_slowdowns.py`.

### Observational testing data

```bash
python scripts/03_ersst_test.py
python scripts/03_ersst_test.py --forced-method linear     # or ensmean (default)
```

Prepares CNN-ready ERSSTv5 testing arrays (standardised, land-masked), removing
the forced response by ensemble mean or linear method.

**Depends on:** `01_ersst_preprocessing.py` (and `02_cesm2le_forced.py` for the
`ensmean` method).

## Stage 04 — Train the CNN

```bash
python scripts/04_cesm2le_cnn_train.py
```

Trains the JJA SST CNN for each of the 9 splits and every random seed. Per split
× seed it loads the split, adds a channel dimension, sets the seed, computes
balanced class weights, trains with early stopping on validation loss, and saves
the model (`results/models/cnn_jja_split{k}_run{r}.h5`). After all seeds for a
split, it writes a metrics dataset with per-run values and 2.5/97.5 percentile
confidence intervals (`results/metrics/cnn_jja_metrics_split{k}.nc`).

**Depends on:** `03_cesm2le_tvt_splits.py`.

## Stage 05 — Explainability (LRP)

```bash
python scripts/05_cesm2le_lrp.py
```

Computes LRP-z attribution maps for each trained model over its training-set
SST, after stripping the output activation. Saves one NetCDF per split × seed
(`results/attributions/lrp_jja_split{k}_run{r}.nc`).

> **Run in a separate process from training.** iNNvestigate requires TF1-style
> graph mode, so importing `src.xai.lrp` disables eager execution. Do not run
> this in the same Python session as `04_cesm2le_cnn_train.py`.

**Depends on:** `03_cesm2le_tvt_splits.py` and `04_cesm2le_cnn_train.py`.

## Stage 06 — Precompute predictions

```bash
python scripts/06_cnn_predict_cesm2le.py
python scripts/06_cnn_predict_ersst.py
python scripts/06_cnn_predict_ersst.py --forced-method ensmean linear
```

Runs every model on the CESM2-LE train/val/test data and on the ERSSTv5
observations, saving predicted probabilities, thresholded predictions, and true
labels. This caches inference so the figure notebooks don't have to re-run all
models. The CESM2-LE script also computes the PR-curve optimal threshold; the
ERSST script uses the default 0.5 sigmoid threshold.

**Depends on:** `04_cesm2le_cnn_train.py`, plus `03_cesm2le_tvt_splits.py`
(CESM2-LE) or `03_ersst_test.py` (ERSST).

## Figures

The paper figures are built from the cached outputs in `figures/` notebooks —
no retraining required:

| Notebook | Figures |
|----------|---------|
| `F1_FS1_FS2_slowdowns.ipynb` | NSIDC + CESM2-LE slowdown overview |
| `FS3-7_model_performance.ipynb` | CNN performance + SST/LRP composite maps |
| `F2-3_FS8-S9_composite_pdf.ipynb` | Slowdown probability by ENSO/IPO/Arctic-SST phase |
| `F4_predict_obs.ipynb` | Predictions on observed (ERSSTv5) SST |
| `FS10_sie_gmt.ipynb` | SIE vs GMT slowdown comparison |

## Example end-to-end run

```bash
export SLOWDOWN_DATA_ROOT=/path/to/slowdowns

# Stage 01 — preprocessing
python scripts/01_cesm2le_grid.py <raw_sst_file.nc> -o cesm2le_sst_grid.nc
python scripts/01_cesm2le_landmask.py
python scripts/01_cesm2le_preprocessing.py --variable all
python scripts/01_ersst_preprocessing.py
python scripts/01_nsidc_slowdown_sie_sia.py

# Stage 02 — indices, forced response, labels
python scripts/02_cesm2le_climate_indices.py
python scripts/02_ersst_climate_indices.py
python scripts/02_cesm2le_forced.py
python scripts/02_cesm2le_slowdowns.py
python scripts/02_cesm2le_slowdowns_gmt.py

# Stage 03 — model-ready data
python scripts/03_cesm2le_tvt_splits.py
python scripts/03_ersst_test.py

# Stage 04–06 — train, explain, predict
python scripts/04_cesm2le_cnn_train.py
python scripts/05_cesm2le_lrp.py          # separate process from training
python scripts/06_cnn_predict_cesm2le.py
python scripts/06_cnn_predict_ersst.py

# Figures: run the notebooks in figures/
```

## Troubleshooting

- **`EnvironmentError: Cannot find the data root`** — set `SLOWDOWN_DATA_ROOT`
  (see [`setup.md`](setup.md)).
- **Import errors** — activate the conda environment and run from the repo root.
- **LRP / eager-execution errors** — run `05_cesm2le_lrp.py` in its own process,
  separate from any script that uses TensorFlow eager mode.
- **Out-of-memory during training** — lower `batch_size` in `configs/training.py`.
