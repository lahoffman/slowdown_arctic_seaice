# Workflow Guide

The full analysis pipeline, step by step. Every step assumes the environment is
installed and `SLOWDOWN_DATA_ROOT` is set (see [`setup.md`](setup.md)). Run
scripts from the repo root.

Scripts are numbered by **dependency stage** (`01_` → `07_`), not as a strict
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

### Relative (epoch-free) slowdown labels

```bash
python scripts/02_cesm2le_slowdowns_relative.py                     # 10-yr, 1σ, per-group demean
python scripts/02_cesm2le_slowdowns_relative.py --window 3 5 7 10 15  # window sweep
python scripts/02_cesm2le_slowdowns_relative.py --demean all        # full-ensemble reference
```

Alternative label definition (revision plan §4.5): a slowdown is a member's
decadal-trend anomaly relative to its *forcing-group* mean trend (members 0–49
CMIP6 BB, 50–99 SMBB) exceeding +n σ, with σ pooled over 1990–2040. This
removes the year-dependence of the base rate produced by scaling the observed
threshold with a near-zero ensemble-mean trend, and keeps the biomass-burning
forcing artifact from leaking between groups. Writes one file per window × σ
to `CESM2LE_SLOWDOWNS_DIR/cesm2le_sie_slowdown_relative_SEP_w{w}_s{s}_{demean}_1990-2100.nc`
and prints slowdown frequency by decade and group. Also writes a 3-panel
diagnostic per label file and a window-sweep overlay to `results/figures/`. The original labels are untouched.

**Depends on:** `01_cesm2le_preprocessing.py` only.

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

## Stage 07 — Scalar baselines (revision plan §1.1)

```bash
python scripts/07_baselines.py                 # original labels, all baselines + cached CNN + figure
python scripts/07_baselines.py --no-cnn --no-fig --n-boot 200

# relative labels / window sweep (outputs go to results/baselines/<tag>/)
L=$SLOWDOWN_DATA_ROOT/cesm2le/slowdowns
for w in 3 5 7 10 15; do
  python scripts/07_baselines.py --labels-file $L/cesm2le_sie_slowdown_relative_SEP_w${w}_s1_group_1990-2100.nc --tag rel_w${w}_s1
done
```

Fits reference and logistic-regression baselines on the training members of
each of the 9 TVT splits and scores them on the test members, using the same
block assignment as the CNN: always-positive, random-at-prevalence, onset-year
climatology, September SIE anomaly at onset, the Arctic SST / Niño3.4 / IPO
indices, and their combinations. If cached CNN predictions exist
(`results/predictions/cesm2le/`) they are scored with the same metric code so
the comparison is like for like; when a different label file is passed the
CNN is scored against the *new* labels (a transfer test, flagged by the
`cnn_labels_match` attribute). Also regresses the CNN test probabilities on
year climatology and SIE anomaly (`cnn_attribution.json`) to quantify how
much of the CNN output those two explain. Writes per-split and stacked
NetCDFs, a markdown summary table and logistic coefficients to
`results/baselines[/<tag>]/`, and `results/figures/baselines_skill[_<tag>].png`.
No TensorFlow required.

**Depends on:** `02_cesm2le_slowdowns.py`, `02_cesm2le_climate_indices.py`,
`01_cesm2le_preprocessing.py` (SIE metrics); optionally
`06_cnn_predict_cesm2le.py`.

## Figures

Every manuscript figure has exactly one implementation, in three layers:

1. `src/plotting/<topic>.py` — pure panel functions taking loaded arrays
   (no file I/O, no `paths`): `slowdowns.py` (time series, trend segments,
   label distributions), `maps.py` (global SST / relevance composites,
   region boxes), `performance.py` (PR curve, confusion matrices, metric
   strip, member timeline), `conditional.py` (P(event | phase) bars),
   `observations.py` (Fig. 4 stack), `baselines.py`. `style.py` holds the
   shared colour-blind-safe palette, `paper_rc`, `tidy`, `save`.
2. `src/plotting/paper.py` — one function per figure (`fig_1` … `fig_4`,
   `fig_s1` … `fig_s12`, plus extras), composing the panels above.
3. `scripts/make_figure.py` — the entry point: loads data via
   `configs.paths` (one memoised `Data` object; composites and phase
   statistics come from a single streaming pass over splits × seeds),
   calls the `paper` function, saves to `FIGURES_DIR/paper/`.

Statistics the figures depend on live in `src/analysis`:
`composites.py` (outcome masks, streaming SST/LRP composites, 97th-pct
normalisation, smoothing, regional means) and `phase_stats.py`
(phase labellers, P(event | phase) with bootstrap CIs, variance explained).

```bash
python scripts/make_figure.py --list
python scripts/make_figure.py all                          # every figure, original labels
python scripts/make_figure.py S1 S2 --labels relative      # relative-label versions
python scripts/make_figure.py 2 S7 S8 S9 S10 regional      # composites share one data pass
python scripts/make_figure.py S3 S4 S6 --split 3 --seed 1  # single-model figures
python scripts/make_figure.py 4 --forced-method linear --single-model
python scripts/make_figure.py 1 --schematic /path/to/cnn_schematic.png --fmt pdf
```

| id | figure | id | figure |
|----|--------|----|--------|
| `1` | schematic + NSIDC + member | `S6` | test-member timeline |
| `2` | TP composite: SST + LRP | `S7` | SST composites, all vs CNN-filtered |
| `3` | P(TP \| phase), test | `S8` `S9` `S10` | FP / TN / FN composites |
| `4` | observations: votes + indices | `S11` | P(event \| phase), train, all vs TP |
| `S1` | slowdown definition | `S12` | SIE vs GMT slowdown counts |
| `S2` | label distributions | `phase_all` | P(slowdown \| phase), all slowdowns |
| `S3` | PR curve | `regional` | relevance with region boxes + bars |
| `S4` | confusion matrices | `sie_gmt_joint` | joint PDF of GMT and SIE trends |
| `S5` | metric strip | `learning_curve` | loss vs epoch |

The notebooks in `figures/` are thin wrappers that import `Data` and
`FIGURES` from `scripts/make_figure.py` and display the same figures
inline (`parse_args([...])` takes the same options as the CLI). The
pre-refactor notebooks are kept in `figures/legacy/` for reference only.
To add a figure: write `paper.fig_<name>`, add a loader/builder in
`make_figure.py`, register it in `FIGURES`.

Maps use cartopy when installed and fall back to a plain lon/lat
pcolormesh otherwise; `cmocean` is optional (falls back to `RdBu_r`).

## Troubleshooting

- **`EnvironmentError: Cannot find the data root`** — set `SLOWDOWN_DATA_ROOT`
  (see [`setup.md`](setup.md)).
- **Import errors** — activate the conda environment and run from the repo root.
- **LRP / eager-execution errors** — run `05_cesm2le_lrp.py` in its own process,
  separate from any script that uses TensorFlow eager mode.
- **Out-of-memory during training** — lower `batch_size` in `configs/training.py`.
