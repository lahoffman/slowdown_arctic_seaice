
# ===== MD [0] =====
# # CNN Predict — ERSST Observational Inputs
# 
# Loads **precomputed CNN predictions** on ERSSTv5 observational SST and visualises them
# alongside climate indices and observed slowdown events.
# 
# **Figure A** — Single selected model: prediction timeline + climate indices  
# **Figure B** — All models: ensemble prediction frequency + climate indices
# 
# **Prerequisites**
# - `scripts/06_cnn_predict_ersst.py` — precomputes and saves predictions

# ===== CODE [1] =====
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
from sklearn.metrics import precision_recall_curve

PROJECT_ROOT = Path('..').resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths

# ── Publication figure defaults ───────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 14*5,
    'axes.titlesize': 16*5,
    'axes.labelsize': 14*5,
    'xtick.labelsize': 12*5,
    'ytick.labelsize': 12*5,
    'legend.fontsize': 12*5,
})

# ===== MD [2] =====
# ## Configuration
# 
# Edit this cell to change which model(s) to load and which files to read.

# ===== CODE [3] =====
# ── Model selection ──────────────────────────────────────────────────────────
N_SPLITS  = 9          # number of TVT splits trained
N_SEEDS   = 5          # random seeds per split
BASE_SEED = 42

# For Figure A: single model
SELECTED_SPLIT = 0
SELECTED_SEED  = 0

# ── File paths ────────────────────────────────────────────────────────────────
PREDICTIONS_DIR = paths.RESULTS_DIR / 'predictions' / 'ersst'

# NSIDC observed slowdown events (September SIE)
SLOWDOWN_OBS_FILE = paths.nsidc_sie_slowdown_events(9)   # month 9 = September

# Climate indices (ERSST)
ARCTIC_SST_FILE = paths.ERSST_DIR / 'ersstv5_arctic_sst_index.nc'
NINO34_FILE     = paths.ERSST_DIR / 'ersstv5_nino34_index.nc'
ENSO_CPTP_FILE  = paths.ERSST_DIR / 'ersstv5_enso_cptp_indices.nc'
IPO_FILE        = paths.ERSST_DIR / 'ersstv5_ipo_index.nc'

# ── Threshold ─────────────────────────────────────────────────────────────────
# Default classification threshold (0.5 for sigmoid).
# Optionally recompute from training PR curve — see below.
THRESHOLD = 0.5

# ── Time ──────────────────────────────────────────────────────────────────────
OBS_START_YEAR = 1990
OBS_END_YEAR   = 2024

# ===== MD [4] =====
# ## Load observation year labels
# 
# Only the year coordinate is needed here; predictions were precomputed by
# `scripts/06_cnn_predict_ersst.py`.

# ===== CODE [5] =====
# Load observation year labels from the testing file (predictions already saved)
ds_test = xr.open_dataset(paths.ERSST_TESTING)
obs_years = ds_test['years'].values               # (nyear,)
ds_test.close()

print(f'Observation years: {obs_years[0]}-{obs_years[-1]}  ({len(obs_years)} years)')

# ===== MD [6] =====
# ## Load observed slowdown events (NSIDC September SIE)

# ===== CODE [7] =====
# Observed September SIE slowdowns
ds_slow = xr.open_dataset(SLOWDOWN_OBS_FILE)
slow_years  = ds_slow['year'].values               # trend-window centre years
slow_events = ds_slow['slowdown'].values            # 1 = slowdown, 0 = normal

print(f'Observed slowdowns: {slow_events.sum()} events over {len(slow_years)} years '
      f'({slow_years[0]}-{slow_years[-1]})')

# ===== MD [8] =====
# ## Load climate indices

# ===== CODE [9] =====
import pandas as pd
import netCDF4 as nc

# --- Arctic SST Index ---
ds_arctic = xr.open_dataset(ARCTIC_SST_FILE)
arctic_sst_monthly = ds_arctic['arctic_sst'].values     # (ntime_monthly,)
arctic_labels      = ds_arctic['labels'].values

# Build monthly date index for the full ERSST record (starts 1854-01)
nt_arctic = len(arctic_sst_monthly)
dates_arctic = pd.date_range(start='1990-01-01', periods=nt_arctic, freq='MS')

# Compute annual JJA mean for the observation window
arctic_years_mask = (dates_arctic.year >= OBS_START_YEAR) & (dates_arctic.year <= OBS_END_YEAR)
arctic_sub = arctic_sst_monthly[arctic_years_mask]
dates_sub  = dates_arctic[arctic_years_mask]
# Reshape to (nyear, 12) and take JJA (months 5:8)
n_full_years = len(arctic_sub) // 12
arctic_annual_jja = np.nanmean(
    arctic_sub[:n_full_years*12].reshape(n_full_years, 12)[:, 5:8], axis=1
)
arctic_plot_years = np.arange(OBS_START_YEAR, OBS_START_YEAR + n_full_years)

print(f'Arctic SST Index: {nt_arctic} monthly steps, '
      f'JJA annual {arctic_plot_years[0]}-{arctic_plot_years[-1]}')

# --- Nino3.4 ---
ds_nino = xr.open_dataset(NINO34_FILE)
nino34_monthly = ds_nino['nino34'].values

# --- ENSO CP/TP ---
ds_cptp = xr.open_dataset(ENSO_CPTP_FILE)
n34_monthly = ds_cptp['n34'].values
n_ct_monthly = ds_cptp['n_ct'].values
n_wp_monthly = ds_cptp['n_wp'].values

# Build date index for ENSO indices (same ERSST timeline)
nt_enso = len(n34_monthly)
dates_enso = pd.date_range(start='1854-01-01', periods=nt_enso, freq='MS')

# Compute annual JJA means for the observation window
enso_mask = (dates_enso.year >= OBS_START_YEAR) & (dates_enso.year <= OBS_END_YEAR)
n34_sub = n34_monthly[enso_mask]
nct_sub = n_ct_monthly[enso_mask]
nwp_sub = n_wp_monthly[enso_mask]
n_enso_yrs = len(n34_sub) // 12
n34_jja = np.nanmean(n34_sub[:n_enso_yrs*12].reshape(n_enso_yrs, 12)[:, 5:8], axis=1)
nct_jja = np.nanmean(nct_sub[:n_enso_yrs*12].reshape(n_enso_yrs, 12)[:, 5:8], axis=1)
nwp_jja = np.nanmean(nwp_sub[:n_enso_yrs*12].reshape(n_enso_yrs, 12)[:, 5:8], axis=1)
enso_plot_years = np.arange(OBS_START_YEAR, OBS_START_YEAR + n_enso_yrs)

print(f'ENSO CP/TP: JJA annual {enso_plot_years[0]}-{enso_plot_years[-1]}')

# --- IPO ---
ds_ipo = xr.open_dataset(IPO_FILE)
ipo_monthly  = ds_ipo['ipo'].values
ipo_filtered = ds_ipo['ipo_filtered'].values
nt_ipo = len(ipo_monthly)
dates_ipo = pd.date_range(start='1854-01-01', periods=nt_ipo, freq='MS')
ipo_mask = (dates_ipo.year >= OBS_START_YEAR) & (dates_ipo.year <= OBS_END_YEAR)
ipo_sub = ipo_filtered[ipo_mask]
n_ipo_yrs = len(ipo_sub) // 12
ipo_jja = np.nanmean(ipo_sub[:n_ipo_yrs*12].reshape(n_ipo_yrs, 12)[:, 5:8], axis=1)
ipo_plot_years = np.arange(OBS_START_YEAR, OBS_START_YEAR + n_ipo_yrs)

print(f'IPO: JJA annual {ipo_plot_years[0]}-{ipo_plot_years[-1]}')

# ===== MD [10] =====
# ## Load precomputed predictions
# 
# Predictions were saved by `scripts/06_cnn_predict_ersst.py`.
# No model loading or inference is needed.

# ===== CODE [11] =====
# Collect predictions from ALL models (all splits x all seeds) from saved files
all_probs = []     # list of (nyear,) arrays
model_ids = []     # (split_idx, run_idx) tuples

for split_idx in range(N_SPLITS):
    for run_idx in range(N_SEEDS):
        pred_path = PREDICTIONS_DIR / f'cnn_prediction_ersst_M{split_idx}_{run_idx}.nc'
        if not pred_path.exists():
            print(f'  [skip] {pred_path.name} not found')
            continue
        with xr.open_dataset(pred_path) as ds_pred:
            probs = ds_pred['y_prob'].values             # (nyear,)
        all_probs.append(probs)
        model_ids.append((split_idx, run_idx))

all_probs = np.array(all_probs)    # (n_models, nyear)
print(f'Loaded {len(model_ids)} prediction files, shape: {all_probs.shape}')

# ===== CODE [12] =====
# --- Single selected model ---
sel_key = (SELECTED_SPLIT, SELECTED_SEED)
if sel_key in model_ids:
    sel_idx = model_ids.index(sel_key)
else:
    sel_idx = 0
    print(f'Selected model {sel_key} not found, falling back to {model_ids[0]}')

probs_selected = all_probs[sel_idx]                        # (nyear,)
preds_selected = (probs_selected >= THRESHOLD).astype(int)  # binary

# --- Ensemble statistics ---
preds_all = (all_probs >= THRESHOLD).astype(int)            # (n_models, nyear)
slowdown_fraction = preds_all.mean(axis=0)                  # fraction [0, 1]

print(f'Selected model: split {model_ids[sel_idx][0]}, seed {model_ids[sel_idx][1]}')
print(f'Ensemble: {len(model_ids)} models, '
      f'mean slowdown fraction = {slowdown_fraction.mean():.3f}')

# ===== MD [13] =====
# ## Load ensmean forced-method predictions
# 
# Predictions from CNNs applied to ERSSTv5 data where the forced signal was
# removed using the CESM2-LE **ensemble-mean** (rather than a per-pixel linear trend).
# Generated by `scripts/06_cnn_predict_ersst.py --forced-method ensmean`.

# ===== CODE [14] =====
# ── Load ensmean predictions ─────────────────────────────────────────────────
PREDICTIONS_DIR_ENSMEAN = paths.ersst_predictions_dir('ensmean')

all_probs_ensmean = []
model_ids_ensmean = []

for split_idx in range(N_SPLITS):
    for run_idx in range(N_SEEDS):
        pred_path = PREDICTIONS_DIR_ENSMEAN / f'cnn_prediction_ersst_M{split_idx}_{run_idx}.nc'
        if not pred_path.exists():
            continue
        with xr.open_dataset(pred_path) as ds_pred:
            probs = ds_pred['y_prob'].values
        all_probs_ensmean.append(probs)
        model_ids_ensmean.append((split_idx, run_idx))

HAS_ENSMEAN = len(all_probs_ensmean) > 0

if HAS_ENSMEAN:
    all_probs_ensmean = np.array(all_probs_ensmean)
    preds_all_ensmean = (all_probs_ensmean >= THRESHOLD).astype(int)
    slowdown_fraction_ensmean = preds_all_ensmean.mean(axis=0)

    # Single selected model (same split/seed as linear)
    sel_key_em = (SELECTED_SPLIT, SELECTED_SEED)
    if sel_key_em in model_ids_ensmean:
        sel_idx_em = model_ids_ensmean.index(sel_key_em)
    else:
        sel_idx_em = 0
    probs_selected_ensmean = all_probs_ensmean[sel_idx_em]
    preds_selected_ensmean = (probs_selected_ensmean >= THRESHOLD).astype(int)

    print(f'Loaded {len(model_ids_ensmean)} ensmean prediction files, '
          f'shape: {all_probs_ensmean.shape}')
    print(f'Ensemble mean slowdown fraction = {slowdown_fraction_ensmean.mean():.3f}')
else:
    print('No ensmean predictions found — run:')
    print('  python scripts/06_cnn_predict_ersst.py --forced-method ensmean')

# ===== CODE [15] =====
# ═══════════════════════════════════════════════════════════════════════════════
# Reusable plotting function for the 5-panel prediction figure
# ═══════════════════════════════════════════════════════════════════════════════

def plot_prediction_figure(
    obs_years, probs_selected, slowdown_fraction,
    slow_years, slow_events,
    arctic_plot_years, arctic_annual_jja,
    enso_plot_years, n34_jja,
    ipo_plot_years, ipo_jja,
    threshold=0.5, frac_threshold=0.15,
    title_suffix='',
):
    """
    Plot the 5-panel prediction figure.

    Parameters
    ----------
    probs_selected : array (nyear,)
        Predicted probabilities from a single selected model.
    slowdown_fraction : array (nyear,)
        Fraction of ensemble models predicting slowdown.
    title_suffix : str
        Appended to the suptitle, e.g. '(linear detrending)'
    """
    FUTURE_START = 2015.5
    FUTURE_END   = 2025.5

    # -- helpers (defined locally so the function is self-contained) --
    def _add_slowdown_bars(ax, label_first=True):
        added = False
        for i, yr in enumerate(slow_years):
            if yr in obs_years and slow_events[i] == 1:
                lbl = 'Actual slowdown' if (not added and label_first) else ''
                ax.axvspan(yr - 0.4, yr + 0.4, color='mediumseagreen', alpha=0.4,
                           zorder=0, label=lbl)
                added = True

    def _add_ensemble_highlight(ax):
        added = False
        for i, yr in enumerate(obs_years):
            if slowdown_fraction[i] > frac_threshold:
                lbl = f'Ensemble frac > {frac_threshold}' if not added else ''
                ax.axvspan(yr - 0.4, yr + 0.4, facecolor='none',
                           edgecolor='firebrick', linewidth=2.0*5,
                           linestyle='--', zorder=1, label=lbl)
                added = True

    def _add_future_shading(ax, label=False):
        lbl = 'Future decades' if label else ''
        ax.axvspan(FUTURE_START, FUTURE_END, color='lightgray', alpha=0.3,
                   zorder=0, label=lbl)

    fig, axes = plt.subplots(5, 1, figsize=(14*5, 16*5), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1, 1, 1, 1]})

    # ── (a) Binary slowdown classification ──────────────────────────────────
    ax = axes[0]
    _add_future_shading(ax, label=True)
    _add_slowdown_bars(ax)
    binary_sel = (probs_selected >= threshold).astype(int)
    ax.plot(obs_years, binary_sel,
            color='firebrick', linestyle='--', linewidth=2.0*5, marker='o',
            markersize=4*5, zorder=2)
    ax.set_ylim(-0.15, 1.15)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['NO slowdown', 'YES slowdown'])
    ax.set_ylabel('')
    ax.set_title('(a)', loc='left', weight='bold')
    cnn_handle = mlines.Line2D([], [], color='firebrick', linestyle='--',
                                linewidth=2.0*5, label='CNN prediction')
    handles, labels = ax.get_legend_handles_labels()
    handles.append(cnn_handle); labels.append('CNN prediction')
    ax.legend(handles=handles, labels=labels, loc='upper right')

    # ── (b) Ensemble prediction frequency ──────────────────────────────────
    ax = axes[1]
    _add_future_shading(ax)
    _add_slowdown_bars(ax, label_first=False)
    ax.bar(obs_years, slowdown_fraction, width=0.7,
           color='firebrick', alpha=0.7, edgecolor='darkred', linewidth=0.5*5,
           zorder=2)
    ax.axhline(frac_threshold, color='k', linestyle='--', alpha=0.5,
               linewidth=0.8*5, label=f'Fraction = {frac_threshold}')
    ax.set_ylabel('Fraction of models\npredicting slowdown')
    ax.set_ylim(-0.02, 1.02)
    ax.set_title('(b)', loc='left', weight='bold')
    ax.legend(loc='upper right')

    # ── (c) Arctic SST Index ───────────────────────────────────────────────
    ax = axes[2]
    _add_future_shading(ax)
    _add_slowdown_bars(ax, label_first=False)
    _add_ensemble_highlight(ax)
    ax.fill_between(arctic_plot_years, arctic_annual_jja, 0,
                    where=(arctic_annual_jja > 0), color='firebrick', alpha=0.3)
    ax.fill_between(arctic_plot_years, arctic_annual_jja, 0,
                    where=(arctic_annual_jja <= 0), color='steelblue', alpha=0.3)
    ax.plot(arctic_plot_years, arctic_annual_jja, color='k', linewidth=1.2*5)
    ax.axhline(0, color='k', linewidth=0.5*5)
    ax.axhline(1,  color='gray', linestyle='--', linewidth=0.5*5, alpha=0.6)
    ax.axhline(-1, color='gray', linestyle='--', linewidth=0.5*5, alpha=0.6)
    ax.set_ylabel('Arctic SST Index')
    ax.set_title('(c)', loc='left', weight='bold')

    # ── (d) Nino 3.4 ──────────────────────────────────────────────────────
    ax = axes[3]
    _add_future_shading(ax)
    _add_slowdown_bars(ax, label_first=False)
    _add_ensemble_highlight(ax)
    ax.fill_between(enso_plot_years, n34_jja, 0,
                    where=(n34_jja > 0), color='firebrick', alpha=0.3)
    ax.fill_between(enso_plot_years, n34_jja, 0,
                    where=(n34_jja <= 0), color='steelblue', alpha=0.3)
    ax.plot(enso_plot_years, n34_jja, color='k', linewidth=1.2*5)
    ax.axhline(0, color='k', linewidth=0.5*5)
    ax.axhline(0.4,  color='gray', linestyle='--', linewidth=0.5*5, alpha=0.6)
    ax.axhline(-0.4, color='gray', linestyle='--', linewidth=0.5*5, alpha=0.6)
    ax.set_ylabel('Nino 3.4 Index')
    ax.set_title('(d)', loc='left', weight='bold')

    # ── (e) IPO ───────────────────────────────────────────────────────────
    ax = axes[4]
    _add_future_shading(ax)
    _add_slowdown_bars(ax, label_first=False)
    _add_ensemble_highlight(ax)
    ax.fill_between(ipo_plot_years, ipo_jja, 0,
                    where=(ipo_jja > 0), color='firebrick', alpha=0.3)
    ax.fill_between(ipo_plot_years, ipo_jja, 0,
                    where=(ipo_jja <= 0), color='steelblue', alpha=0.3)
    ax.plot(ipo_plot_years, ipo_jja, color='k', linewidth=1.2*5)
    ax.axhline(0, color='k', linewidth=0.5*5)
    ax.set_ylabel('IPO Index')
    ax.set_xlabel('Year')
    ax.set_title('(e)', loc='left', weight='bold')

    if title_suffix:
        fig.suptitle(title_suffix, fontsize=18*5, weight='bold', y=1.01)

    plt.tight_layout()
    plt.show()
    return fig

# ===== MD [16] =====
# ---
# ## Figure — Linear detrending predictions
# 
# CNN predictions on ERSSTv5 where the forced signal was removed via
# **per-pixel linear detrending** (legacy method).

# ===== CODE [17] =====
plot_prediction_figure(
    obs_years, probs_selected, slowdown_fraction,
    slow_years, slow_events,
    arctic_plot_years, arctic_annual_jja,
    enso_plot_years, n34_jja,
    ipo_plot_years, ipo_jja,
    threshold=THRESHOLD,
    title_suffix='Forced removal: per-pixel linear detrending',
);

# ===== MD [18] =====
# ## Figure — Ensemble-mean predictions
# 
# CNN predictions on ERSSTv5 where the forced signal was removed using
# the **CESM2-LE ensemble-mean** forced response.

# ===== CODE [19] =====
if HAS_ENSMEAN:
    plot_prediction_figure(
        obs_years, probs_selected_ensmean, slowdown_fraction_ensmean,
        slow_years, slow_events,
        arctic_plot_years, arctic_annual_jja,
        enso_plot_years, n34_jja,
        ipo_plot_years, ipo_jja,
        threshold=THRESHOLD,
        title_suffix='Forced removal: CESM2-LE ensemble mean',
    )
else:
    print('Ensmean predictions not available — skipping figure.')
    print('Run:  python scripts/06_cnn_predict_ersst.py --forced-method ensmean')
