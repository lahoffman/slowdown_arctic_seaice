"""
configs/paths.py — single source of truth for all data paths.

HOW TO SWITCH ENVIRONMENTS
---------------------------
Set the SLOWDOWN_DATA_ROOT environment variable before running any script:

    # Mac (add to ~/.zshrc so it persists)
    export SLOWDOWN_DATA_ROOT=/Users/lahoffma/data/slowdowns

    # Remote server (add to ~/.bashrc)
    export SLOWDOWN_DATA_ROOT=/cofast/lhoffman/slowdowns

If the variable is not set, the file falls back to checking known local paths.
If neither is found, an error is raised — intentionally, so missing data is
caught early rather than silently writing outputs to the wrong place.

DATA DIRECTORY LAYOUT
----------------------
All inputs and outputs live under DATA_ROOT, organised by dataset:

    DATA_ROOT/
    ├── nsidc/          raw Excel file + processed slowdown/riles outputs
    ├── ersst/          raw sst.mnmean.nc + regridded + climate indices
    ├── cesm2le/        downloaded ensemble files + processed outputs
    └── results/        models, XAI attributions, figures

ADDING A NEW PATH
-----------------
1. Add the variable here, under the appropriate dataset section.
2. Use it anywhere via:  from configs import paths;  paths.MY_NEW_PATH
"""

import os
from pathlib import Path

# =============================================================================
# ROOT — only thing that changes between environments
# =============================================================================

_env_root = os.environ.get('SLOWDOWN_DATA_ROOT')

if _env_root:
    DATA_ROOT = Path(_env_root)
elif Path('/Users/lahoffma/data/slowdowns').exists():
    DATA_ROOT = Path('/Users/lahoffma/data/slowdowns')
elif Path('/cofast/lhoffman/slowdowns').exists():
    DATA_ROOT = Path('/cofast/lhoffman/slowdowns')
elif Path('/mnt/tank/Oceanography/data/OGCM/LLC/Fronts/lohoff/arcticWatch').exists():
    DATA_ROOT = Path('/mnt/tank/Oceanography/data/OGCM/LLC/Fronts/lohoff/arcticWatch')
else:
    raise EnvironmentError(
        "\n\nCannot find the data root directory.\n"
        "Set the SLOWDOWN_DATA_ROOT environment variable, e.g.:\n"
        "    export SLOWDOWN_DATA_ROOT=/Users/lahoffma/data/slowdowns\n"
    )

# Repo root (for config/code paths only — never for data)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# NSIDC  —  sea ice extent / area
# =============================================================================

NSIDC_DIR = DATA_ROOT / 'nsidc'
NSIDC_DIR.mkdir(parents=True, exist_ok=True)

# raw
NSIDC_FILE = NSIDC_DIR / 'Sea_Ice_Index_Monthly_Data_with_Statistics_G02135_v3.0.xlsx'

# processed  (outputs of scripts/01_slowdown_nsidc_sie_sia.py)
NSIDC_SIE_SLOWDOWN_THRESHOLDS  = NSIDC_DIR / 'nsidc_sie_slowdown_thresholds.nc'
NSIDC_SIA_SLOWDOWN_THRESHOLDS  = NSIDC_DIR / 'nsidc_sia_slowdown_thresholds.nc'

def nsidc_sie_slowdown_events(month: int) -> Path:
    """Path to SIE slowdown events file for a given month (1-based)."""
    return NSIDC_DIR / f'nsidc_sie_slowdown_events_month{month:02d}.nc'

def nsidc_sia_slowdown_events(month: int) -> Path:
    """Path to SIA slowdown events file for a given month (1-based)."""
    return NSIDC_DIR / f'nsidc_sia_slowdown_events_month{month:02d}.nc'


# =============================================================================
# ERSSTv5  —  observed SST and climate indices
# =============================================================================

ERSST_DIR = DATA_ROOT / 'ersst'
ERSST_DIR.mkdir(parents=True, exist_ok=True)

# raw  (output of src/data/observations/ersst/download.py)
ERSST_FILE = ERSST_DIR / 'sst.mnmean.nc'

# processed  (outputs of scripts that follow download)
ERSST_REGRIDDED    = ERSST_DIR / 'sst_regrid_cesm2le.nc'
ERSST_NINO34       = ERSST_DIR / 'ersst_nino34_index.nc'
ERSST_ENSO_CPTP    = ERSST_DIR / 'ersst_enso_cptp_indices.nc'
ERSST_IPO          = ERSST_DIR / 'ersst_ipo_index.nc'


# =============================================================================
# CESM2-LE  —  model ensemble
# =============================================================================

CESM2LE_DIR = DATA_ROOT / 'cesm2le'
CESM2LE_DIR.mkdir(parents=True, exist_ok=True)

# raw  (downloaded files, organised by variable)
# *** REMOVE THIS
CESM2LE_RAW_DIR  = CESM2LE_DIR / 'raw'
CESM2LE_RAW_DIR.mkdir(parents=True, exist_ok=True)

# processed  (outputs of src/data/cesm2le/combine.py)
CESM2LE_AICE_DIR = CESM2LE_DIR / 'aice'
CESM2LE_SST_DIR  = CESM2LE_DIR / 'sst'
CESM2LE_AICE_DIR.mkdir(parents=True, exist_ok=True)
CESM2LE_SST_DIR.mkdir(parents=True, exist_ok=True)


# grid / ancillary  (used for regridding)
CESM2LE_GRID_FILE = CESM2LE_RAW_DIR / 'b.e21.BHISTcmip6.f09_g17.LE2-1001.001.cam.h0.TREFHT.199001-199912.nc'
LANDMASK_FILE     = CESM2LE_DIR / 'sst' / 'grid' / 'cesm2le_landmask.nc'

# processed SST monthly (template — {month} filled at runtime)
CESM2LE_SST_MONTHLY = {
    'first50': str(CESM2LE_SST_DIR / 'mon' / 'sst_cesmle_first50members_mon_{month}_199001-210012.nc'),
    'last50':  str(CESM2LE_SST_DIR / 'mon' / 'sst_cesmle_last50members_mon_{month}_199001-210012.nc'),
}

# SIE trend data (output of CESM2-LE processing pipeline)
CESM2LE_SIE_TRENDS = CESM2LE_DIR / 'sie_cesmle_linear_decadal_trend_monthly_1990-2100.nc'

# Slowdown / RILES classification (output of scripts/02_cesm2le_slowdowns.py)
CESM2LE_SLOWDOWNS_DIR = CESM2LE_DIR / 'slowdowns'
CESM2LE_SLOWDOWNS_DIR.mkdir(parents=True, exist_ok=True)

def cesm2le_slowdown_file(variable: str = 'sie', month: str = 'SEP',
                           start_year: int = 1990, end_year: int = 2100) -> Path:
    """Path to the slowdown classification NetCDF for one variable and month.

    The file is produced by scripts/02_cesm2le_slowdowns.py and contains the
    binary slowdown mask (nens × nyr) plus ensemble-mean trends and thresholds.

    Parameters
    ----------
    variable : str
        ``'sie'`` (sea ice extent, default) or ``'sia'`` (sea ice area).
    month : str
        Three-letter calendar month, e.g. ``'SEP'``.
    start_year, end_year : int
        Year range encoded in the filename (default: 1990–2100).
    """
    return CESM2LE_SLOWDOWNS_DIR / (
        f'cesm2le_{variable}_slowdown_riles_{month}_{start_year}-{end_year}.nc'
    )


# =============================================================================
# RESULTS  —  models, XAI, figures
# =============================================================================

RESULTS_DIR = DATA_ROOT / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR       = RESULTS_DIR / 'models'
ATTRIBUTIONS_DIR = RESULTS_DIR / 'attributions'
FIGURES_DIR      = RESULTS_DIR / 'figures'
LOGS_DIR         = RESULTS_DIR / 'logs'
TVT_SPLITS_DIR   = RESULTS_DIR / 'tvt_splits'

for _d in (MODELS_DIR, ATTRIBUTIONS_DIR, FIGURES_DIR, LOGS_DIR, TVT_SPLITS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def tvt_split_path(split_idx: int) -> Path:
    """Path to the saved TVT split NetCDF for a given split index (0–8).

    The file is produced by scripts/03_cesm2le_tvt_and_train.py via
    src.cnn.splits.save_tvt_split and contains standardised JJA SST arrays
    and binary slowdown labels for the train, validation, and test partitions,
    plus the normalisation statistics (mu_train, sigma_train).

    Parameters
    ----------
    split_idx : int
        Zero-based TVT split index (0–8).
    """
    return TVT_SPLITS_DIR / f'cesm2le_sst_jja_slowdown_split{split_idx}.nc'


def model_path(split_idx: int, run_idx: int) -> Path:
    """Path to a saved CNN model (HDF5) for a given split and seed index."""
    return MODELS_DIR / f'cnn_jja_split{split_idx}_run{run_idx}.h5'


def attribution_path(split_idx: int, run_idx: int) -> Path:
    """Path to a saved LRP attribution NetCDF for a given split and seed index."""
    return ATTRIBUTIONS_DIR / f'lrp_jja_split{split_idx}_run{run_idx}.nc'


def metrics_path(split_idx: int) -> Path:
    """Path to a saved metrics Dataset NetCDF for a given split."""
    return RESULTS_DIR / 'metrics' / f'cnn_jja_metrics_split{split_idx}.nc'
