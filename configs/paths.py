"""
Path configuration for all data, models, and outputs.

CHANGE PATHS HERE - nowhere else!
"""

from pathlib import Path
import os

# =============================================================================
# BASE PATHS
# =============================================================================

# Root data directory - CHANGE THIS to your environment
if os.path.exists('/cofast/lhoffman/slowdown/'):
    # Cluster environment
    ROOT_DATA_PATH = Path('/cofast/lhoffman/slowdown/')
elif os.path.exists('/Users/lahoffma/projects/arcticWATCH/slowdown_arctic_seaice'):
    # Local environment
    ROOT_DATA_PATH = Path('/Users/lahoffma/projects/arcticWATCH/slowdown_arctic_seaice')
else:
    # Default to current repo
    ROOT_DATA_PATH = Path(__file__).parent.parent

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Results directory (all outputs go here)
RESULTS_DIR = PROJECT_ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# =============================================================================
# INPUT DATA PATHS
# =============================================================================

# CESM2-LE data
CESM2_GRID_FILE = ROOT_DATA_PATH / 'b.e21.BHISTcmip6.f09_g17.LE2-1001.001.cam.h1.Z200.18500101-18591231.nc'
LANDMASK_FILE = ROOT_DATA_PATH / 'cnn_cesm2le_landmask.nc'

# SST data
ERSST_FILE = ROOT_DATA_PATH / 'sst.mnmean.nc'
ERSST_REGRIDDED = ROOT_DATA_PATH / 'D1_ersstv5_sst_regridded_to_cesm2_1854-2024.nc'

# CESM2-LE monthly SST (on cluster)
CESM2_SST_MONTHLY = {
    'first50': '/cofast/lhoffman/cesmle/sst/mon/sst_cesmle_first50members_mon_{month}_199001-210012.nc',
    'last50': '/cofast/lhoffman/cesmle/sst/mon/sst_cesmle_last50members_mon_{month}_199001-210012.nc'
}

# SIE trend data
SIE_TRENDS_FILE = ROOT_DATA_PATH / 'D2_cnn_cesm2le_sieTREND_seasons_TVT_1990-2100.nc'

# NSIDC observational data
NSIDC_FILE = ROOT_DATA_PATH / 'Sea_Ice_Index_Monthly_Data_by_Year_G02135_v3.0.xlsx'

# =============================================================================
# OUTPUT PATHS
# =============================================================================

# Processed data
DATA_DIR = RESULTS_DIR / 'data'
DATA_DIR.mkdir(exist_ok=True)

# Models
MODELS_DIR = RESULTS_DIR / 'models'
MODELS_DIR.mkdir(exist_ok=True)

# XAI attributions
ATTRIBUTIONS_DIR = RESULTS_DIR / 'attributions'
ATTRIBUTIONS_DIR.mkdir(exist_ok=True)

# Figures
FIGURES_DIR = RESULTS_DIR / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

# Logs
LOGS_DIR = RESULTS_DIR / 'logs'
LOGS_DIR.mkdir(exist_ok=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_data_path(filename):
    """Get path in data directory."""
    return DATA_DIR / filename

def get_model_path(filename):
    """Get path in models directory."""
    return MODELS_DIR / filename

def get_attribution_path(filename):
    """Get path in attributions directory."""
    return ATTRIBUTIONS_DIR / filename

def get_figure_path(filename):
    """Get path in figures directory."""
    return FIGURES_DIR / filename

def get_log_path(filename):
    """Get path in logs directory."""
    return LOGS_DIR / filename
