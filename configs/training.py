"""
Training hyperparameters and configurations.
"""

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================

TRAINING_CONFIG = {
    # Optimizer
    'optimizer': 'adam',
    'learning_rate': 0.001,

    # Loss and metrics
    'loss': 'categorical_crossentropy',
    'metrics': ['accuracy'],

    # Training
    'epochs': 100,
    'batch_size': 32,
    'validation_split': 0.0,  # Using explicit validation set

    # Early stopping
    'early_stopping': {
        'monitor': 'val_loss',
        'patience': 10,
        'restore_best_weights': True
    },

    # Class weights
    'use_class_weights': True,

    # Random seeds (for reproducibility)
    'random_seeds': [42, 123, 456, 789, 1011],

    # Multi-GPU
    'use_multi_gpu': False
}

# =============================================================================
# DATA SPLITTING
# =============================================================================

SPLIT_CONFIG = {
    'train_ratio': 0.8,  # 80 ensemble members
    'val_ratio': 0.1,    # 10 ensemble members
    'test_ratio': 0.1,   # 10 ensemble members
    'n_splits': 9        # Number of different splits for robustness
}

# =============================================================================
# XAI PARAMETERS
# =============================================================================

XAI_CONFIG = {
    # LRP method
    'lrp_method': 'lrp.z',  # Options: 'lrp.z', 'lrp.epsilon', 'lrp.alpha_beta'

    # Attribution aggregation
    'aggregation_method': 'mean',  # Options: 'mean', 'median', 'max', 'abs_mean'

    # Normalization
    'normalize_method': 'absmax',  # Options: 'absmax', 'minmax', 'percentile'

    # Regions for analysis
    'regions': {
        'arctic': {'lat_min': 60, 'lat_max': 90, 'lon_min': 0, 'lon_max': 360},
        'tropics': {'lat_min': -30, 'lat_max': 30, 'lon_min': 0, 'lon_max': 360},
        'north_pacific': {'lat_min': 20, 'lat_max': 60, 'lon_min': 120, 'lon_max': 240}
    }
}

# =============================================================================
# ANALYSIS PARAMETERS
# =============================================================================

ANALYSIS_CONFIG = {
    # Climate indices
    'baseline_period': (1990, 2020),
    'nino34_threshold': 0.4,
    'nino34_min_length': 6,  # months

    # Nino 3.4 region
    'nino34_region': {
        'lat_min': -5,
        'lat_max': 5,
        'lon_min': 190,  # 170W
        'lon_max': 240   # 120W
    },

    # Trend analysis
    'trend_window': 10,  # years for moving trends

    # Bootstrap
    'bootstrap_iterations': 1000,
    'confidence_level': 0.95
}

# =============================================================================
# SEASONS
# =============================================================================

SEASONS = {
    'JJA': {'months': ['JUN', 'JUL', 'AUG'], 'name': 'Summer'},
    'MAM': {'months': ['MAR', 'APR', 'MAY'], 'name': 'Spring'},
    'DJF': {'months': ['DEC', 'JAN', 'FEB'], 'name': 'Winter'},
    'SON': {'months': ['SEP', 'OCT', 'NOV'], 'name': 'Fall'}
}

# =============================================================================
# PLOTTING PARAMETERS
# =============================================================================

PLOT_CONFIG = {
    'dpi': 300,
    'figsize_single': (12, 8),
    'figsize_multi': (18, 12),
    'font_size': 12,
    'title_size': 14,
    'cmap_attribution': 'RdBu_r',
    'cmap_sst': 'cmocean.thermal',

    # Colors for slowdown analysis
    'colors': {
        'slowdown': 'green',
        'no_slowdown': 'red',
        'ensemble_bg': 'lightgray',
        'mean': 'black'
    }
}
