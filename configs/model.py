"""
Model architecture configurations.
"""

# =============================================================================
# CNN ARCHITECTURE
# =============================================================================

CNN_CONFIG = {
    # Input shape (set dynamically based on data)
    'input_shape': (192, 288, 1),  # (lat, lon, channels)

    # Convolutional layers
    'conv_layers': [
        {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'},
        {'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu'},
        {'filters': 128, 'kernel_size': (3, 3), 'activation': 'relu'}
    ],

    # Pooling
    'pool_size': (2, 2),

    # Dense layers
    'dense_layers': [
        {'units': 128, 'activation': 'relu', 'dropout': 0.5},
        {'units': 64, 'activation': 'relu', 'dropout': 0.3}
    ],

    # Output
    'n_classes': 2,  # Binary: slowdown vs no slowdown
    'output_activation': 'softmax',

    # Regularization
    'l2_reg': 0.001
}

# =============================================================================
# MODEL VARIANTS
# =============================================================================

# Seasonal model (all 4 seasons)
SEASONAL_MODEL_CONFIG = CNN_CONFIG.copy()

# JJA-only model
JJA_MODEL_CONFIG = CNN_CONFIG.copy()

# =============================================================================
# ENSEMBLE CONFIGURATION
# =============================================================================

ENSEMBLE_CONFIG = {
    'n_ensemble_members': 100,
    'n_blocks': 10,
    'block_size': 10,
    'n_splits': 9  # Number of train/val/test configurations
}
