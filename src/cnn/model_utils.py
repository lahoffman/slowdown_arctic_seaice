"""
Model loading and data preparation utilities for CNN models.
"""

import numpy as np
from pathlib import Path

# ML imports - these require the ML conda environment
try:
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.utils.class_weight import compute_class_weight as sklearn_compute_class_weight
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("WARNING: TensorFlow/sklearn not available. ML functions will not work.")


def load_model(model_path):
    """
    Load a trained Keras model.

    Parameters
    ----------
    model_path : str or Path
        Path to the saved model file

    Returns
    -------
    keras.Model
        Loaded model
    """
    if not ML_AVAILABLE:
        raise ImportError("TensorFlow not available. Please activate the ML conda environment.")

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = keras.models.load_model(model_path)
    return model


def prepare_training_data(sst, sie, landmask, apply_mask=True):
    """
    Prepare SST and SIE data for CNN training.

    Parameters
    ----------
    sst : ndarray
        SST data with shape (n_samples, n_times, lat, lon)
    sie : ndarray
        Sea ice extent labels
    landmask : ndarray
        Land mask to apply (0=ocean, 1=land)
    apply_mask : bool, optional
        Whether to apply landmask (default: True)

    Returns
    -------
    tuple
        (X, y) prepared data
    """
    if apply_mask:
        # Expand landmask dimensions to match data
        lm_expanded = np.array(landmask)[np.newaxis, np.newaxis, :, :]
        lm_expanded = np.repeat(lm_expanded, sst.shape[0], axis=0)
        lm_expanded = np.repeat(lm_expanded, sst.shape[1], axis=1)

        # Apply mask (set land to NaN or 0)
        sst_masked = np.where(lm_expanded == 1, 0, sst)
        return sst_masked, sie

    return sst, sie


def compute_class_weights(labels):
    """
    Compute class weights for imbalanced datasets.

    Parameters
    ----------
    labels : ndarray
        Class labels

    Returns
    -------
    dict
        Dictionary mapping class indices to weights
    """
    if not ML_AVAILABLE:
        raise ImportError("sklearn not available. Please activate the ML conda environment.")

    # Get unique classes
    classes = np.unique(labels)

    # Compute weights
    weights = sklearn_compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=labels
    )

    # Return as dictionary
    return {int(c): w for c, w in zip(classes, weights)}


def reshape_for_cnn(data, add_channel_dim=True):
    """
    Reshape data for CNN input.

    Parameters
    ----------
    data : ndarray
        Input data
    add_channel_dim : bool, optional
        Whether to add a channel dimension (default: True)

    Returns
    -------
    ndarray
        Reshaped data
    """
    if add_channel_dim and data.ndim == 3:
        # Add channel dimension (batch, height, width) -> (batch, height, width, channels)
        return data[..., np.newaxis]

    return data


def normalize_data(data, method='standardize', axis=(0, 1, 2)):
    """
    Normalize data.

    Parameters
    ----------
    data : ndarray
        Data to normalize
    method : str, optional
        Normalization method: 'standardize', 'minmax', 'robust'
    axis : tuple, optional
        Axes along which to compute statistics

    Returns
    -------
    tuple
        (normalized_data, normalization_params)
    """
    if method == 'standardize':
        mean = np.nanmean(data, axis=axis, keepdims=True)
        std = np.nanstd(data, axis=axis, keepdims=True)
        normalized = (data - mean) / (std + 1e-8)
        params = {'mean': mean, 'std': std}

    elif method == 'minmax':
        min_val = np.nanmin(data, axis=axis, keepdims=True)
        max_val = np.nanmax(data, axis=axis, keepdims=True)
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
        params = {'min': min_val, 'max': max_val}

    elif method == 'robust':
        median = np.nanmedian(data, axis=axis, keepdims=True)
        q75 = np.nanpercentile(data, 75, axis=axis, keepdims=True)
        q25 = np.nanpercentile(data, 25, axis=axis, keepdims=True)
        iqr = q75 - q25
        normalized = (data - median) / (iqr + 1e-8)
        params = {'median': median, 'iqr': iqr}

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized, params
