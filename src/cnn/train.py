"""
src/cnn/train.py — Training utilities for the JJA SST slowdown CNN.

Contents
--------
set_seed               : Set NumPy and TensorFlow random seeds for reproducibility.
compute_class_weights  : Compute balanced class weights with an optional
                         adjustment for the slowdown (minority) class.
train_model            : Compile and fit the CNN; returns the fitted model and
                         training history.
predict_splits         : Run inference on all three splits; returns predicted
                         probabilities.
collect_metrics_dataset: Evaluate metrics across splits and seeds, pack into an
                         xarray Dataset.

Workflow (per TVT split, per seed)
-----------------------------------
1. ``set_seed(seed)``
2. ``compute_class_weights(y_train)``  →  class_weights_dict
3. ``train_model(model, x_tr, y_tr, x_va, y_va, config, class_weights_dict)``
4. ``predict_splits(model, x_tr, x_va, x_te)``  →  y_scores dict
5. ``collect_metrics_dataset(...)``  →  xr.Dataset of metric values

Authors: Lauren Hoffman  <lhoffma2@ucsc.edu>
"""

import numpy as np
import xarray as xr
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.callbacks import EarlyStopping
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

from sklearn.utils.class_weight import compute_class_weight

from .model import compute_metrics, METRIC_NAMES, focal_loss, build_cnn


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int) -> None:
    """
    Set random seeds for NumPy and TensorFlow to ensure reproducible training.

    Parameters
    ----------
    seed : int
        The seed value.  The project convention is ``42 + run_index``, giving
        seeds 42, 43, 44, … for successive runs.

    Examples
    --------
    >>> for r_idx in range(5):
    ...     set_seed(42 + r_idx)
    ...     model = build_cnn(...)
    ...     train_model(model, ...)
    """
    np.random.seed(seed)
    if _TF_AVAILABLE:
        tf.random.set_seed(seed)


# =============================================================================
# Class weights
# =============================================================================

def compute_class_weights(
    y_train: np.ndarray,
    fract_weight: float = 1.5,
) -> Dict[int, float]:
    """
    Compute balanced class weights for binary classification, with an optional
    upward adjustment for the positive (slowdown) class.

    Balanced weights are computed by sklearn and then multiplied by
    ``fract_weight`` for class 1 (slowdown = 1) to further compensate for the
    rarity of slowdown events in the training data.
    
    """
    classes      = np.unique(y_train)
    weights      = compute_class_weight('balanced', classes=classes, y=y_train)
    cw_dict      = {int(c): float(w) for c, w in zip(classes, weights)}
    cw_dict[1]  *= fract_weight      # boost slowdown weight
    return cw_dict


# =============================================================================
# Training
# =============================================================================

def train_model(
    model: "tf.keras.Model",
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: Optional[Dict] = None,
    class_weights: Optional[Dict[int, float]] = None,
) -> Tuple["tf.keras.Model", object]:
    """
    Compile and train the CNN model with early stopping.

    The model is compiled with BinaryFocalCrossentropy (the Keras built-in
    version with equivalent parameters to the project's custom focal_loss),
    Adam optimiser, and accuracy as a monitored metric.  Training stops early
    when validation loss fails to improve.

    Parameters
    ----------
    model : tf.keras.Model
        Uncompiled Keras model (from ``build_cnn``).
    x_train : np.ndarray
        Training inputs, shape ``(n_tr, nx, ny, nch)``.
    y_train : np.ndarray
        Training binary labels, shape ``(n_tr,)``.
    x_val : np.ndarray
        Validation inputs, shape ``(n_va, nx, ny, nch)``.
    y_val : np.ndarray
        Validation binary labels, shape ``(n_va,)``.
    config : dict, optional
        Training hyperparameters.  Recognised keys:
            ``learning_rate``  (default: 1e-4)
            ``num_epochs``     (default: 50)
            ``batch_size``     (default: 120)
            ``optimizer``      (default: ``'adam'``)
            ``patience``       (default: 10)
            ``focal_alpha``    (default: 0.75)
            ``focal_gamma``    (default: 2.0)
    class_weights : dict, optional
        ``{0: w0, 1: w1}`` class weights passed to ``model.fit``.
        If ``None``, no class weighting is applied.

    Returns
    -------
    model : tf.keras.Model
        Fitted Keras model (best weights restored via ``restore_best_weights``).
    history : tf.keras.callbacks.History
        Keras training history object.

    Examples
    --------
    >>> model = build_cnn(nx=192, ny=288, nch=1)
    >>> model, history = train_model(model, x_tr, y_tr, x_va, y_va,
    ...                              class_weights=cw)
    >>> history.history['val_loss'][-1]
    """
    if not _TF_AVAILABLE:
        raise ImportError("TensorFlow is required for model training.")

    cfg = config or {}
    lr           = cfg.get('learning_rate', 1e-4)
    num_epochs   = cfg.get('num_epochs',    50)
    batch_size   = cfg.get('batch_size',    120)
    optimizer    = cfg.get('optimizer',     'adam')
    patience     = cfg.get('patience',      10)
    focal_alpha  = cfg.get('focal_alpha',   0.75)
    focal_gamma  = cfg.get('focal_gamma',   2.0)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr) if lr != 1e-3 else optimizer,
        loss=keras.losses.BinaryFocalCrossentropy(
            alpha=focal_alpha,
            gamma=focal_gamma,
        ),
        metrics=['accuracy'],
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
    )

    history = model.fit(
        x_train, y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(x_val, y_val),
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=0,
    )

    return model, history


# =============================================================================
# Inference
# =============================================================================

def predict_splits(
    model: "tf.keras.Model",
    x_tr: np.ndarray,
    x_va: np.ndarray,
    x_te: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Run the trained model on all three data splits and return predicted
    probability scores.

    Parameters
    ----------
    model : tf.keras.Model
        Trained Keras model.
    x_tr, x_va, x_te : np.ndarray
        Inputs for training, validation, and test splits, each shape
        ``(n_samples, nx, ny, nch)``.

    Returns
    -------
    dict
        Keys: ``'train'``, ``'val'``, ``'test'``.
        Values: 1-D arrays of predicted probabilities ``∈ (0, 1)``.

    Examples
    --------
    >>> y_scores = predict_splits(model, x_tr, x_va, x_te)
    >>> y_scores['test'].shape   # (500,)
    """
    return {
        'train': model.predict(x_tr, verbose=0).ravel(),
        'val':   model.predict(x_va, verbose=0).ravel(),
        'test':  model.predict(x_te, verbose=0).ravel(),
    }


# =============================================================================
# Metrics collection
# =============================================================================

def collect_metrics_dataset(
    y_true: Dict[str, np.ndarray],
    y_scores_runs: List[Dict[str, np.ndarray]],
    metrics_to_collect: Optional[List[str]] = None,
    split_names: Optional[List[str]] = None,
) -> xr.Dataset:
    """
    Evaluate classification metrics across multiple runs (seeds) and all splits,
    then pack the results into an xarray Dataset.

    This function is designed to be called once per TVT split after all seeds
    have been trained.  It computes per-run metric values and, when more than
    one run is provided, bootstrap confidence intervals.

    Parameters
    ----------
    y_true : dict
        Ground-truth labels for each split.  Keys must match ``split_names``
        (default: ``['train', 'val', 'test']``).  Values are 1-D integer arrays.
    y_scores_runs : list of dict
        One entry per seed/run.  Each dict maps split name → predicted-probability
        array (shape ``(n_samples,)``).
    metrics_to_collect : list of str, optional
        Subset of metric names to store.  Defaults to all names in
        ``model.METRIC_NAMES``.
    split_names : list of str, optional
        Ordered list of split labels (default: ``['train', 'val', 'test']``).

    Returns
    -------
    xr.Dataset
        Dimensions: ``metric × split × run``.
        Data variables:
            ``metric_value`` — shape ``(n_metrics, n_splits, n_runs)``
            ``ci_low``       — 2.5th percentile across runs, shape ``(n_metrics, n_splits)``
            ``ci_high``      — 97.5th percentile across runs, shape ``(n_metrics, n_splits)``

    Examples
    --------
    >>> y_true = {'train': y_tr, 'val': y_va, 'test': y_te}
    >>> y_scores_runs = [predict_splits(m, x_tr, x_va, x_te) for m in models]
    >>> ds = collect_metrics_dataset(y_true, y_scores_runs)
    >>> ds['metric_value'].sel(metric='AUPRC', split='test').values
    """
    metrics_list = metrics_to_collect or METRIC_NAMES
    splits       = split_names or ['train', 'val', 'test']
    n_runs       = len(y_scores_runs)
    n_metrics    = len(metrics_list)
    n_splits     = len(splits)

    vals  = np.full((n_metrics, n_splits, n_runs), np.nan, dtype=np.float32)

    for r_idx, y_scores in enumerate(y_scores_runs):
        for s_idx, split in enumerate(splits):
            mvals = compute_metrics(y_true[split], y_scores[split])
            for m_idx, m in enumerate(metrics_list):
                vals[m_idx, s_idx, r_idx] = mvals[m]

    # Confidence intervals across runs (only meaningful when n_runs > 1)
    ci_lo = np.nanpercentile(vals,  2.5, axis=2)   # (n_metrics, n_splits)
    ci_hi = np.nanpercentile(vals, 97.5, axis=2)

    ds = xr.Dataset(
        {
            "metric_value": (("metric", "split", "run"), vals),
            "ci_low":       (("metric", "split"), ci_lo),
            "ci_high":      (("metric", "split"), ci_hi),
        },
        coords={
            "metric": metrics_list,
            "split":  splits,
            "run":    np.arange(n_runs),
        },
        attrs={"description": "CNN classification metrics per run with 2.5/97.5 CIs across runs."},
    )
    return ds


# =============================================================================
# Model save / load helpers
# =============================================================================

def save_model(
    model: "tf.keras.Model",
    savepath: Path,
    split_idx: int,
    run_idx: int,
) -> None:
    """
    Save a trained model to disk in Keras HDF5 format.

    File name encodes the split index and run (seed) index so results from
    different configurations don't overwrite one another.

    Parameters
    ----------
    model : tf.keras.Model
        Trained Keras model to save.
    savepath : Path
        Directory to save into.  Created if it doesn't exist.
    split_idx : int
        TVT split index (0–8).
    run_idx : int
        Seed / run index (0-based).
    """
    savepath = Path(savepath)
    savepath.mkdir(parents=True, exist_ok=True)
    fpath = savepath / f"cnn_jja_split{split_idx}_run{run_idx}.h5"
    model.save(str(fpath))
    print(f"  Model saved → {fpath}")


def load_model(
    loadpath: Path,
    split_idx: int,
    run_idx: int,
) -> "tf.keras.Model":
    """
    Load a previously saved Keras model.

    Parameters
    ----------
    loadpath : Path
        Directory containing the model file.
    split_idx : int
        TVT split index.
    run_idx : int
        Seed / run index.

    Returns
    -------
    tf.keras.Model
    """
    if not _TF_AVAILABLE:
        raise ImportError("TensorFlow is required to load models.")
    fpath = Path(loadpath) / f"cnn_jja_split{split_idx}_run{run_idx}.h5"
    return tf.keras.models.load_model(str(fpath))


def save_metrics_dataset(ds: xr.Dataset, savepath: Path, split_idx: int) -> None:
    """
    Save an xarray metrics Dataset to NetCDF.

    Parameters
    ----------
    ds : xr.Dataset
        Output of ``collect_metrics_dataset``.
    savepath : Path
        Directory or full file path.  If a directory, the filename encodes
        ``split_idx``.
    split_idx : int
        TVT split index, used when ``savepath`` is a directory.
    """
    savepath = Path(savepath)
    if savepath.suffix != '.nc':
        savepath.mkdir(parents=True, exist_ok=True)
        savepath = savepath / f"cnn_jja_metrics_split{split_idx}.nc"
    else:
        savepath.parent.mkdir(parents=True, exist_ok=True)

    comp = {"zlib": True, "complevel": 4}
    enc  = {v: comp for v in ds.data_vars if ds[v].ndim > 0}
    ds.to_netcdf(savepath, format="NETCDF4", encoding=enc)
    print(f"  Metrics saved → {savepath}")
