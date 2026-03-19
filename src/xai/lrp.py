"""
src/xai/lrp.py — Layer-wise Relevance Propagation (LRP-z) attributions.

This module wraps the iNNvestigate library to compute LRP-z attributions for
the trained CNN models.  LRP-z requires eager execution to be disabled, so
``tf.compat.v1.disable_eager_execution()`` is called at import time.

Pipeline
--------
1. ``strip_sigmoid``   — Remove the sigmoid activation from the output Dense
                         layer so that LRP sees logits instead of probabilities.
                         LRP-z is theoretically sound only for layers without
                         saturating activations on the output.
2. ``compute_lrp_z``   — Create an iNNvestigate LRP-z analyser and run it over
                         the training data in chunks to avoid memory overflow.
3. ``save_lrp``        — Persist the attribution maps to NetCDF with lat/lon
                         coordinates.
4. ``load_lrp``        — Reload saved attributions for downstream analysis.

Input convention
----------------
Before calling ``compute_lrp_z``, set the land-fill sentinel (``-10``) to 0 in
the input data.  Land pixels were set to ``-10`` after standardisation so that
the model learns to ignore them, but LRP interprets large-magnitude inputs as
important.  Zeroing them prevents spurious attributions to the land mask.

Authors: Lauren Hoffman  <lhoffma2@ucsc.edu>
"""

import numpy as np
import xarray as xr
from pathlib import Path
from typing import Optional

# iNNvestigate requires TF v1 graph mode
try:
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    from tensorflow.keras.models import Model
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

try:
    import innvestigate
    _INNV_AVAILABLE = True
except ImportError:
    _INNV_AVAILABLE = False


# =============================================================================
# 1. Model preparation
# =============================================================================

def strip_sigmoid(model: "tf.keras.Model") -> "tf.keras.Model":
    """
    Return a copy of the model with the sigmoid removed from the output layer.

    LRP-z propagates relevance backwards through the network.  Sigmoid
    saturation near 0 or 1 can collapse gradients and distort attributions,
    so the standard practice is to replace the output sigmoid with a linear
    (identity) activation before running LRP.

    The function:
      1. Reads the configuration of the last Dense layer.
      2. Rebuilds an identical Dense layer with ``activation=None``.
      3. Connects it to the second-to-last layer's output.
      4. Copies weights from the original output layer into the new one.

    Parameters
    ----------
    model : tf.keras.Model
        Trained Keras model whose last layer is a ``Dense`` unit with
        ``activation='sigmoid'``.

    Returns
    -------
    tf.keras.Model
        New model with the same architecture and weights except that the
        output activation is ``None`` (linear / logits).

    Examples
    --------
    >>> model_logits = strip_sigmoid(model)
    >>> analyzer = innvestigate.create_analyzer('lrp.z', model_logits)
    """
    if not _TF_AVAILABLE:
        raise ImportError("TensorFlow is required for strip_sigmoid.")

    last   = model.layers[-1]
    config = last.get_config()
    config['activation'] = None              # remove sigmoid

    # Rebuild Dense layer without sigmoid
    logits_layer = type(last).from_config(config)

    # Connect to the penultimate layer's output
    x = model.layers[-2].output
    y = logits_layer(x)

    model_logits = Model(inputs=model.input, outputs=y)

    # Transfer trained weights
    logits_layer.set_weights(last.get_weights())

    return model_logits


# =============================================================================
# 2. LRP-z attribution
# =============================================================================

def compute_lrp_z(
    model_logits: "tf.keras.Model",
    x_data: np.ndarray,
    chunk_size: int = 100,
    lrp_method: str = 'lrp.z',
    land_fill_value: float = -10.0,
    replace_fill_with: float = 0.0,
) -> np.ndarray:
    """
    Compute LRP-z attribution maps for all samples in ``x_data``.

    Attributions are computed in chunks to avoid running out of memory for
    large training sets.  Before analysis, land-fill sentinel values are
    replaced with zero to prevent large-magnitude land pixels from receiving
    spurious relevance.

    Parameters
    ----------
    model_logits : tf.keras.Model
        Model with sigmoid stripped (output of ``strip_sigmoid``).
    x_data : np.ndarray
        Input data, shape ``(n_samples, nx, ny, nch)``.  Land pixels should
        still be at ``land_fill_value`` (they will be zeroed internally).
    chunk_size : int
        Number of samples to analyse per batch (default: 100).
        Reduce this if you run into memory errors.
    lrp_method : str
        iNNvestigate analyser name (default: ``'lrp.z'``).
    land_fill_value : float
        Sentinel value used for land pixels (default: ``-10.0``).
    replace_fill_with : float
        Value to substitute in place of ``land_fill_value`` before LRP
        (default: ``0.0``).

    Returns
    -------
    attributions : np.ndarray
        LRP relevance maps, shape ``(n_chunks, chunk_size, nx, ny, nch)``.
        Note: the first two dimensions are *not* flattened to preserve
        the chunk structure for later reshaping or inspection.

        To get a flat array of all samples use:
            ``attributions.reshape(-1, nx, ny, nch)``

    Raises
    ------
    ImportError
        If iNNvestigate or TensorFlow are not installed.

    Examples
    --------
    >>> model_logits = strip_sigmoid(model)
    >>> x_train_lrp = x_train.copy()
    >>> x_train_lrp[np.isclose(x_train_lrp, -10.0)] = 0.0
    >>> attr = compute_lrp_z(model_logits, x_train_lrp)
    >>> attr.shape   # (n_chunks, 100, 192, 288, 1)
    """
    if not _TF_AVAILABLE:
        raise ImportError("TensorFlow is required for LRP computation.")
    if not _INNV_AVAILABLE:
        raise ImportError(
            "iNNvestigate is required for LRP computation.  "
            "Install it with:  pip install innvestigate"
        )

    # Replace land fill with zero before analysis
    x_lrp = x_data.copy()
    x_lrp[np.isclose(x_lrp, land_fill_value)] = replace_fill_with

    # Create analyser once
    analyzer = innvestigate.create_analyzer(lrp_method, model_logits)

    n_samples  = x_lrp.shape[0]
    n_chunks   = n_samples // chunk_size
    remainder  = n_samples % chunk_size

    if remainder != 0:
        print(
            f"  Warning: {n_samples} samples is not evenly divisible by "
            f"chunk_size={chunk_size}.  "
            f"The last {remainder} sample(s) will be dropped."
        )

    chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end   = start + chunk_size
        chunk = analyzer.analyze(x_lrp[start:end])
        chunks.append(chunk)
        if (i + 1) % 10 == 0 or i == n_chunks - 1:
            print(f"  LRP chunk {i + 1}/{n_chunks} done")

    attributions = np.array(chunks)   # (n_chunks, chunk_size, nx, ny, nch)
    return attributions


# =============================================================================
# 3. Save / load attributions
# =============================================================================

def save_lrp(
    attributions: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    savepath: Path,
    split_idx: Optional[int] = None,
    run_idx: Optional[int] = None,
    attrs: Optional[dict] = None,
) -> None:
    """
    Save LRP attribution maps to a NetCDF file with lat/lon coordinates.

    The attribution array retains the chunk structure ``(nch, nt, nx, ny, nc)``
    used in the original M2 script to preserve compatibility with downstream
    figure-generation code.

    Parameters
    ----------
    attributions : np.ndarray
        LRP relevance maps from ``compute_lrp_z``, shape
        ``(n_chunks, chunk_size, nx, ny, nch)``.
    lat : np.ndarray
        Latitude values, shape ``(nx,)``.
    lon : np.ndarray
        Longitude values, shape ``(ny,)``.
    savepath : Path or str
        Output NetCDF file path.
    split_idx : int, optional
        TVT split index stored as a file attribute.
    run_idx : int, optional
        Seed / run index stored as a file attribute.
    attrs : dict, optional
        Additional global attributes.

    Examples
    --------
    >>> save_lrp(attributions, lat, lon,
    ...          savepath=paths.ATTRIBUTIONS_DIR / 'lrp_split0_run0.nc',
    ...          split_idx=0, run_idx=0)
    """
    savepath = Path(savepath)
    savepath.parent.mkdir(parents=True, exist_ok=True)

    # Keep the original dim convention from M2: (nch, nt, nx, ny, nc)
    # attributions shape: (n_chunks, chunk_size, nx, ny, nch)
    # → transpose to (nch, nt, nx, ny, nc) where nch=n_chunks, nt=chunk_size, nc=nch_input
    # For the 1-channel JJA case nc=1 and nch_input=1, so the transposition is trivial.
    # We store as-is and document the dimension order.
    nch_out, nt_out, nx_out, ny_out, nc_out = attributions.shape

    ds = xr.Dataset(
        {
            "lrp_attributions": (
                ("nch", "nt", "nx", "ny", "nc"),
                attributions.astype(np.float32),
            ),
            "lat": (("nx",), lat.astype(np.float32)),
            "lon": (("ny",), lon.astype(np.float32)),
        },
        coords={
            "nch": np.arange(nch_out),
            "nt":  np.arange(nt_out),
            "nx":  np.arange(nx_out),
            "ny":  np.arange(ny_out),
            "nc":  np.arange(nc_out),
        },
    )

    ds["lrp_attributions"].attrs["long_name"] = (
        "LRP-z relevance scores for JJA SST → September slowdown prediction"
    )
    ds["lat"].attrs["units"] = "degrees_north"
    ds["lon"].attrs["units"] = "degrees_east"

    ds.attrs["description"] = (
        "LRP-z attribution maps computed by iNNvestigate on the training-set "
        "inputs of the JJA SST CNN."
    )
    ds.attrs["lrp_method"] = "lrp.z"
    if split_idx is not None:
        ds.attrs["split_idx"] = split_idx
    if run_idx is not None:
        ds.attrs["run_idx"] = run_idx
    if attrs:
        ds.attrs.update(attrs)

    encoding = {v: {"zlib": True, "complevel": 4}
                for v in ds.data_vars if ds[v].ndim > 0}
    ds.to_netcdf(savepath, format="NETCDF4", encoding=encoding)
    print(f"  LRP attributions saved → {savepath}")


def load_lrp(filepath: Path) -> dict:
    """
    Load LRP attribution maps from a NetCDF file saved by ``save_lrp``.

    Parameters
    ----------
    filepath : Path or str
        Path to a NetCDF file written by ``save_lrp``.

    Returns
    -------
    dict with keys:
        ``attributions`` : np.ndarray   shape ``(nch, nt, nx, ny, nc)``
        ``lat``          : np.ndarray   shape ``(nx,)``
        ``lon``          : np.ndarray   shape ``(ny,)``
        ``split_idx``    : int or None
        ``run_idx``      : int or None

    Examples
    --------
    >>> result = load_lrp(paths.ATTRIBUTIONS_DIR / 'lrp_split0_run0.nc')
    >>> result['attributions'].shape   # (n_chunks, chunk_size, nx, ny, 1)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"LRP file not found: {filepath}")

    with xr.open_dataset(filepath) as ds:
        out = {
            "attributions": ds["lrp_attributions"].values,
            "lat":          ds["lat"].values,
            "lon":          ds["lon"].values,
            "split_idx":    int(ds.attrs["split_idx"]) if "split_idx" in ds.attrs else None,
            "run_idx":      int(ds.attrs["run_idx"])   if "run_idx"   in ds.attrs else None,
        }
    return out
