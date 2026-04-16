"""
src/xai/k_means.py — Spatial k-means clustering of LRP maps.

This module applies k-means **in image space** using the full ensemble of
per-model LRP maps (typically 45 maps from 9 splits × 5 seeds).

Clustering formulation
----------------------
For each grid cell (i, j), construct a feature vector of length ``n_models``
containing the relevance at that pixel from each model.  K-means then clusters
**pixels** according to how their relevance behaves across the model ensemble,
preserving the inter-model variability that a single ensemble-mean map
discards.

    samples to k-means  = valid ocean pixels
    features            = the n_models relevance values at each pixel

Normalisation
-------------
Each per-model map is normalised by its **97th percentile** (of |LRP| for
signed, or of LRP for positive-only) before any thresholding or clustering.
This maps signed relevance to approximately [-1, 1] and positive-only to
[0, 1], preventing outlier-driven compression and ensuring thresholds have
a consistent meaning across models and pixels.

Workflow
--------
1. ``normalize_map``              — normalise a single 2-D map (97th pctile).
2. ``normalize_stack``            — normalise every map in a (n_models, …)
                                    stack independently.
3. ``prepare_ensemble_features``  — build (n_pixels, n_models) feature matrix
                                    from a normalised stack.
4. ``run_kmeans_spatial``         — fit sklearn KMeans on pixel features.
5. ``reshape_clusters``           — scatter 1-D labels back to 2-D grid.
6. ``compute_cluster_means``      — per-cluster statistics on a 2-D map.
7. ``sort_clusters_by_mean``      — relabel for stable figure ordering.
8. ``cluster_ensemble``           — convenience wrapper: normalise → features
                                    → cluster → reshape → sort.
9. ``threshold_frequency_map``    — exceedance frequency across per-model maps
                                    (applies 97th-pctile normalisation).

Authors: Lauren Hoffman  <lhoffma2@ucsc.edu>
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


# =============================================================================
# 1. Normalisation
# =============================================================================

def normalize_map(
    lrp_map: np.ndarray,
    method: str = "maxabs",
    percentile: float = 97.0,
) -> np.ndarray:
    """
    Normalise a 2-D relevance map using percentile-based scaling.

    The chosen percentile maps to ±1 (signed) or 1 (positive-only), then
    values are clipped.  Using a percentile (default 97th) instead of the
    absolute maximum prevents extreme outliers from compressing the colour
    range.

    Parameters
    ----------
    lrp_map : np.ndarray, shape (nx, ny)
        Composite relevance map.  May contain NaNs (land cells).
    method : {'maxabs', 'max'}
        - 'maxabs': normalise by the ``percentile``-th percentile of
          ``|lrp|``.  Preserves sign; clips to ``[-1, 1]``.
        - 'max'   : normalise by the ``percentile``-th percentile of the
          raw (positive) values; clips to ``[0, 1]``.
    percentile : float
        Percentile for the scaling denominator (default 97).

    Returns
    -------
    np.ndarray, same shape as input.
    """
    if method == "maxabs":
        denom = float(np.nanpercentile(np.abs(lrp_map), percentile))
    elif method == "max":
        denom = float(np.nanpercentile(lrp_map, percentile))
    else:
        raise ValueError(f"unknown method: {method!r}")

    if not np.isfinite(denom) or denom == 0:
        return lrp_map.copy()

    normed = lrp_map / denom

    if method == "maxabs":
        normed = np.clip(normed, -1.0, 1.0)
    else:
        normed = np.clip(normed, 0.0, 1.0)

    return normed


def normalize_stack(
    lrp_stack: np.ndarray,
    method: str = "maxabs",
    percentile: float = 97.0,
) -> np.ndarray:
    """
    Normalise each map in a stack independently using ``normalize_map``.

    Parameters
    ----------
    lrp_stack : np.ndarray, shape (n_models, nx, ny)
    method, percentile : passed to ``normalize_map``.

    Returns
    -------
    np.ndarray, same shape as input.  Each slice along axis 0 is
    independently normalised.
    """
    out = np.empty_like(lrp_stack, dtype=float)
    for i in range(lrp_stack.shape[0]):
        out[i] = normalize_map(lrp_stack[i], method=method,
                               percentile=percentile)
    return out


# =============================================================================
# 2. Feature preparation
# =============================================================================

def prepare_spatial_features(
    lrp_map: np.ndarray,
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
    use_coords: bool = False,
):
    """
    Flatten a single 2-D LRP map into pixel-wise features for clustering.

    .. note:: For ensemble-based clustering (the recommended workflow), use
       ``prepare_ensemble_features`` instead.

    Parameters
    ----------
    lrp_map : np.ndarray, shape (nx, ny)
    lat, lon : optional 1-D arrays.
    use_coords : bool

    Returns
    -------
    features : np.ndarray, shape (n_valid, n_features)
    valid_mask : np.ndarray of bool, shape (nx, ny)
    """
    valid_mask = np.isfinite(lrp_map)
    vals = lrp_map[valid_mask].reshape(-1, 1).astype(float)

    if not use_coords:
        return vals, valid_mask

    if lat is None or lon is None:
        raise ValueError("lat and lon must be provided when use_coords=True")

    lon2d, lat2d = np.meshgrid(lon, lat)
    lat_rad = np.deg2rad(lat2d)
    x = np.cos(np.deg2rad(lon2d)) * np.cos(lat_rad)
    y = np.sin(np.deg2rad(lon2d)) * np.cos(lat_rad)

    coords = np.stack([lat2d[valid_mask], x[valid_mask], y[valid_mask]], axis=1)
    features = np.concatenate([vals, coords], axis=1)
    return features, valid_mask


def prepare_ensemble_features(
    lrp_stack_norm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the k-means feature matrix from a **normalised** stack of per-model
    LRP maps.

    For each grid cell (i, j), the feature vector is the vector of relevance
    values across all models:

        feature[pixel] = [lrp_model_0[i,j], lrp_model_1[i,j], …]

    A pixel is valid (included) only if it is finite in **every** model.

    Parameters
    ----------
    lrp_stack_norm : np.ndarray, shape (n_models, nx, ny)
        Already-normalised stack (use ``normalize_stack`` first).

    Returns
    -------
    features : np.ndarray, shape (n_valid_pixels, n_models)
        Feature matrix ready for ``run_kmeans_spatial``.
    valid_mask : np.ndarray of bool, shape (nx, ny)
        True where the pixel is valid (finite in all models).
    """
    # valid = finite in every model
    valid_mask = np.all(np.isfinite(lrp_stack_norm), axis=0)   # (nx, ny)
    n_models = lrp_stack_norm.shape[0]

    # (n_models, n_valid) → transpose → (n_valid, n_models)
    features = lrp_stack_norm[:, valid_mask].T.astype(float)    # (n_valid, n_models)
    return features, valid_mask


# =============================================================================
# 3. Clustering
# =============================================================================

def run_kmeans_spatial(
    features: np.ndarray,
    n_clusters: int = 2,
    random_state: int = 0,
    n_init: int = 10,
) -> np.ndarray:
    """
    Fit sklearn KMeans on pixel features.

    Parameters
    ----------
    features : np.ndarray, shape (n_pixels, n_features)
        Output of ``prepare_ensemble_features`` or ``prepare_spatial_features``.
    n_clusters : int
    random_state : int
    n_init : int

    Returns
    -------
    labels : np.ndarray of int, shape (n_pixels,)
    """
    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )
    km.fit(features)
    return km.labels_


def reshape_clusters(
    labels: np.ndarray,
    valid_mask: np.ndarray,
    fill: float = np.nan,
) -> np.ndarray:
    """
    Map 1-D cluster labels back onto the 2-D grid.

    Parameters
    ----------
    labels : np.ndarray, shape (n_valid,)
    valid_mask : np.ndarray of bool, shape (nx, ny)
    fill : float

    Returns
    -------
    np.ndarray, shape (nx, ny)
    """
    out = np.full(valid_mask.shape, fill, dtype=float)
    out[valid_mask] = labels
    return out


# =============================================================================
# 4. Cluster statistics
# =============================================================================

def compute_cluster_means(
    lrp_map: np.ndarray,
    cluster_map: np.ndarray,
) -> dict:
    """
    Mean, std, and pixel count of LRP values inside each cluster.

    Parameters
    ----------
    lrp_map : np.ndarray, shape (nx, ny)
    cluster_map : np.ndarray, shape (nx, ny)

    Returns
    -------
    dict : {cluster_id: {'mean': ..., 'std': ..., 'n': ...}}
    """
    stats = {}
    valid = np.isfinite(cluster_map)
    ids = np.unique(cluster_map[valid]).astype(int)
    for cid in ids:
        sel = (cluster_map == cid) & np.isfinite(lrp_map)
        vals = lrp_map[sel]
        stats[int(cid)] = {
            "mean": float(vals.mean()) if vals.size else np.nan,
            "std":  float(vals.std())  if vals.size else np.nan,
            "n":    int(vals.size),
        }
    return stats


def sort_clusters_by_mean(
    cluster_map: np.ndarray,
    lrp_map: np.ndarray,
    descending: bool = True,
) -> np.ndarray:
    """
    Relabel clusters so that cluster 0 has the largest (or smallest) mean LRP.

    Parameters
    ----------
    cluster_map : np.ndarray, shape (nx, ny)
    lrp_map : np.ndarray, shape (nx, ny)
    descending : bool

    Returns
    -------
    np.ndarray, relabelled cluster map.
    """
    stats = compute_cluster_means(lrp_map, cluster_map)
    ids = list(stats.keys())
    ids_sorted = sorted(ids, key=lambda c: stats[c]["mean"],
                        reverse=descending)
    remap = {old: new for new, old in enumerate(ids_sorted)}

    out = np.full_like(cluster_map, np.nan, dtype=float)
    for old, new in remap.items():
        out[cluster_map == old] = new
    return out


# =============================================================================
# 5. Convenience: full ensemble-clustering pipeline
# =============================================================================

def cluster_ensemble(
    lrp_stack: np.ndarray,
    n_clusters: int = 2,
    method: str = "maxabs",
    percentile: float = 97.0,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    End-to-end ensemble clustering: normalise → features → k-means → reshape.

    Clustering is performed across **pixels**, using the vector of per-model
    relevance values at each pixel as the feature vector.  This preserves
    inter-model variability that would be lost by clustering a single mean map.

    Parameters
    ----------
    lrp_stack : np.ndarray, shape (n_models, nx, ny)
        Raw (un-normalised) per-model LRP maps.
    n_clusters : int
    method : str
        Normalisation method passed to ``normalize_stack`` / ``normalize_map``.
    percentile : float
        Percentile for normalisation (default 97).
    random_state : int

    Returns
    -------
    stack_norm : np.ndarray, shape (n_models, nx, ny)
        Normalised per-model maps.
    composite_norm : np.ndarray, shape (nx, ny)
        Ensemble-mean of the normalised maps (for display).
    cluster_map : np.ndarray, shape (nx, ny)
        Integer cluster labels, sorted by descending mean LRP.
        NaN outside valid region.
    """
    # 1. Normalise each model map independently
    stack_norm = normalize_stack(lrp_stack, method=method,
                                 percentile=percentile)

    # 2. Build feature matrix: (n_valid_pixels, n_models)
    features, valid_mask = prepare_ensemble_features(stack_norm)

    # 3. Run k-means
    labels = run_kmeans_spatial(features, n_clusters=n_clusters,
                                random_state=random_state)

    # 4. Reshape to spatial grid
    cluster_map = reshape_clusters(labels, valid_mask)

    # 5. Ensemble-mean composite (for display / sorting)
    composite_norm = np.nanmean(stack_norm, axis=0)

    # 6. Sort clusters by descending mean LRP in the composite
    cluster_map = sort_clusters_by_mean(cluster_map, composite_norm,
                                         descending=True)

    return stack_norm, composite_norm, cluster_map


# =============================================================================
# 6. Threshold-frequency map
# =============================================================================

def threshold_frequency_map(
    lrp_stack: np.ndarray,
    threshold: float = 0.3,
    mode: str = "absolute",
    normalize: bool = True,
    norm_method: str = "maxabs",
    norm_percentile: float = 97.0,
) -> np.ndarray:
    """
    Frequency with which normalised LRP exceeds a threshold, computed across
    an axis-0 stack of maps (e.g. the 45 per-model composite maps).

    .. important:: Normalisation uses the 97th percentile (via
       ``normalize_stack``), matching the project standard.  This ensures
       the threshold (e.g. 0.3) is applied to values on a consistent,
       non-compressed scale.

    Parameters
    ----------
    lrp_stack : np.ndarray, shape (n_maps, nx, ny)
        Stack of per-model LRP maps.
    threshold : float
        Threshold in normalised units (default 0.3).
    mode : {'absolute', 'positive', 'negative', 'signed'}
        - 'absolute': fraction of maps with ``|LRP_norm| > threshold``.
        - 'positive': fraction with ``LRP_norm > +threshold``.
        - 'negative': fraction with ``LRP_norm < -threshold``.
        - 'signed'  : positive_freq − negative_freq.
    normalize : bool
        If True (default) each map is normalised by its 97th percentile
        before thresholding.  If False, the raw stack is used.
    norm_method : str
        Normalisation method ('maxabs' or 'max').  Default 'maxabs'.
    norm_percentile : float
        Percentile for normalisation (default 97).

    Returns
    -------
    np.ndarray, shape (nx, ny)
        Exceedance frequency in [0, 1] (or [-1, 1] for mode='signed').
    """
    stack = lrp_stack.astype(float).copy()

    if normalize:
        # Per-model 97th-percentile normalisation (consistent with clustering)
        stack = normalize_stack(stack, method=norm_method,
                                percentile=norm_percentile)

    if mode == "absolute":
        exceed = np.abs(stack) > threshold
        return np.nanmean(exceed.astype(float), axis=0)
    if mode == "positive":
        exceed = stack > threshold
        return np.nanmean(exceed.astype(float), axis=0)
    if mode == "negative":
        exceed = stack < -threshold
        return np.nanmean(exceed.astype(float), axis=0)
    if mode == "signed":
        pos = np.nanmean((stack >  threshold).astype(float), axis=0)
        neg = np.nanmean((stack < -threshold).astype(float), axis=0)
        return pos - neg
    raise ValueError(f"unknown mode: {mode!r}")
