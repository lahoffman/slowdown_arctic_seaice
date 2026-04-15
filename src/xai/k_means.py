"""
src/xai/k_means.py — Spatial k-means clustering of LRP composite maps.

This module applies k-means **in image space**: clustering is performed across
the grid cells (pixels) of a single ensemble-mean LRP composite map, not across
samples/events.  The goal is to segment the composite map into spatial
"regions of similar relevance", analogous to a coarse land-use / ROI
segmentation.

Workflow
--------
1. ``normalize_map``           — normalise the composite (max-abs, consistent
                                 with the project's LRP plotting convention).
2. ``prepare_spatial_features`` — flatten a 2-D map into (n_pixels, n_features)
                                 and return a mask of valid (non-NaN) cells.
3. ``run_kmeans_spatial``      — fit ``sklearn.cluster.KMeans`` on the pixel
                                 features.
4. ``reshape_clusters``        — scatter the 1-D labels back onto the 2-D grid.
5. ``compute_cluster_means``   — mean LRP value inside each cluster.
6. ``threshold_frequency_map`` — frequency with which |LRP|/LRP exceeds a
                                 threshold across a per-model stack.
7. ``sort_clusters_by_mean``   — optional label renumbering for stable figure
                                 ordering.

Authors: Lauren Hoffman  <lhoffma2@ucsc.edu>
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


# =============================================================================
# 1. Normalisation
# =============================================================================

def normalize_map(lrp_map: np.ndarray, method: str = "maxabs") -> np.ndarray:
    """
    Normalise a 2-D relevance map.

    Parameters
    ----------
    lrp_map : np.ndarray, shape (nx, ny)
        Composite relevance map.  May contain NaNs (land cells).
    method : {'maxabs', 'max'}
        - 'maxabs': divide by ``max(|lrp|)`` — works for signed and positive
          maps, preserves sign, and is consistent with the project's existing
          LRP-plotting convention.
        - 'max'   : divide by ``max(lrp)``, useful for positive-only maps.

    Returns
    -------
    np.ndarray, same shape as input.
    """
    if method == "maxabs":
        denom = np.nanmax(np.abs(lrp_map))
    elif method == "max":
        denom = np.nanmax(lrp_map)
    else:
        raise ValueError(f"unknown method: {method!r}")

    if denom is None or not np.isfinite(denom) or denom == 0:
        return lrp_map.copy()
    return lrp_map / denom


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
    Flatten a 2-D LRP map into pixel-wise features for clustering.

    Parameters
    ----------
    lrp_map : np.ndarray, shape (nx, ny)
        The composite map to be clustered.  NaNs are treated as invalid.
    lat, lon : np.ndarray, optional
        1-D latitude (nx,) / longitude (ny,) arrays.  Required if
        ``use_coords=True``.
    use_coords : bool
        If True, append latitude and (cos-scaled) longitude to the feature
        vector so clusters may be contiguous in space.  Default False
        (cluster on LRP value alone).

    Returns
    -------
    features : np.ndarray, shape (n_valid, n_features)
        Feature matrix for valid (non-NaN) pixels only.
    valid_mask : np.ndarray of bool, shape (nx, ny)
        True where the pixel was kept; False where it was NaN.
    """
    valid_mask = np.isfinite(lrp_map)
    vals = lrp_map[valid_mask].reshape(-1, 1).astype(float)

    if not use_coords:
        return vals, valid_mask

    if lat is None or lon is None:
        raise ValueError("lat and lon must be provided when use_coords=True")

    lon2d, lat2d = np.meshgrid(lon, lat)
    # cos-scale longitude to avoid wrap-around distortion at high latitudes
    lat_rad = np.deg2rad(lat2d)
    x = np.cos(np.deg2rad(lon2d)) * np.cos(lat_rad)
    y = np.sin(np.deg2rad(lon2d)) * np.cos(lat_rad)

    coords = np.stack([lat2d[valid_mask], x[valid_mask], y[valid_mask]], axis=1)
    features = np.concatenate([vals, coords], axis=1)
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
        Output of ``prepare_spatial_features``.
    n_clusters : int
        Number of clusters.  Default 2.
    random_state : int
        Fixed for reproducibility.
    n_init : int
        Number of KMeans initialisations.

    Returns
    -------
    labels : np.ndarray of int, shape (n_pixels,)
        Cluster label for each valid pixel.
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

    Invalid pixels are filled with ``fill`` (default NaN).

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
        Integer cluster labels, NaN outside the valid region.

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

    This makes figure ordering deterministic across runs / seeds.

    Parameters
    ----------
    cluster_map : np.ndarray, shape (nx, ny)
    lrp_map : np.ndarray, shape (nx, ny)
    descending : bool
        True ⇒ cluster 0 = highest-mean-LRP region.

    Returns
    -------
    np.ndarray, same shape as input, with labels renumbered.
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
# 5. Threshold-frequency map
# =============================================================================

def threshold_frequency_map(
    lrp_stack: np.ndarray,
    threshold: float = 0.3,
    mode: str = "absolute",
    per_sample_norm: bool = True,
) -> np.ndarray:
    """
    Frequency with which normalised LRP exceeds a threshold, computed across
    an axis-0 stack of maps (e.g. the 45 per-model composite maps).

    Parameters
    ----------
    lrp_stack : np.ndarray, shape (n_maps, nx, ny)
        Stack of LRP maps (e.g. one per model).
    threshold : float
        Threshold in normalised units (default 0.3).
    mode : {'absolute', 'positive', 'negative', 'signed'}
        - 'absolute': fraction of maps with ``|LRP_norm| > threshold``.
          Default; recommended for the signed-relevance case because it gives
          a single, interpretable panel alongside the composite.
        - 'positive': fraction of maps with ``LRP_norm > +threshold``.
        - 'negative': fraction of maps with ``LRP_norm < -threshold``.
        - 'signed'  : positive_frequency − negative_frequency
                      (dominance; ranges in [-1, 1]).
    per_sample_norm : bool
        If True (default) each map is normalised by its own max(|LRP|) before
        thresholding.  If False, the raw stack values are used.

    Returns
    -------
    np.ndarray, shape (nx, ny)
        Exceedance frequency in [0, 1] (or [-1, 1] for ``mode='signed'``).
    """
    stack = lrp_stack.astype(float).copy()

    if per_sample_norm:
        # per-map max-abs normalisation — robust to NaNs
        denom = np.nanmax(np.abs(stack.reshape(stack.shape[0], -1)), axis=1)
        denom = np.where(denom > 0, denom, 1.0)
        stack = stack / denom[:, None, None]

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
