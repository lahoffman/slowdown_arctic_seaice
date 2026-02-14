"""
Explainable AI utilities for attribution analysis (LRP, etc.).
"""

import numpy as np

# Try to import innvestigate (requires older TF version)
try:
    import innvestigate
    XAI_AVAILABLE = True
except ImportError:
    XAI_AVAILABLE = False
    print("WARNING: innvestigate not available. XAI functions will not work.")


def compute_lrp_analysis(model, data, method='lrp.z'):
    """
    Compute Layer-wise Relevance Propagation (LRP) attribution maps.

    Parameters
    ----------
    model : keras.Model
        Trained model
    data : ndarray
        Input data for analysis
    method : str, optional
        LRP method (default: 'lrp.z')
        Options: 'lrp.z', 'lrp.epsilon', 'lrp.alpha_beta'

    Returns
    -------
    ndarray
        Attribution maps with same spatial dimensions as input
    """
    if not XAI_AVAILABLE:
        raise ImportError(
            "innvestigate not available. Please install in ML environment:\n"
            "pip install innvestigate"
        )

    # Create analyzer
    analyzer = innvestigate.create_analyzer(method, model)

    # Compute attributions
    attributions = analyzer.analyze(data)

    return attributions


def compute_attribution_maps(model, data, target_class=None, methods=None):
    """
    Compute multiple attribution maps for comparison.

    Parameters
    ----------
    model : keras.Model
        Trained model
    data : ndarray
        Input data
    target_class : int, optional
        Target class for attribution (if None, uses predicted class)
    methods : list of str, optional
        List of attribution methods to compute

    Returns
    -------
    dict
        Dictionary mapping method names to attribution arrays
    """
    if not XAI_AVAILABLE:
        raise ImportError("innvestigate not available.")

    if methods is None:
        methods = ['lrp.z', 'lrp.epsilon', 'gradient', 'integrated_gradients']

    attributions = {}

    for method in methods:
        try:
            analyzer = innvestigate.create_analyzer(method, model)
            attr = analyzer.analyze(data)
            attributions[method] = attr
        except Exception as e:
            print(f"Warning: Failed to compute {method}: {e}")
            attributions[method] = None

    return attributions


def aggregate_attributions(attributions, method='mean', axis=None):
    """
    Aggregate attribution maps (e.g., across ensemble members or time).

    Parameters
    ----------
    attributions : ndarray
        Attribution maps
    method : str, optional
        Aggregation method: 'mean', 'median', 'sum', 'max', 'abs_mean'
    axis : int or tuple, optional
        Axis/axes along which to aggregate

    Returns
    -------
    ndarray
        Aggregated attributions
    """
    if method == 'mean':
        return np.nanmean(attributions, axis=axis)
    elif method == 'median':
        return np.nanmedian(attributions, axis=axis)
    elif method == 'sum':
        return np.nansum(attributions, axis=axis)
    elif method == 'max':
        return np.nanmax(attributions, axis=axis)
    elif method == 'abs_mean':
        return np.nanmean(np.abs(attributions), axis=axis)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def normalize_attributions(attributions, method='minmax'):
    """
    Normalize attribution maps for visualization.

    Parameters
    ----------
    attributions : ndarray
        Attribution maps
    method : str, optional
        Normalization method: 'minmax', 'absmax', 'percentile'

    Returns
    -------
    ndarray
        Normalized attributions
    """
    if method == 'minmax':
        vmin = np.nanmin(attributions)
        vmax = np.nanmax(attributions)
        return (attributions - vmin) / (vmax - vmin + 1e-8)

    elif method == 'absmax':
        vmax = np.nanmax(np.abs(attributions))
        return attributions / (vmax + 1e-8)

    elif method == 'percentile':
        vmin = np.nanpercentile(attributions, 2)
        vmax = np.nanpercentile(attributions, 98)
        return np.clip((attributions - vmin) / (vmax - vmin + 1e-8), 0, 1)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def spatial_average_attributions(attributions, lat, lon, region=None):
    """
    Compute spatial average of attributions over a region.

    Parameters
    ----------
    attributions : ndarray
        Attribution maps with shape (..., lat, lon)
    lat : ndarray
        Latitude values
    lon : ndarray
        Longitude values
    region : dict, optional
        Region bounds: {'lat_min', 'lat_max', 'lon_min', 'lon_max'}

    Returns
    -------
    float or ndarray
        Spatially averaged attributions
    """
    if region is not None:
        # Subset region
        lat_mask = (lat >= region['lat_min']) & (lat <= region['lat_max'])
        lon_mask = (lon >= region['lon_min']) & (lon <= region['lon_max'])

        # Apply masks
        attr_subset = attributions[..., lat_mask, :]
        attr_subset = attr_subset[..., lon_mask]
    else:
        attr_subset = attributions

    # Compute area-weighted mean if lat provided
    if lat is not None:
        weights = np.cos(np.deg2rad(lat))
        weights = weights / np.sum(weights)

        # Broadcast weights
        if region is not None:
            weights = weights[lat_mask]

        w2d = weights[:, np.newaxis]
        weighted = attr_subset * w2d
        return np.nanmean(weighted)
    else:
        return np.nanmean(attr_subset)
