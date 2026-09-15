"""
methods.py — several attribution methods on one trained CNN (step 5.3b / AIES §6).

Wraps iNNvestigate analyzers (gradient, input×gradient, integrated gradients,
SmoothGrad, DeepTaylor, LRP variants) and, when installed, SHAP DeepExplainer,
behind one call. All return map relevance of shape (n, nx, ny). Composites,
inter-method agreement and region shares are computed here so the paper can put
"where each method looks" next to "what occlusion says the model uses".
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from src.analysis.occlusion import REGIONS, region_mask

# name → iNNvestigate analyzer (None = handled separately)
METHODS: Dict[str, Optional[str]] = {
    "lrp_z":          "lrp.z",
    "lrp_epsilon":    "lrp.epsilon",
    "lrp_a2b1":       "lrp.alpha_2_beta_1",
    "deep_taylor":    "deep_taylor",
    "integrated_gradients": "integrated_gradients",
    "smoothgrad":     "smoothgrad",
    "input_x_gradient": "input_t_gradient",
    "gradient":       "gradient",
    "shap_deep":      None,
}
# IG and SmoothGrad are 16–32× the cost of the others and hung for >1 h on the two-input model; opt in with --methods
DEFAULT_METHODS = ["lrp_z", "lrp_a2b1", "deep_taylor", "input_x_gradient", "shap_deep"]


def _zero_land(x, land_fill=-10.0):
    multi = isinstance(x, (list, tuple))
    maps = (x[0] if multi else x).copy(); maps[np.isclose(maps, land_fill)] = 0.0
    return [maps, *x[1:]] if multi else maps


def attribute(model_logits, x, method: str, chunk: int = 100, background=None, **kw) -> np.ndarray:
    """Map relevance (n, nx, ny) of ``x`` for one method; ``background`` (a few hundred samples) is used by SHAP."""
    x = _zero_land(x)
    multi = isinstance(x, (list, tuple))
    n = (x[0] if multi else x).shape[0]
    if method == "shap_deep":
        import shap
        bg = _zero_land(background)
        expl = shap.DeepExplainer(model_logits, bg if multi else bg)
        out = []
        for i in range(0, n, chunk):
            xi = [a[i:i + chunk] for a in x] if multi else x[i:i + chunk]
            sv = expl.shap_values(xi)
            sv = sv[0] if isinstance(sv, list) else sv                 # first output / first input
            if isinstance(sv, list):
                sv = sv[0]
            out.append(np.asarray(sv)[..., 0] if np.asarray(sv).ndim == 4 else np.asarray(sv))
        return np.concatenate(out)
    import innvestigate
    name = METHODS[method]
    opts = {"integrated_gradients": dict(steps=32), "smoothgrad": dict(augment_by_n=16, noise_scale=0.2)}.get(method, {})
    opts.update(kw)
    an = innvestigate.create_analyzer(name, model_logits, **opts)
    out = []
    for i in range(0, n, chunk):
        xi = [a[i:i + chunk] for a in x] if multi else x[i:i + chunk]
        r = an.analyze(xi)
        r = r[0] if isinstance(r, list) else r
        out.append(np.asarray(r)[..., 0])
    return np.concatenate(out)


def composite(rel: np.ndarray, sel: np.ndarray, normalise: bool = True, q: float = 99.0) -> np.ndarray:
    """Mean relevance over the selected samples; if ``normalise``, scaled so the q-th percentile of |value| = 1 and clipped to ±1."""
    m = np.nanmean(rel[sel], axis=0)
    if not normalise:
        return m
    s = np.nanpercentile(np.abs(m[m != 0]), q) if np.any(m != 0) else 0.0
    return np.clip(m / s, -1, 1) if s > 0 else m


def spatial_correlation(maps: Dict[str, np.ndarray], ocean: np.ndarray) -> np.ndarray:
    """Pearson correlation between method composites over ocean cells → (n_methods, n_methods)."""
    keys = list(maps); v = np.array([maps[k][ocean] for k in keys])
    v = (v - v.mean(1, keepdims=True)) / v.std(1, keepdims=True)
    return v @ v.T / v.shape[1]


def region_share(rel_map: np.ndarray, lat, lon, regions: Sequence[str] = ("arctic", "north_pacific", "tropical_pacific", "north_atlantic"),
                 ocean: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Fraction of total |relevance| inside each region (ocean cells only)."""
    a = np.abs(np.where(ocean, rel_map, 0.0)) if ocean is not None else np.abs(rel_map)
    tot = a.sum()
    return {r: float(a[region_mask(lat, lon, r)].sum() / tot) for r in regions}


def region_area_share(lat, lon, ocean: np.ndarray, regions=("arctic", "north_pacific", "tropical_pacific", "north_atlantic")) -> Dict[str, float]:
    """Fraction of ocean area in each region — the share a spatially uniform relevance would give."""
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon))) * ocean
    return {r: float(w[region_mask(lat, lon, r)].sum() / w.sum()) for r in regions}
