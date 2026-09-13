"""
occlusion.py — region-occlusion test (revision step 5.3).

Zero (= mean anomaly) one region of the standardised test maps and re-score
the trained CNN; the drop in AUROC / AUPRC is that region's contribution to
skill — a model-agnostic check on the LRP maps.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import xarray as xr
from sklearn.metrics import average_precision_score, roc_auc_score

#: lon in 0–360. 'arctic' is everything north of 65°N.
REGIONS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "arctic":          {"lat": (65, 90),   "lon": (0, 360)},
    "north_pacific":   {"lat": (20, 60),   "lon": (120, 250)},
    "tropical_pacific": {"lat": (-15, 15), "lon": (120, 285)},
    "north_atlantic":  {"lat": (30, 70),   "lon": (280, 360)},
    "not_arctic":      {"lat": (-90, 65),  "lon": (0, 360)},
}


def region_mask(lat: np.ndarray, lon: np.ndarray, name: str) -> np.ndarray:
    """Boolean (nx, ny) mask of one REGIONS entry."""
    r = REGIONS[name]
    lat2d, lon2d = np.meshgrid(lat, np.mod(lon, 360), indexing="ij")
    return (lat2d >= r["lat"][0]) & (lat2d < r["lat"][1]) & (lon2d >= r["lon"][0]) & (lon2d < r["lon"][1])


def occlude(maps: np.ndarray, mask: np.ndarray, land_fill: float = -10.0) -> np.ndarray:
    """Set ocean cells inside ``mask`` to 0 (the standardised mean); land sentinel kept."""
    out = maps.copy()
    sel = mask[None, :, :, None] & ~np.isclose(out, land_fill)
    out[sel] = 0.0
    return out


def score(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    return {"AUROC": float(roc_auc_score(y, p)), "AUPRC": float(average_precision_score(y, p))}


def occlusion_dataset(results: Dict[str, Dict[str, np.ndarray]], regions, runs) -> xr.Dataset:
    """results[metric][region] = array over runs → Dataset(metric, region, run)."""
    metrics = list(results)
    data = np.array([[results[m][r] for r in regions] for m in metrics])   # (metric, region, run)
    return xr.Dataset({"value": (("metric", "region", "run"), data)},
                      coords={"metric": metrics, "region": list(regions), "run": list(runs)})
