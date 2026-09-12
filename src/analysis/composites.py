"""
composites.py — SST and LRP composites by prediction outcome.

Composites are formed per split × seed on the training partition (where LRP
was computed), then averaged across all split × seed pairs. Accumulation is
streaming so the full SST stack for all splits is never held in memory.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter
from sklearn.metrics import precision_recall_curve

SCENARIOS = ("TP", "FP", "TN", "FN", "ALL_SLOW", "ALL_NONSLOW")
LAND_FILL = -5.0   # values below this are the land sentinel (−10)

#: Region boxes for relevance summaries (lat, lon in 0–360).
REGIONS = {
    "Arctic":                  dict(lat=(60, 90),  lon=(0, 360)),
    "Tropical Pacific (ENSO)": dict(lat=(-10, 10), lon=(170, 270)),
    "North Pacific (PDO)":     dict(lat=(15, 55),  lon=(140, 230)),
    "Pacific (IPO)":           dict(lat=(-55, 55), lon=(140, 270)),
    "Atlantic (AMO)":          dict(lat=(0, 60),   lon=(280, 350)),
    "Indian Ocean (IOD)":      dict(lat=(-35, 15), lon=(50, 100)),
    "Niño 3":                  dict(lat=(-5, 5),   lon=(210, 270)),
    "Niño 4":                  dict(lat=(-5, 5),   lon=(160, 210)),
}
NINO_BOXES = {"Niño3": (210, 270), "Niño4": (160, 210), "Niño3.4": (190, 240)}


def pr_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Precision ≈ recall threshold (the project's operating rule)."""
    p, r, t = precision_recall_curve(y_true, y_score)
    return float(t[np.argmin(np.abs(p[:-1] - r[:-1]))]) if t.size else 0.5


def scenario_mask(y_true: np.ndarray, y_pred: np.ndarray, scenario: str) -> np.ndarray:
    y_true, y_pred = y_true == 1, y_pred == 1
    return {"TP": y_pred & y_true, "FP": y_pred & ~y_true, "TN": ~y_pred & ~y_true,
            "FN": ~y_pred & y_true, "ALL_SLOW": y_true, "ALL_NONSLOW": ~y_true}[scenario]


def load_lrp(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LRP attributions flattened to (n, nlat, nlon) plus lat, lon."""
    with xr.open_dataset(path) as ds:
        raw = ds["lrp_attributions"].values
        lat, lon = ds["lat"].values, ds["lon"].values
    return raw.reshape(-1, raw.shape[-3], raw.shape[-2]), lat, lon


class CompositeAccumulator:
    """Running mean of per-(split, seed) composite maps for several scenarios."""

    def __init__(self, scenarios: Iterable[str] = SCENARIOS, positive_only: bool = False):
        self.scenarios = tuple(scenarios)
        self.positive_only = positive_only
        self.sst = {s: [] for s in self.scenarios}
        self.lrp = {s: [] for s in self.scenarios}
        self.regional = {s: [] for s in self.scenarios}
        self.lat = self.lon = None

    def add(self, sst: np.ndarray, y_true: np.ndarray, y_score: np.ndarray,
            lrp: Optional[np.ndarray] = None, lat=None, lon=None) -> None:
        """Add one split × seed. ``sst`` (n, nlat, nlon) standardised with land sentinel."""
        thr = pr_threshold(y_true, y_score)
        y_pred = (y_score >= thr).astype(int)
        n = sst.shape[0] if lrp is None else min(sst.shape[0], lrp.shape[0])
        sst_f = sst[:n].astype(float)
        sst_f[sst_f < LAND_FILL] = np.nan
        if lat is not None and self.lat is None:
            self.lat, self.lon = lat, lon
        for s in self.scenarios:
            m = scenario_mask(y_true[:n], y_pred[:n], s)
            if not m.any():
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN land columns
                self._add_maps(s, sst_f[m], None if lrp is None else lrp[:n][m])

    def _add_maps(self, s, sst_sel, lrp_sel):
        self.sst[s].append(np.nanmean(sst_sel, 0))
        if lrp_sel is None:
            return
        sel = np.where(lrp_sel > 0, lrp_sel, np.nan) if self.positive_only else lrp_sel
        lm = np.nanmean(sel, 0)
        self.lrp[s].append(lm)
        if self.lat is not None:
            self.regional[s].append(regional_means(lm, self.lat, self.lon))

    def result(self, scenario: str, normalise: bool = True) -> Dict:
        """Grand-mean SST and LRP maps (97th-percentile normalised) + counts."""
        out = {"n": len(self.sst[scenario]), "lat": self.lat, "lon": self.lon}
        if not out["n"]:
            return out
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return self._result(scenario, normalise, out)

    def _result(self, scenario, normalise, out):
        sst = np.nanmean(np.stack(self.sst[scenario]), 0)
        out["sst"] = normalise_map(sst, signed=True) if normalise else sst
        if self.lrp[scenario]:
            lrp = np.nanmean(np.stack(self.lrp[scenario]), 0)
            out["lrp"] = normalise_map(lrp, signed=not self.positive_only) if normalise else lrp
            out["regional"] = {k: np.array([d[k] for d in self.regional[scenario]])
                               for k in REGIONS}
        return out


def normalise_map(a: np.ndarray, signed: bool = True, pct: float = 97) -> np.ndarray:
    """Scale by the 97th percentile of |a| (or a, if positive-only) and clip."""
    if signed:
        s = np.nanpercentile(np.abs(a), pct) + 1e-12
        return np.clip(a / s, -1, 1)
    s = np.nanpercentile(a, pct) + 1e-12
    return np.clip(a / s, 0, 1)


def smooth_nan(a: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gaussian smoothing that ignores NaNs and keeps them in place."""
    m = np.isnan(a)
    num = gaussian_filter(np.where(m, 0, a), sigma)
    den = gaussian_filter((~m).astype(float), sigma)
    with np.errstate(invalid="ignore"):
        out = num / den
    out[m] = np.nan
    return out


def region_mask(lat: np.ndarray, lon: np.ndarray, box: Dict) -> np.ndarray:
    la, lo = lat[:, None], lon[None, :]
    return ((la >= box["lat"][0]) & (la <= box["lat"][1]) &
            (lo >= box["lon"][0]) & (lo <= box["lon"][1]))


def regional_means(a: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                   regions: Dict = REGIONS) -> Dict[str, float]:
    out = {}
    for k, box in regions.items():
        v = a[region_mask(lat, lon, box)]
        v = v[np.isfinite(v)]
        out[k] = float(v.mean()) if v.size else np.nan
    return out
