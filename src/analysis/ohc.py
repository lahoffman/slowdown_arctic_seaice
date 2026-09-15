"""
ohc.py — upper-ocean heat content as a predictor: the Labe & Barnes test and the sea-ice ledger item.

Works on the cmip6 forcing group only (50 members; the AWS store has no smbb historical TEMP). All
anomalies are relative to the 50-member mean at each cell and year. Member-block cross-validation with
blocks of 10 members; a single split (last block held out) is the default so the first look is quick.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

from src.data.cesm2le.download import CMIP6_MEMBERS

# Regions on the CAM grid: (lat0, lat1, lon0, lon1) in 0–360
REGIONS = {
    "barents_kara":  (65, 82, 15, 100),
    "nordic_gin":    (60, 80, 340, 15),
    "labrador_baffin": (55, 75, 290, 320),
    "bering_chukchi": (55, 75, 170, 200),
    "subpolar_atl":  (45, 65, 300, 350),
    "north_pacific": (25, 50, 140, 240),
    "trop_pacific":  (-10, 10, 160, 280),
    "global":        (-90, 90, 0, 360),
}


# ---------------------------------------------------------------- loading
def _project_key(member_id: str) -> str:
    """'r10i1181p1f1' → '1181.010' (the LE2-<init>.<real> naming used in download.py)."""
    m = re.match(r"r(\d+)i(\d+)p", member_id)
    return f"{m.group(2)}.{int(m.group(1)):03d}"


def load_ohc(path: Path, months: Sequence[int] = range(1, 13)) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    OHC (nens, nyear, lat, lon) as the mean over ``months`` of each year (default annual; (6, 7, 8) for JJA),
    from the monthly file, one member at a time (the full monthly array is ~15 GB), reordered to the project's
    cmip6 member order; returns (ohc, years, lat, lon).
    """
    from src.data.cesm2le.ohc import seasonal_mean
    with xr.open_dataset(path) as ds:
        lat, lon = ds["lat"].values, ds["lon"].values
        ids = [str(v) for v in ds["member_id"].values]
        yr, mo = ds["year"].values.astype(int), ds["month"].values.astype(int)
        years = np.unique(yr); sel = np.flatnonzero(np.isin(mo, list(months)))
        out = np.empty((len(ids), years.size, lat.size, lon.size), np.float32)
        for i in range(len(ids)):
            monthly = ds["ohc"].isel(nens=i, time=sel).values
            out[i] = seasonal_mean(monthly, yr[sel], mo[sel], years, months)
    want = [f"{str(m).split('.')[0]}.{str(m).split('.')[1]:0>3}" for m in CMIP6_MEMBERS]
    have = {_project_key(i): k for k, i in enumerate(ids)}
    order = [have[w] for w in want if w in have]
    if len(order) != len(ids):
        raise ValueError(f"member ids do not map onto CMIP6_MEMBERS ({len(order)}/{len(ids)})")
    return out[order], years, lat, lon


def demean(field: np.ndarray) -> np.ndarray:
    """Anomaly from the ensemble mean at each year/cell (field: (nens, nyear, ...))."""
    return field - np.nanmean(field, axis=0, keepdims=True)


def area_mean(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, box: Tuple[float, float, float, float]) -> np.ndarray:
    """cos(lat)-weighted mean of (..., lat, lon) over a box (lon0 > lon1 wraps through 0°)."""
    la0, la1, lo0, lo1 = box
    lonm = np.mod(lon, 360)
    if (lo0, lo1) == (0, 360):
        lsel = np.ones(lon.size, bool)
    elif lo0 <= lo1:
        lsel = (lonm >= lo0) & (lonm <= lo1)
    else:                                                     # box crossing 0°E
        lsel = (lonm >= lo0) | (lonm <= lo1)
    m = ((lat >= la0) & (lat <= la1))[:, None] & lsel[None, :]
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, lon.size)) * m
    w = np.where(np.isfinite(field[(0,) * (field.ndim - 2)]), w, 0.0)
    return np.nansum(field * w, axis=(-2, -1)) / w.sum()


def region_scalars(anom: np.ndarray, lat, lon, regions: Dict = REGIONS) -> Dict[str, np.ndarray]:
    return {k: area_mean(anom, lat, lon, box) for k, box in regions.items()}


# ---------------------------------------------------------------- targets
def trend_anomaly(series: np.ndarray, years: np.ndarray, onsets: np.ndarray, window: int = 10, offset: int = 0) -> np.ndarray:
    """OLS slope over t+offset … t+offset+window−1 minus the ensemble-mean slope; (nens, n_onsets), NaN past the end."""
    x = np.arange(window, dtype=float); x -= x.mean()
    out = np.full((series.shape[0], onsets.size), np.nan)
    for k, t in enumerate(onsets):
        i0 = int(np.searchsorted(years, t)) + offset
        if i0 + window <= years.size:
            out[:, k] = (series[:, i0:i0 + window] * x).sum(1) / (x * x).sum()
    return out - np.nanmean(out, axis=0, keepdims=True)


def binary_labels(trend_anom: np.ndarray, sign: int = +1, n_sigma: float = 1.0) -> np.ndarray:
    """z = anom/σ_pool; slowdown = sign·z > n_sigma (SIE: +1, slower decline; GMT: −1, slower warming)."""
    z = trend_anom / np.nanstd(trend_anom)
    return (sign * z > n_sigma).astype(float)


# ---------------------------------------------------------------- splits & models
def member_blocks(nens: int, block: int = 10) -> List[np.ndarray]:
    return [np.arange(i, min(i + block, nens)) for i in range(0, nens, block)]


def splits(nens: int, block: int = 10, single: bool = True) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """(train_members, test_members); the last block held out by default, every block with single=False."""
    blocks = member_blocks(nens, block)
    tests = [blocks[-1]] if single else blocks
    for te in tests:
        tr = np.setdiff1d(np.arange(nens), te)
        yield tr, te


def flatten(X: np.ndarray, members: np.ndarray, ok_year: np.ndarray) -> np.ndarray:
    """(nens, nyear, …) → (n_samples, features) for the given members and valid onset years."""
    sub = X[members][:, ok_year]
    return sub.reshape(-1, *sub.shape[2:]) if sub.ndim > 2 else sub.reshape(-1, 1)


def fit_score(Xtr, ytr, Xte, yte, task: str, alpha: float = 10.0) -> Tuple[float, np.ndarray]:
    """Ridge (R²) or balanced logistic (AUROC) on standardised features; returns (score, test prediction)."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    if task == "regression":
        m = Ridge(alpha=alpha).fit(Xtr, ytr); p = m.predict(Xte)
        return 1 - np.sum((yte - p) ** 2) / np.sum((yte - yte.mean()) ** 2), p
    m = LogisticRegression(class_weight="balanced", C=1.0, max_iter=3000).fit(Xtr, ytr.astype(int))
    p = m.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte.astype(int), p), p


def fit_pcs(Xtr: np.ndarray, Xte: np.ndarray, n_pcs: int) -> Tuple[np.ndarray, np.ndarray, PCA]:
    """PCA fitted on the training samples (rows = samples, cols = ocean cells); returns train/test scores."""
    pca = PCA(n_components=n_pcs, svd_solver="randomized", random_state=0).fit(Xtr)
    return pca.transform(Xtr), pca.transform(Xte), pca


def ann_score(Xtr, ytr, Xte, yte, task: str, hidden=(30, 30), seed: int = 0) -> float:
    """A small LB22-style MLP (two layers of 30) on the same features; one seed, no early-stopping tuning."""
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    if task == "regression":
        m = MLPRegressor(hidden_layer_sizes=hidden, alpha=1e-3, max_iter=500, early_stopping=True,
                         random_state=seed).fit(Xtr, ytr)
        p = m.predict(Xte); return 1 - np.sum((yte - p) ** 2) / np.sum((yte - yte.mean()) ** 2)
    m = MLPClassifier(hidden_layer_sizes=hidden, alpha=1e-3, max_iter=500, early_stopping=True,
                      random_state=seed).fit(Xtr, ytr.astype(int))
    return roc_auc_score(yte.astype(int), m.predict_proba(Xte)[:, 1])


def member_order_check(ohc_gm: np.ndarray, gmt_anom: np.ndarray) -> Tuple[float, float]:
    """Mean correlation of matched members vs mismatched pairs (global-mean OHC vs GMT anomalies)."""
    n = min(ohc_gm.shape[0], gmt_anom.shape[0])
    C = np.corrcoef(np.vstack([ohc_gm[:n], gmt_anom[:n]]))[:n, n:]
    return float(np.nanmean(np.diag(C))), float(np.nanmean(C[~np.eye(n, dtype=bool)]))
