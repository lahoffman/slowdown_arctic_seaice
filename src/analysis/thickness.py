"""
thickness.py — sea-ice thickness at onset as a predictor of the following decade's trend (step 8.12).

Works on the CICE grid (no regrid): monthly grid-cell-mean thickness ``hi`` north of 60°N,
demeaned per forcing group. Provides regional volumes, area-weighted EOFs of the thickness
anomaly, pointwise correlation maps with the (offset) trend anomaly, and member-block
out-of-sample R² for scalar sets built from them — the linear look before any CNN.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import netCDF4 as nc
import numpy as np

from src.data.cesm2le.slowdowns_relative import group_mean_trends
from . import baselines as bl
from .residual import _ols, _r2

# (lat range, lon range in 0–360) on TLAT/TLON
SECTORS: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {
    "central":   ((80, 90), (0, 360)),
    "beaufort":  ((68, 80), (200, 245)),
    "chukchi_ess": ((68, 80), (150, 200)),
    "laptev_kara": ((68, 82), (60, 150)),
    "barents":   ((68, 82), (15, 60)),
    "caa_greenland": ((68, 84), (245, 345)),
}


def load_hi_month(template: Dict[str, str], groups: Sequence[str], month: str, years_all: np.ndarray,
                  years: np.ndarray, var: str = "hi_mon") -> np.ndarray:
    """Thickness (nens, nyear, nj, ni) for ``month`` and the requested years; NaN → 0 (land/open water)."""
    arrs = []
    for g in groups:
        with nc.Dataset(template[g].format(month=month)) as ds:
            a = np.array(ds[var][:], np.float32)
        arrs.append(a)
    hi = np.concatenate(arrs, axis=0)
    idx = np.searchsorted(years_all, years)
    hi = hi[:, idx]
    hi[~np.isfinite(hi)] = 0.0
    return hi


def group_demean(field: np.ndarray) -> np.ndarray:
    """Remove each forcing group's mean at every cell and year; field (nens, nyear, ...)."""
    out = np.empty_like(field)
    for sl in (slice(0, 50), slice(50, 100)):
        out[sl] = field[sl] - field[sl].mean(axis=0, keepdims=True)
    return out


def sector_volumes(hi: np.ndarray, tlat: np.ndarray, tlon: np.ndarray, tarea: np.ndarray,
                   sectors: Dict = SECTORS) -> Dict[str, np.ndarray]:
    """Volume (10³ km³) per sector, (nens, nyear); tarea in cm², hi in m."""
    lon = np.mod(tlon, 360)
    out = {}
    for name, ((la0, la1), (lo0, lo1)) in sectors.items():
        m = (tlat >= la0) & (tlat < la1) & (lon >= lo0) & (lon < lo1)
        out[name] = (hi[..., m] * tarea[m]).sum(-1) * 1e-4 / 1e9 / 1e3
    return out


def eofs(anom: np.ndarray, weights: np.ndarray, mask: np.ndarray, n: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Area-weighted EOFs of the thickness anomaly over ``mask`` cells.
    anom (nens, nyear, nj, ni) → PCs (nens, nyear, n), EOF patterns (n, nj, ni), explained variance fraction (n,).
    """
    nens, nyr = anom.shape[:2]
    X = anom[..., mask].reshape(nens * nyr, -1) * np.sqrt(weights[mask])[None, :]
    X = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    pcs = (U[:, :n] * S[:n]).reshape(nens, nyr, n)
    pats = np.full((n,) + anom.shape[2:], np.nan, np.float32)
    pats[:, mask] = Vt[:n] / np.sqrt(weights[mask])[None, :]
    return pcs, pats, (S[:n] ** 2) / (S ** 2).sum()


def correlation_map(target: np.ndarray, field: np.ndarray) -> np.ndarray:
    """Pointwise corr of target (nens, nyear) with field (nens, nyear, nj, ni), NaN where field is constant."""
    t = target.reshape(-1); f = field.reshape(t.size, -1)
    ok = np.isfinite(t)
    t = t[ok] - t[ok].mean(); f = f[ok] - f[ok].mean(0)
    den = np.sqrt((t @ t) * (f * f).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (t @ f) / den
    r[den == 0] = np.nan
    return r.reshape(field.shape[2:])


def scalar_set_r2(trend: np.ndarray, sets: Dict[str, List[np.ndarray]], years: np.ndarray, n_splits: int = 9
                  ) -> Dict[str, np.ndarray]:
    """Test R² per split of trend ~ each scalar set (OLS on training members). sets: name → list of (nens, nyear) arrays."""
    out = {}
    for name, arrs in sets.items():
        fields = {"y": trend, **{f"x{i}": a for i, a in enumerate(arrs)}}
        r2 = np.full(n_splits, np.nan)
        for k, te_b, va_b, tr_b in bl.iter_split_blocks(n_splits, 10):
            d = bl.split_scalars(fields, years, tr_b, va_b, te_b)
            X_tr = np.column_stack([d["tr"][f"x{i}"] for i in range(len(arrs))])
            X_te = np.column_stack([d["te"][f"x{i}"] for i in range(len(arrs))])
            ok_tr, ok_te = np.isfinite(d["tr"]["y"]) & np.isfinite(X_tr).all(1), np.isfinite(d["te"]["y"]) & np.isfinite(X_te).all(1)
            _, p, _ = _ols(X_tr[ok_tr], d["tr"]["y"][ok_tr], X_te[ok_te])
            r2[k] = _r2(d["te"]["y"][ok_te], p)
        out[name] = r2
    return out


def scalar_set_auroc(labels: np.ndarray, sets: Dict[str, List[np.ndarray]], years: np.ndarray, n_splits: int = 9
                     ) -> Dict[str, np.ndarray]:
    """Test AUROC per split of a logistic regression on each scalar set (binary label, e.g. slowdown or RILE)."""
    from sklearn.metrics import roc_auc_score
    out = {}
    for name, arrs in sets.items():
        fields = {"y": labels.astype(float), **{f"x{i}": a for i, a in enumerate(arrs)}}
        au = np.full(n_splits, np.nan)
        for k, te_b, va_b, tr_b in bl.iter_split_blocks(n_splits, 10):
            d = bl.split_scalars(fields, years, tr_b, va_b, te_b)
            X_tr = np.column_stack([d["tr"][f"x{i}"] for i in range(len(arrs))])
            X_te = np.column_stack([d["te"][f"x{i}"] for i in range(len(arrs))])
            y_tr, y_te = d["tr"]["y"].astype(int), d["te"]["y"].astype(int)
            ok_tr, ok_te = np.isfinite(X_tr).all(1), np.isfinite(X_te).all(1)
            _, s_te, _ = bl.fit_logistic(X_tr[ok_tr], y_tr[ok_tr], X_te[ok_te])
            au[k] = roc_auc_score(y_te[ok_te], s_te)
        out[name] = au
    return out
