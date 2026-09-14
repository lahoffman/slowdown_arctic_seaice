"""
interannual.py — sanity checks behind the decadal result (step 8.6).

(a) Does CESM2-LE carry the year-to-year Pacific → September SIE link at all,
    and how nonstationary is it across members and 30-yr windows
    (cf. Bonan & Blanchard-Wrigglesworth 2020 for CESM1-LE)?
(b) How much of the "ice state predicts the decadal trend" result is the onset
    year sitting at the start of the trend window? Compare trend windows
    t..t+9, t+1..t+10, t+2..t+11 regressed on SIE(t).
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from src.data.cesm2le.slowdowns_relative import group_mean_trends
from . import baselines as bl
from .residual import _ols, _r2


def member_correlations(sie_anom: np.ndarray, index: np.ndarray) -> np.ndarray:
    """corr(SIE anomaly, index) per member over all years, (nens,)."""
    a = sie_anom - sie_anom.mean(1, keepdims=True); b = index - index.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))


def running_correlations(sie_anom: np.ndarray, index: np.ndarray, years: np.ndarray,
                         window: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """Per-member correlation in sliding ``window``-yr blocks → (nens, nwin), centre years."""
    n = years.size - window + 1
    out = np.stack([member_correlations(sie_anom[:, k:k + window], index[:, k:k + window]) for k in range(n)], 1)
    return out, years[:n] + window // 2


def lagged_regression_r2(sie_anom: np.ndarray, indices: Dict[str, np.ndarray], years: np.ndarray,
                         n_splits: int = 9) -> Dict[str, np.ndarray]:
    """
    Test R² of SIE(t) ~ SIE(t−1) [+ concurrent JJA indices], member-block splits.
    Returns {'persistence': (nsplit,), 'persistence+<set>': (nsplit,), ...}.
    """
    prev = np.full_like(sie_anom, np.nan); prev[:, 1:] = sie_anom[:, :-1]
    fields = {"y": sie_anom, "prev": prev, **indices}
    sets = {"nino34": ["nino34"], "ipo": ["ipo"], "pacific": ["nino34", "ipo"], "arctic": ["arctic"]}
    out = {k: np.full(n_splits, np.nan) for k in ["persistence"] + [f"persistence+{s}" for s in sets]}
    for k, te_b, va_b, tr_b in bl.iter_split_blocks(n_splits, 10):
        d = bl.split_scalars(fields, years, tr_b, va_b, te_b)
        ok_tr, ok_te = np.isfinite(d["tr"]["prev"]), np.isfinite(d["te"]["prev"])
        y_tr, y_te = d["tr"]["y"][ok_tr], d["te"]["y"][ok_te]
        _, p, _ = _ols(d["tr"]["prev"][ok_tr, None], y_tr, d["te"]["prev"][ok_te, None])
        out["persistence"][k] = _r2(y_te, p)
        for s, f in sets.items():
            X_tr = np.column_stack([d["tr"]["prev"][ok_tr]] + [d["tr"][v][ok_tr] for v in f])
            X_te = np.column_stack([d["te"]["prev"][ok_te]] + [d["te"][v][ok_te] for v in f])
            _, p, _ = _ols(X_tr, y_tr, X_te)
            out[f"persistence+{s}"][k] = _r2(y_te, p)
    return out


def trend_anomaly_offset(sie: np.ndarray, years: np.ndarray, onsets: np.ndarray, window: int = 10,
                         offset: int = 0) -> np.ndarray:
    """
    Group-relative trend anomaly over years t+offset .. t+offset+window−1 for each onset t.
    offset=0 is the paper's definition (onset year inside the window).
    """
    x = np.arange(window, dtype=float); x -= x.mean()
    out = np.full((sie.shape[0], onsets.size), np.nan)
    for k, t in enumerate(onsets):
        i0 = int(np.searchsorted(years, t)) + offset
        if i0 + window > years.size:
            continue
        seg = sie[:, i0:i0 + window]
        out[:, k] = (seg * x).sum(1) / (x * x).sum()
    return out - group_mean_trends(out, "group")


def offset_r2(sie: np.ndarray, sie_anom: np.ndarray, years: np.ndarray, onsets: np.ndarray,
              offsets: Sequence[int] = (0, 1, 2), window: int = 10, n_splits: int = 9) -> Dict[int, np.ndarray]:
    """Test R² of trend anomaly (window starting t+offset) ~ SIE anomaly(t), per split."""
    out = {}
    for off in offsets:
        tr = trend_anomaly_offset(sie, years, onsets, window, off)
        r2 = np.full(n_splits, np.nan)
        for k, te_b, va_b, tr_b in bl.iter_split_blocks(n_splits, 10):
            d = bl.split_scalars({"y": tr, "x": sie_anom}, onsets, tr_b, va_b, te_b)
            ok_tr, ok_te = np.isfinite(d["tr"]["y"]), np.isfinite(d["te"]["y"])
            _, p, _ = _ols(d["tr"]["x"][ok_tr, None], d["tr"]["y"][ok_tr], d["te"]["x"][ok_te, None])
            r2[k] = _r2(d["te"]["y"][ok_te], p)
        out[off] = r2
    return out
