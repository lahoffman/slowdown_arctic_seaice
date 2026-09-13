"""
sweep.py — score a subset of the logistic baselines over the 9 block splits,
optionally on a subset of test members (e.g. one forcing group). Shared by
the label-sensitivity sweep (step 4.2) and the per-group check (step 4.4).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import xarray as xr

from . import baselines as bl

METRICS = ("AUROC", "AUPRC", "F1")


def score_models(fields: Dict[str, np.ndarray], labels: np.ndarray, years: np.ndarray,
                 models: Sequence[str], member_filter: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 cnn_preds: Optional[Dict[int, List]] = None, n_splits: int = 9) -> xr.Dataset:
    """
    Fit on training members, score on test members, for the named BASELINE_FEATURES.

    member_filter(member_index_array) → bool mask restricts the *test* samples scored
    (training is unchanged). cnn_preds[k] = [(y_true, y_prob, thr), ...] adds a
    'cnn_median' model scored on the same subset. Returns Dataset(metric, model, split).
    """
    all_models = list(models) + (["cnn_median"] if cnn_preds else [])
    out = np.full((len(METRICS), len(all_models), n_splits), np.nan)
    prev = np.full(n_splits, np.nan)
    for k, te_b, va_b, tr_b in bl.iter_split_blocks(n_splits, 10):
        d = bl.split_scalars({**fields, "label": labels}, years, tr_b, va_b, te_b)
        y_tr, y_te = d["tr"]["label"].astype(int), d["te"]["label"].astype(int)
        yr_tr, yr_te = d["tr"]["year"].astype(int), d["te"]["year"].astype(int)
        d["tr"]["yearclim"] = bl._logit(bl.year_climatology(y_tr, yr_tr, yr_tr))
        d["te"]["yearclim"] = bl._logit(bl.year_climatology(y_tr, yr_tr, yr_te))
        keep = np.ones(y_te.size, bool) if member_filter is None else member_filter(d["te"]["member"].astype(int))
        prev[k] = y_te[keep].mean()
        for j, name in enumerate(models):
            feats = bl.BASELINE_FEATURES[name]
            X_tr = np.column_stack([d["tr"][f] for f in feats]); X_te = np.column_stack([d["te"][f] for f in feats])
            ok_tr, ok_te = np.isfinite(X_tr).all(1), np.isfinite(X_te).all(1) & keep
            s_tr, s_te, _ = bl.fit_logistic(X_tr[ok_tr], y_tr[ok_tr], X_te[ok_te])
            thr = bl.pr_intersection_threshold(y_tr[ok_tr], s_tr)
            m = bl.compute_metrics(y_te[ok_te], s_te, thr)
            out[:, j, k] = [m[x] for x in METRICS]
        if cnn_preds and cnn_preds.get(k):
            runs = [bl.compute_metrics(y_te[keep], yp[keep], thr) for _, yp, thr in cnn_preds[k]
                    if yp.size == y_te.size]
            if runs:
                out[:, -1, k] = [np.median([r[x] for r in runs]) for x in METRICS]
    return xr.Dataset({"value": (("metric", "model", "split"), out), "prevalence": ("split", prev)},
                      coords={"metric": list(METRICS), "model": all_models, "split": np.arange(n_splits)})
