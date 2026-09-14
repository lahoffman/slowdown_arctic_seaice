"""
residual.py — what explains the decadal SIE trend that the ice state does not? (step 8.1)

Stage 1 regresses the continuous trend anomaly on the September SIE anomaly at
onset (member-block splits, fit on training members). Stage 2 asks how much of
the *residual* the SST indices explain, at onset and averaged over the trend
decade (concurrent). No CNN; the 1σ threshold is not used at all.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

from . import baselines as bl

INDEX_SETS: Dict[str, Sequence[str]] = {
    "arctic":  ("arctic",),
    "nino34":  ("nino34",),
    "ipo":     ("ipo",),
    "pacific": ("nino34", "ipo"),
    "indices": ("arctic", "nino34", "ipo"),
}


def window_mean(field: np.ndarray, years: np.ndarray, target_years: np.ndarray, window: int) -> np.ndarray:
    """Mean of ``field[:, t : t + window]`` for every target year; field is (nens, nyear, ...)."""
    out = np.full((field.shape[0], target_years.size) + field.shape[2:], np.nan, field.dtype)
    for k, t in enumerate(target_years):
        i0 = int(np.searchsorted(years, t))
        if i0 + window <= years.size and years[i0] == t:
            out[:, k] = np.nanmean(field[:, i0:i0 + window], axis=1)
    return out


def _ols(X_tr, y_tr, X_te):
    A = np.column_stack([np.ones(len(X_tr)), X_tr])
    beta = np.linalg.lstsq(A, y_tr, rcond=None)[0]
    return A @ beta, np.column_stack([np.ones(len(X_te)), X_te]) @ beta, beta


def _r2(y, yhat):
    return 1.0 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


def residual_skill(trend_anom: np.ndarray, sie_anom: np.ndarray, idx_onset: Dict[str, np.ndarray],
                   idx_conc: Dict[str, np.ndarray], years: np.ndarray, n_splits: int = 9,
                   extra_state: Optional[Dict[str, np.ndarray]] = None) -> xr.Dataset:
    """
    Per split: test R² of trend ~ ice state (stage 1) and of residual ~ index set (stage 2),
    for onset-year and concurrent (decade-mean) indices. Returns Dataset(stage/set, timing, split).
    ``extra_state`` (e.g. {'siv_anom': …}) adds scalars to the stage-1 state alongside SIE.
    """
    sets = list(INDEX_SETS)
    extra_state = extra_state or {}
    r2_stage1 = np.full(n_splits, np.nan)
    r2_resid = np.full((2, len(sets), n_splits), np.nan)          # timing, set, split
    r2_full = np.full((2, len(sets), n_splits), np.nan)           # trend ~ SIE + set
    fields = {"trend": trend_anom, "sie_anom": sie_anom, **extra_state,
              **{f"{k}_onset": v for k, v in idx_onset.items()},
              **{f"{k}_conc": v for k, v in idx_conc.items()}}
    for k, te_b, va_b, tr_b in bl.iter_split_blocks(n_splits, 10):
        d = bl.split_scalars(fields, years, tr_b, va_b, te_b)
        y_tr, y_te = d["tr"]["trend"], d["te"]["trend"]
        state = ["sie_anom"] + list(extra_state)
        s_tr = np.column_stack([d["tr"][f] for f in state]); s_te = np.column_stack([d["te"][f] for f in state])
        fit_tr, fit_te, _ = _ols(s_tr, y_tr, s_te)
        r2_stage1[k] = _r2(y_te, fit_te)
        res_tr, res_te = y_tr - fit_tr, y_te - fit_te
        for j, name in enumerate(sets):
            for i, timing in enumerate(("onset", "conc")):
                X_tr = np.column_stack([d["tr"][f"{f}_{timing}"] for f in INDEX_SETS[name]])
                X_te = np.column_stack([d["te"][f"{f}_{timing}"] for f in INDEX_SETS[name]])
                ok_tr, ok_te = np.isfinite(X_tr).all(1), np.isfinite(X_te).all(1)
                _, p_te, _ = _ols(X_tr[ok_tr], res_tr[ok_tr], X_te[ok_te])
                r2_resid[i, j, k] = _r2(res_te[ok_te], p_te)
                _, f_te, _ = _ols(np.column_stack([s_tr[ok_tr], X_tr[ok_tr]]), y_tr[ok_tr],
                                  np.column_stack([s_te[ok_te], X_te[ok_te]]))
                r2_full[i, j, k] = _r2(y_te[ok_te], f_te)
    return xr.Dataset(
        {"r2_sie": ("split", r2_stage1),
         "r2_residual": (("timing", "set", "split"), r2_resid),
         "r2_sie_plus": (("timing", "set", "split"), r2_full)},
        coords={"timing": ["onset", "conc"], "set": sets, "split": np.arange(n_splits)},
        attrs={"state": " + ".join(["sie_anom"] + list(extra_state))},
    )


def pooled_residual(trend_anom: np.ndarray, *state: np.ndarray) -> Tuple[np.ndarray, float]:
    """Residual of trend ~ state scalars fit on all member-years (descriptive maps only); returns (resid, r²)."""
    y = trend_anom.ravel(); X = np.column_stack([s.ravel() for s in state])
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    fit, _, _ = _ols(X[ok], y[ok], X[ok])
    resid = np.full(y.shape, np.nan); resid[ok] = y[ok] - fit
    return resid.reshape(trend_anom.shape), _r2(y[ok], fit)


def correlation_map(resid: np.ndarray, sst: np.ndarray) -> np.ndarray:
    """Pointwise correlation of the residual (nens, nyear) with an SST field (nens, nyear, nx, ny)."""
    r = resid.reshape(-1); f = sst.reshape(r.size, -1)
    ok = np.isfinite(r) & np.isfinite(f).all(1)
    r, f = r[ok] - r[ok].mean(), f[ok] - f[ok].mean(0)
    num = r @ f
    den = np.sqrt((r @ r) * (f * f).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (num / den).reshape(sst.shape[2:])


def summary_markdown(ds: xr.Dataset, r2_pooled: float, n_events: Optional[int] = None) -> str:
    med = lambda a: float(np.nanmedian(a))
    lines = [f"# Residual analysis — trend anomaly beyond the ice state ({ds.attrs.get('state', 'sie_anom')})\n",
             f"Stage 1: trend anomaly ~ {ds.attrs.get('state', 'sie_anom')} at onset. Test R² median over 9 splits "
             f"**{med(ds['r2_sie']):.3f}** (range {float(ds['r2_sie'].min()):.3f}–{float(ds['r2_sie'].max()):.3f}); "
             f"pooled fit R² {r2_pooled:.3f}.\n",
             "Stage 2: residual ~ SST indices, test R² (median over splits). "
             "`onset` = JJA of the onset year; `conc` = mean over the 10 JJA seasons of the trend window.\n",
             "| index set | R²(resid) onset | R²(resid) conc | ΔR² over SIE-only, onset | ΔR² over SIE-only, conc |",
             "|---|---|---|---|---|"]
    for s in ds["set"].values:
        ro, rc = med(ds["r2_residual"].sel(set=s, timing="onset")), med(ds["r2_residual"].sel(set=s, timing="conc"))
        do = med(ds["r2_sie_plus"].sel(set=s, timing="onset") - ds["r2_sie"])
        dc = med(ds["r2_sie_plus"].sel(set=s, timing="conc") - ds["r2_sie"])
        lines.append(f"| {s} | {ro:+.3f} | {rc:+.3f} | {do:+.3f} | {dc:+.3f} |")
    lines.append("\nNegative R² = the fitted index set predicts the test residual worse than its mean.")
    return "\n".join(lines) + "\n"
