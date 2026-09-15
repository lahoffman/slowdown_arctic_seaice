"""
obs_predict.py — the scalar baselines applied to observations (AIES §8, Fig. 8b).

Fits the logistic baselines on CESM2-LE (one fit per member-block split, so the
spread across fits is visible), applies them to the observed September SIE
anomaly and JJA Pacific indices year by year, and — where the following decade
is already observed — computes the observed offset-window trend z-score so the
predictions can be checked against what happened.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.linear_model import LogisticRegression

from src.data.cesm2le.slowdowns import load_sie_monthly_files
from src.data.observations import obs_input as oi
from src.data.observations.ersst.climate_indices import _correct_forced_mean_and_trend
from . import baselines as bl

FEATURE_SETS = {"logit_sie_anom": ["sie_anom"], "logit_ipo": ["ipo"], "logit_sie_ipo": ["sie_anom", "ipo"],
                "logit_sie_pacific": ["sie_anom", "nino34", "ipo"]}
ERSST_T0 = "1854-01-01"


def _jja_by_year(monthly: np.ndarray, t0: Optional[str], years: np.ndarray, times=None) -> np.ndarray:
    """JJA mean per year of a monthly series; time axis from ``times`` if given, else monthly from ``t0``."""
    index = pd.DatetimeIndex(times) if times is not None else pd.date_range(t0, periods=monthly.size, freq="MS")
    s = pd.Series(monthly, index=index)
    s = s[s.index.month.isin([6, 7, 8])]
    return s.groupby(s.index.year).mean().reindex(years).values


def _nsidc_series(nsidc_events_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    with xr.open_dataset(nsidc_events_file) as ds:
        return ds["yearice"].values.astype(int), ds["seaice"].values.astype(float)


def _poly_reference(years: np.ndarray, x: np.ndarray, deg: int) -> np.ndarray:
    """Polynomial fit of degree ``deg`` to the finite values of x(years); NaN where x is missing."""
    ok = np.isfinite(x); t = years - years.mean(); ref = np.full(years.size, np.nan)
    ref[ok] = np.polyval(np.polyfit(t[ok], x[ok], deg), t[ok])
    return ref


def _index_series(f: Path, var: str, years: np.ndarray, t0: Optional[str]) -> np.ndarray:
    """JJA-by-year of one index file; uses its ``time``/``dates`` coordinate when present (OISST), else ``t0`` (ERSST, 1854-01)."""
    with xr.open_dataset(f) as ds:
        v = ds[var].values
        tcoord = next((ds[c].values for c in ("time", "dates") if c in ds), None)
        start = ds.attrs.get("start_date")
    if tcoord is not None and len(tcoord) == v.size:
        return _jja_by_year(v, None, years, times=tcoord)
    return _jja_by_year(v, t0 or start or ERSST_T0, years)


def observed_scalars(years: np.ndarray, forced_method: str, nsidc_events_file: Path, metrics_dir: Path,
                     ipo_file: Path, nino34_file: Path, index_t0: Optional[str] = ERSST_T0) -> Dict[str, np.ndarray]:
    """Observed sie_anom (forced removed as in the CNN pipeline), JJA IPO (filtered) and Niño3.4, each (nyear,)."""
    if forced_method in ("linear", "quadratic"):         # observation-only reference (NaN-safe)
        oy, osie = _nsidc_series(nsidc_events_file)
        sie = pd.Series(osie, index=oy).reindex(years).values
        out = {"sie_anom": sie - _poly_reference(years, sie, 1 if forced_method == "linear" else 2)}
    else:
        out = {"sie_anom": oi.observed_sie_anomaly(nsidc_events_file, years, forced_method, metrics_dir)}
    out["ipo"] = _index_series(ipo_file, "ipo_filtered", years, index_t0)
    out["nino34"] = _index_series(nino34_file, "nino34", years, index_t0)
    return out


def fit_predictors(fields: Dict[str, np.ndarray], labels: np.ndarray, years: np.ndarray,
                   feature_sets: Dict[str, List[str]] = FEATURE_SETS, n_splits: int = 9
                   ) -> Dict[str, List[Tuple[np.ndarray, np.ndarray, LogisticRegression]]]:
    """One balanced logistic fit per split (training blocks only) → {set: [(mu, sd, clf), ...]}."""
    out = {}
    for name, feats in feature_sets.items():
        fits = []
        for k, te_b, va_b, tr_b in bl.iter_split_blocks(n_splits, 10):
            d = bl.split_scalars({**{f: fields[f] for f in feats}, "label": labels}, years, tr_b, va_b, te_b)
            X = np.column_stack([d["tr"][f] for f in feats]); y = d["tr"]["label"].astype(int)
            ok = np.isfinite(X).all(1)
            mu, sd = X[ok].mean(0), X[ok].std(0) + 1e-12
            clf = LogisticRegression(class_weight="balanced", max_iter=2000).fit((X[ok] - mu) / sd, y[ok])
            clf.prevalence_ = float(y[ok].mean())            # for calibration of the balanced fit
            fits.append((mu, sd, clf))
        out[name] = fits
    return out


def calibrate(p_balanced: np.ndarray, prevalence: float) -> np.ndarray:
    """Undo class balancing: a balanced fit scales the odds by (1−π)/π, so multiply them back by π/(1−π)."""
    odds = p_balanced / (1 - p_balanced) * prevalence / (1 - prevalence)
    return odds / (1 + odds)


def predict_observed(fits, obs: Dict[str, np.ndarray], feature_sets=FEATURE_SETS, calibrated: bool = True
                     ) -> Dict[str, np.ndarray]:
    """
    P(slowdown) for every year and split-fit → {set: (n_fits, nyear)}; NaN where an input is missing.
    ``calibrated`` converts the balanced-fit output to a probability at the training prevalence
    (0.5 balanced ≙ the base rate), which is what a reader should compare with the base rate line.
    """
    out = {}
    for name, feats in feature_sets.items():
        X = np.column_stack([obs[f] for f in feats]); ok = np.isfinite(X).all(1)
        P = np.full((len(fits[name]), X.shape[0]), np.nan)
        for i, (mu, sd, clf) in enumerate(fits[name]):
            p = clf.predict_proba((X[ok] - mu) / sd)[:, 1]
            P[i, ok] = calibrate(p, clf.prevalence_) if calibrated else p
        out[name] = P
    return out


def observed_offset_z(years: np.ndarray, nsidc_events_file: Path, labels_file: Path, window: int = 10,
                      offset: int = 1, label_ref: str = "linear", sigma_from: str = "model") -> Tuple[np.ndarray, float]:
    """
    Observed trend-anomaly z at onset t: OLS slope of NSIDC September SIE over t+offset … t+offset+window−1
    minus the slope of an observation-only reference (``label_ref`` 'linear' or 'quadratic' fit to the full
    NSIDC record), divided by σ — the model's pooled σ (``sigma_from='model'``, the paper's definition) or the
    standard deviation of the observed trend anomalies (``'obs'``, as in v1). NaN where the window is not
    yet observed. Returns (z, sigma).
    """
    oy, osie = _nsidc_series(nsidc_events_file)
    ref = _poly_reference(oy, osie, {"linear": 1, "quadratic": 2}[label_ref])
    x = np.arange(window) - (window - 1) / 2
    anom = np.full(years.size, np.nan)
    for i, t in enumerate(years):
        w = np.arange(t + offset, t + offset + window)
        if w[-1] > oy[-1] or w[0] < oy[0]:
            continue
        idx = np.searchsorted(oy, w)
        if np.isfinite(osie[idx]).all() and np.isfinite(ref[idx]).all():
            anom[i] = ((x @ osie[idx]) - (x @ ref[idx])) / (x @ x)
    if sigma_from == "model":
        with xr.open_dataset(labels_file) as ds:
            sigma = float(np.nanmean(ds["sigma"].values))
    else:
        sigma = float(np.nanstd(anom))
    return anom / sigma, sigma


def observed_slowdown_years(nsidc_events_file: Path):
    """Onset years flagged as slowdowns in the NSIDC events file (v1 definition, observed linear-trend threshold)."""
    with xr.open_dataset(nsidc_events_file) as ds:
        y, s = ds["year"].values.astype(int), ds["slowdown"].values.astype(int)
    return y[s == 1]


def cnn_vote_fraction(pred_dir: Path, years: np.ndarray, threshold: float = 0.5) -> Optional[np.ndarray]:
    """Fraction of CNN runs voting slowdown per year, from cnn_prediction_*.nc in ``pred_dir``; None if absent."""
    files = sorted(pred_dir.glob("cnn_prediction_*_M*_*.nc"))
    if not files:
        return None
    probs, yrs = [], None
    for f in files:
        with xr.open_dataset(f) as ds:
            probs.append(ds["y_prob"].values); yrs = ds["year"].values.astype(int) if "year" in ds.coords else yrs
    frac = (np.array(probs) >= threshold).mean(0)
    if yrs is None or yrs.size != frac.size:
        return frac if frac.size == years.size else None
    return pd.Series(frac, index=yrs).reindex(years).values


def summary_markdown(years, obs, probs, z, frac, forced_method, claim=(2016, 2025), base_rate=0.17,
                     label_note="") -> str:
    c0, c1 = claim; sel = (years >= c0) & (years <= c1)
    lines = [f"# Observed baseline predictions — predictor anomaly reference `{forced_method}`\n",
             f"Calibrated P(slowdown in t+1…t+10) from logistic fits on CESM2-LE (median over 9 split-fits; base rate {base_rate:.2f}); "
             f"observed offset z where the decade is complete ({label_note}).\n",
             "| onset t | SIE anom [Mkm²] | IPO | Niño3.4 | " + " | ".join(probs) + (" | CNN votes" if frac is not None else "") + " | obs z |",
             "|---|---|---|---|" + "---|" * len(probs) + ("---|" if frac is not None else "") + "---|"]
    for i, t in enumerate(years):
        row = [f"{t}", f"{obs['sie_anom'][i]:+.2f}", f"{obs['ipo'][i]:+.2f}", f"{obs['nino34'][i]:+.2f}"]
        row += [f"{np.nanmedian(probs[p][:, i]):.2f}" for p in probs]
        if frac is not None:
            row.append(f"{frac[i]:.2f}" if np.isfinite(frac[i]) else "—")
        row.append(f"{z[i]:+.2f}" if np.isfinite(z[i]) else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append(f"\nMean P over {c0}–{c1}: " + ", ".join(f"{p} {np.nanmedian(probs[p][:, sel], 0).mean():.2f}" for p in probs)
                 + f"; base rate {base_rate:.2f}.\n")
    return "\n".join(lines) + "\n"
