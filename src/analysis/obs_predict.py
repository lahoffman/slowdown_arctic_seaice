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


def _jja_by_year(monthly: np.ndarray, t0: str, years: np.ndarray) -> np.ndarray:
    s = pd.Series(monthly, index=pd.date_range(t0, periods=monthly.size, freq="MS"))
    s = s[s.index.month.isin([6, 7, 8])]
    return s.groupby(s.index.year).mean().reindex(years).values


def observed_scalars(years: np.ndarray, forced_method: str, nsidc_events_file: Path, metrics_dir: Path,
                     ipo_file: Path, nino34_file: Path) -> Dict[str, np.ndarray]:
    """Observed sie_anom (forced removed as in the CNN pipeline), JJA IPO (filtered) and Niño3.4, each (nyear,)."""
    if forced_method == "linear":                       # NaN-safe detrend (the shared helper is not)
        with xr.open_dataset(nsidc_events_file) as ds:
            oy, osie = ds["yearice"].values.astype(int), ds["seaice"].values.astype(float)
        sie = pd.Series(osie, index=oy).reindex(years).values; ok = np.isfinite(sie)
        t = years - years.mean(); anom = np.full(years.size, np.nan)
        anom[ok] = sie[ok] - np.polyval(np.polyfit(t[ok], sie[ok], 1), t[ok])
        out = {"sie_anom": anom}
    else:
        out = {"sie_anom": oi.observed_sie_anomaly(nsidc_events_file, years, forced_method, metrics_dir)}
    with xr.open_dataset(ipo_file) as ds:
        out["ipo"] = _jja_by_year(ds["ipo_filtered"].values, ERSST_T0, years)
    with xr.open_dataset(nino34_file) as ds:
        out["nino34"] = _jja_by_year(ds["nino34"].values, ERSST_T0, years)
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
            fits.append((mu, sd, clf))
        out[name] = fits
    return out


def predict_observed(fits, obs: Dict[str, np.ndarray], feature_sets=FEATURE_SETS) -> Dict[str, np.ndarray]:
    """P(slowdown) for every year and split-fit → {set: (n_fits, nyear)}; NaN where an input is missing."""
    out = {}
    for name, feats in feature_sets.items():
        X = np.column_stack([obs[f] for f in feats]); ok = np.isfinite(X).all(1)
        P = np.full((len(fits[name]), X.shape[0]), np.nan)
        for i, (mu, sd, clf) in enumerate(fits[name]):
            P[i, ok] = clf.predict_proba((X[ok] - mu) / sd)[:, 1]
        out[name] = P
    return out


def observed_offset_z(years: np.ndarray, forced_method: str, nsidc_events_file: Path, metrics_dir: Path,
                      labels_file: Path, window: int = 10, offset: int = 1) -> np.ndarray:
    """
    Observed trend anomaly z at onset t: OLS slope of NSIDC September SIE over t+offset … t+offset+window−1
    minus the same slope of the (mean- and trend-corrected) forced reference, over the model's pooled σ.
    NaN where the window is not yet observed.
    """
    with xr.open_dataset(nsidc_events_file) as ds:
        oy, osie = ds["yearice"].values.astype(int), ds["seaice"].values.astype(float)
    with xr.open_dataset(labels_file) as ds:
        sigma = float(np.nanmean(ds["sigma"].values))
    if forced_method == "linear":
        okf = np.isfinite(osie); ref = np.full(oy.size, np.nan)
        ref[okf] = np.polyval(np.polyfit(oy[okf] - oy.mean(), osie[okf], 1), oy[okf] - oy.mean())
    else:
        m_sie, m_yrs = load_sie_monthly_files(str(metrics_dir), "SEP", variable="sie", start_year=1990, end_year=2100)
        sl = oi.GROUP_SLICES[forced_method.split("_", 1)[1]] if forced_method.startswith("group_") else slice(0, 100)
        sel = np.isin(oy, m_yrs); forced = m_sie[sl].mean(0)[np.isin(m_yrs, oy[sel])]
        ref = np.full(oy.size, np.nan)
        ref[sel], _, _ = _correct_forced_mean_and_trend(osie[sel], forced, np.arange(sel.sum(), dtype=float))
    z = np.full(years.size, np.nan); x = np.arange(window) - (window - 1) / 2
    for i, t in enumerate(years):
        w = np.arange(t + offset, t + offset + window)
        if w[-1] > oy[-1] or w[0] < oy[0]:
            continue
        idx = np.searchsorted(oy, w)
        if not (np.isfinite(ref[idx]).all() and np.isfinite(osie[idx]).all()):
            continue
        z[i] = ((x @ osie[idx]) - (x @ ref[idx])) / (x @ x) / sigma
    return z


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


def summary_markdown(years, obs, probs, z, frac, forced_method, claim=(2016, 2025)) -> str:
    c0, c1 = claim; sel = (years >= c0) & (years <= c1)
    lines = [f"# Observed baseline predictions — forced reference `{forced_method}`\n",
             "P(slowdown in t+1…t+10) from logistic fits on CESM2-LE (median over 9 split-fits); observed offset z where the decade is complete.\n",
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
                 + f"; base rate 0.17.\n")
    return "\n".join(lines) + "\n"
