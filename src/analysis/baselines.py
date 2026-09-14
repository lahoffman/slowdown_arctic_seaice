"""
baselines.py — scalar baselines for slowdown classification.

Every baseline is fit on the training members of a TVT split and scored on
the test members, using the same block assignment as the CNN
(``src.cnn.splits._get_block_indices`` / ``block_tvt_split``). The point is
to measure what the CNN adds beyond (a) the class prior, (b) the onset-year
label climatology, (c) the September SIE anomaly at onset, and (d) the
scalar climate indices already used in the paper.

TensorFlow is deliberately not imported here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, f1_score,
    log_loss, matthews_corrcoef, precision_recall_curve, precision_score,
    recall_score, roc_auc_score,
)

from src.cnn.splits import _get_block_indices, block_tvt_split
from src.data.cesm2le.slowdowns import load_sie_monthly_files
from src.data.cesm2le.slowdowns_relative import group_mean_trends

METRIC_NAMES = ["AUPRC", "AUROC", "Brier", "Precision", "Recall", "F1",
                "Accuracy", "MCC", "Threshold", "Prevalence"]

#: Baseline name → list of feature keys. ``[]`` means no features (prior only).
BASELINE_FEATURES: Dict[str, List[str]] = {
    "always_positive":      [],
    "random_prevalence":    [],
    "year_climatology":     ["yearclim"],
    # single predictors
    "logit_sie_anom":       ["sie_anom"],
    "logit_arctic_sst":     ["arctic"],
    "logit_nino34":         ["nino34"],
    "logit_ipo":            ["ipo"],
    # index combinations (no ice state)
    "logit_pacific":        ["nino34", "ipo"],
    "logit_indices":        ["arctic", "nino34", "ipo"],
    # what each adds on top of the ice state
    "logit_sie_arctic":     ["sie_anom", "arctic"],
    "logit_sie_nino34":     ["sie_anom", "nino34"],
    "logit_sie_ipo":        ["sie_anom", "ipo"],
    "logit_sie_pacific":    ["sie_anom", "nino34", "ipo"],
    "logit_sie_year":       ["sie_anom", "yearclim"],
    "logit_all_scalars":    ["sie_anom", "arctic", "nino34", "ipo", "yearclim"],
    # sea-ice volume (step 8.5; skipped automatically when sivoln_* files are absent)
    "logit_siv":            ["siv_anom"],
    "logit_sie_siv":        ["sie_anom", "siv_anom"],
    "logit_sie_siv_pacific": ["sie_anom", "siv_anom", "nino34", "ipo"],
}


# =============================================================================
# Loading
# =============================================================================

def load_labels(slowdown_file: Path, start_year: int, end_year: int
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Binary slowdown labels (nens, nyear) and onset years from a slowdown file."""
    with xr.open_dataset(slowdown_file) as ds:
        sub = ds["slowdown"].sel(nyr=slice(start_year, end_year))
        return sub.values.astype(np.int8), sub["nyr"].values.astype(int)


def load_sie_anomaly(metrics_dir: Path, years: np.ndarray, month: str = "SEP",
                     variable: str = "sie", demean: str = "all"
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    September SIE at onset year and its anomaly from the forced response.

    ``demean='all'`` subtracts the 100-member mean (default, as in the Fig. S3
    baselines); ``'group'`` subtracts the forcing-group mean (matches the
    relative labels and the ``--demean group`` SST preprocessing).
    Returns (sie, sie_anom), each (nens, nyear) aligned to ``years``.
    """
    sie_all, yrs_all = load_sie_monthly_files(str(metrics_dir), month,
                                              variable=variable,
                                              start_year=1990, end_year=2100)
    idx = np.searchsorted(yrs_all, years)
    if not np.array_equal(yrs_all[idx], years):
        raise ValueError("SIE years do not cover the requested onset years.")
    sie = sie_all[:, idx]
    return sie, sie - group_mean_trends(sie, demean)


def load_siv_anomaly(metrics_dir: Path, years: np.ndarray, month: str = "SEP", demean: str = "group"):
    """Sea-ice volume anomaly (10³ km³) at onset, or None if the sivoln_* files are missing."""
    try:
        _, siv_anom = load_sie_anomaly(metrics_dir, years, month=month, variable="siv", demean=demean)
    except FileNotFoundError:
        return None
    return siv_anom


def load_climate_indices_jja(indices_dir: Path, start_year: int, end_year: int
                             ) -> Dict[str, np.ndarray]:
    """JJA-mean Niño3.4, IPO and Arctic SST indices, each (nens, nyear)."""
    specs = [("nino34", "cesm2le_nino34_index.nc",     "nino34_months"),
             ("ipo",    "cesm2le_ipo_index.nc",        "ipo_filtered"),
             ("arctic", "cesm2le_arctic_sst_index.nc", "arctic_sst_months")]
    out = {}
    for key, fname, var in specs:
        fpath = indices_dir / fname
        if not fpath.exists():
            print(f"  [skip] {fname} not found — {key} omitted")
            continue
        with xr.open_dataset(fpath) as ds:
            arr = ds[var].sel(nyr=slice(start_year, end_year)).values
        out[key] = np.nanmean(arr[:, :, 5:8], axis=2)
    return out


def load_cnn_test_predictions(pred_dir: Path, split_idx: int
                              ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Cached CNN test predictions for one split: [(y_true, y_prob, thr), ...]."""
    out = []
    for f in sorted(pred_dir.glob(f"cnn_prediction_cesm2le_M{split_idx}_*.nc")):
        with xr.open_dataset(f) as ds:
            out.append((ds["y_true_test"].values.astype(int),
                        ds["y_prob_test"].values.astype(float),
                        float(ds["threshold"].values)))
    return out


# =============================================================================
# Metrics
# =============================================================================

def pr_intersection_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Threshold where precision and recall are closest (same rule as the CNN)."""
    if np.ptp(y_score) == 0:
        return 0.5
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    if thr.size == 0:
        return 0.5
    return float(thr[np.argmin(np.abs(prec[:-1] - rec[:-1]))])


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float
                    ) -> Dict[str, float]:
    """Binary-classification metrics at a fixed, externally chosen threshold."""
    y_pred = (y_score >= threshold).astype(int)
    const = np.ptp(y_score) == 0
    p = np.clip(y_score, 1e-6, 1 - 1e-6)
    return {
        "AUPRC":      float(np.mean(y_true)) if const else float(average_precision_score(y_true, y_score)),
        "AUROC":      0.5 if const else float(roc_auc_score(y_true, y_score)),
        "Brier":      float(brier_score_loss(y_true, p)),
        "Precision":  float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall":     float(recall_score(y_true, y_pred, zero_division=0)),
        "F1":         float(f1_score(y_true, y_pred, zero_division=0)),
        "Accuracy":   float(accuracy_score(y_true, y_pred)),
        "MCC":        float(matthews_corrcoef(y_true, y_pred)),
        "Threshold":  float(threshold),
        "Prevalence": float(np.mean(y_true)),
    }


def member_bootstrap_f1(y_true: np.ndarray, y_score: np.ndarray, threshold: float,
                        member_id: np.ndarray, n_boot: int = 1000,
                        seed: int = 0) -> Tuple[float, float]:
    """95% CI on F1 from resampling whole test members (block bootstrap)."""
    rng = np.random.default_rng(seed)
    members = np.unique(member_id)
    groups = {m: np.where(member_id == m)[0] for m in members}
    y_pred = (y_score >= threshold).astype(int)
    f1s = []
    for _ in range(n_boot):
        pick = rng.choice(members, size=members.size, replace=True)
        idx = np.concatenate([groups[m] for m in pick])
        f1s.append(f1_score(y_true[idx], y_pred[idx], zero_division=0))
    return tuple(np.percentile(f1s, [2.5, 97.5]))


def random_prevalence_f1(y_true: np.ndarray, n_draws: int = 1000, seed: int = 0
                         ) -> Dict[str, float]:
    """F1 of a classifier that guesses positives at the training prevalence."""
    rng = np.random.default_rng(seed)
    p = y_true.mean()
    f1s = [f1_score(y_true, rng.random(y_true.size) < p, zero_division=0)
           for _ in range(n_draws)]
    return {"mean": float(np.mean(f1s)), "p95": float(np.percentile(f1s, 95)),
            "p99": float(np.percentile(f1s, 99))}


# =============================================================================
# Feature construction and fitting
# =============================================================================

def year_climatology(labels_tr: np.ndarray, years_tr: np.ndarray,
                     years: np.ndarray) -> np.ndarray:
    """P(slowdown | onset year) from training samples, evaluated at ``years``."""
    clim = {y: labels_tr[years_tr == y].mean() for y in np.unique(years_tr)}
    return np.array([clim.get(y, labels_tr.mean()) for y in years])


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def fit_logistic(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray,
                 C: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardise on train, fit balanced logistic regression, return scores + coefs."""
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-12
    clf = LogisticRegression(class_weight="balanced", C=C, max_iter=2000)
    clf.fit((X_tr - mu) / sd, y_tr)
    return (clf.predict_proba((X_tr - mu) / sd)[:, 1],
            clf.predict_proba((X_te - mu) / sd)[:, 1],
            clf.coef_.ravel())


def split_scalars(arrays: Dict[str, np.ndarray], years: np.ndarray,
                  train_blocks, val_block, test_block, n_blocks=10, block_size=10
                  ) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Apply the CNN block split to every (nens, nyear) scalar field.

    Returns ``{'tr': {...}, 'te': {...}}`` with flattened 1-D arrays, plus
    ``'year'`` and ``'member'`` keys giving the onset year and member index.
    """
    nens = next(iter(arrays.values())).shape[0]
    aug = dict(arrays)
    aug["year"] = np.broadcast_to(years, (nens, years.size)).astype(float)
    aug["member"] = np.broadcast_to(np.arange(nens)[:, None], (nens, years.size)).astype(float)
    out = {"tr": {}, "te": {}}
    for k, arr in aug.items():
        tr, _, te = block_tvt_split(np.asarray(arr, float), train_blocks,
                                    val_block, test_block, n_blocks, block_size)
        out["tr"][k], out["te"][k] = tr, te
    return out


def run_split(fields: Dict[str, np.ndarray], labels: np.ndarray, years: np.ndarray,
              split_idx: int, test_block: int, val_block: int, train_blocks,
              n_boot: int = 1000, cnn_preds=None) -> Tuple[xr.Dataset, dict]:
    """
    Fit and score all baselines (and cached CNN runs) for one TVT split.

    Returns a Dataset (metric × model) and a dict of extra diagnostics
    (coefficients, random-F1 quantiles, bootstrap CIs).
    """
    data = split_scalars({**fields, "label": labels}, years,
                         train_blocks, val_block, test_block)
    y_tr, y_te = data["tr"]["label"].astype(int), data["te"]["label"].astype(int)
    yr_tr, yr_te = data["tr"]["year"].astype(int), data["te"]["year"].astype(int)
    mem_te = data["te"]["member"].astype(int)

    # year climatology as a feature (logit of training P(slowdown | year))
    data["tr"]["yearclim"] = _logit(year_climatology(y_tr, yr_tr, yr_tr))
    data["te"]["yearclim"] = _logit(year_climatology(y_tr, yr_tr, yr_te))

    results, extras = {}, {"coefs": {}, "f1_ci": {}}

    # --- reference baselines ------------------------------------------------
    ones = np.ones_like(y_te, float)
    results["always_positive"] = compute_metrics(y_te, ones, 0.5)
    extras["random_f1"] = random_prevalence_f1(y_te, n_draws=n_boot, seed=split_idx)
    rnd = dict(results["always_positive"])
    rnd.update({"F1": extras["random_f1"]["mean"], "Precision": float(y_te.mean()),
                "Recall": float(y_tr.mean()), "MCC": 0.0,
                "Accuracy": float(y_te.mean() * y_tr.mean() + (1 - y_te.mean()) * (1 - y_tr.mean()))})
    results["random_prevalence"] = rnd

    p_tr = year_climatology(y_tr, yr_tr, yr_tr)
    p_te = year_climatology(y_tr, yr_tr, yr_te)
    thr = pr_intersection_threshold(y_tr, p_tr)
    results["year_climatology"] = compute_metrics(y_te, p_te, thr)
    extras["f1_ci"]["year_climatology"] = member_bootstrap_f1(y_te, p_te, thr, mem_te, n_boot)

    # --- logistic baselines -------------------------------------------------
    for name, feats in BASELINE_FEATURES.items():
        if not name.startswith("logit_"):
            continue
        if any(f not in data["tr"] for f in feats):
            print(f"  [skip] {name}: missing features {feats}")
            continue
        X_tr = np.column_stack([data["tr"][f] for f in feats])
        X_te = np.column_stack([data["te"][f] for f in feats])
        ok_tr, ok_te = np.isfinite(X_tr).all(1), np.isfinite(X_te).all(1)
        s_tr, s_te, coef = fit_logistic(X_tr[ok_tr], y_tr[ok_tr], X_te[ok_te])
        thr = pr_intersection_threshold(y_tr[ok_tr], s_tr)
        results[name] = compute_metrics(y_te[ok_te], s_te, thr)
        extras["coefs"][name] = dict(zip(feats, coef.round(3).tolist()))
        extras["f1_ci"][name] = member_bootstrap_f1(y_te[ok_te], s_te, thr, mem_te[ok_te], n_boot)

    # --- cached CNN runs ----------------------------------------------------
    # The cached files carry the labels the CNN was trained on. If the label
    # set passed here differs (e.g. relative labels), the CNN is scored against
    # the *new* labels — a transfer test, flagged in the dataset attrs.
    if cnn_preds and len(cnn_preds[0][0]) == len(y_te):
        same_labels = all(np.array_equal(yt, y_te) for yt, _, _ in cnn_preds)
        per_run = []
        for r, (yt, yp, thr) in enumerate(cnn_preds):
            m = compute_metrics(y_te, yp, thr)
            results[f"cnn_run{r}"] = m
            per_run.append(m)
        results["cnn_median"] = {k: float(np.median([m[k] for m in per_run]))
                                 for k in METRIC_NAMES}
        yp0, thr0 = cnn_preds[0][1], cnn_preds[0][2]
        extras["f1_ci"]["cnn_run0"] = member_bootstrap_f1(y_te, yp0, thr0, mem_te, n_boot)
        extras["cnn_labels_match"] = bool(same_labels)
        yp_mean = np.mean([yp for _, yp, _ in cnn_preds], axis=0)
        extras["cnn_attribution"] = cnn_output_attribution(yp_mean, p_te, data["te"]["sie_anom"])
    elif cnn_preds:
        print(f"  [skip] CNN predictions have {len(cnn_preds[0][0])} test samples, "
              f"labels have {len(y_te)}")

    models = list(results)
    vals = np.array([[results[m][k] for m in models] for k in METRIC_NAMES], float)
    ds = xr.Dataset({"metric_value": (("metric", "model"), vals)},
                    coords={"metric": METRIC_NAMES, "model": models},
                    attrs={"split_idx": split_idx, "test_block": test_block,
                           "val_block": val_block, "n_test": int(y_te.size),
                           "n_train": int(y_tr.size),
                           "always_positive_f1_analytic": float(2 * y_te.mean() / (1 + y_te.mean())),
                           "random_f1_p95": extras["random_f1"]["p95"],
                           "cnn_labels_match": int(extras.get("cnn_labels_match", -1))})
    lo = np.array([extras["f1_ci"].get(m, (np.nan, np.nan))[0] for m in models])
    hi = np.array([extras["f1_ci"].get(m, (np.nan, np.nan))[1] for m in models])
    ds["f1_ci_low"] = ("model", lo)
    ds["f1_ci_high"] = ("model", hi)
    return ds, extras


def cnn_output_attribution(y_prob: np.ndarray, yearclim_p: np.ndarray,
                           sie_anom: np.ndarray) -> Dict[str, float]:
    """
    R² of CNN test probabilities explained by year climatology, SIE anomaly,
    and both (OLS). Also the partial R² each adds on top of the other.
    """
    def r2(X):
        X1 = np.column_stack([np.ones(len(y_prob)), X])
        beta, *_ = np.linalg.lstsq(X1, y_prob, rcond=None)
        res = y_prob - X1 @ beta
        return 1 - res.var() / y_prob.var()
    r_year, r_sie = r2(yearclim_p), r2(sie_anom)
    r_both = r2(np.column_stack([yearclim_p, sie_anom]))
    return {"r2_yearclim": round(float(r_year), 3), "r2_sie_anom": round(float(r_sie), 3),
            "r2_both": round(float(r_both), 3),
            "partial_r2_sie_given_year": round(float(r_both - r_year), 3),
            "partial_r2_year_given_sie": round(float(r_both - r_sie), 3),
            "corr_prob_yearclim": round(float(np.corrcoef(y_prob, yearclim_p)[0, 1]), 3),
            "corr_prob_sie_anom": round(float(np.corrcoef(y_prob, sie_anom)[0, 1]), 3)}


def attribution_markdown(attrib: Dict[str, Dict[str, float]]) -> str:
    """Median across splits of the CNN-output attribution diagnostics."""
    keys = list(next(iter(attrib.values())))
    lines = ["CNN test probability explained by (median across splits):"]
    for k in keys:
        lines.append(f"  {k:28s} {np.median([a[k] for a in attrib.values()]):6.3f}")
    return "\n".join(lines)


def iter_split_blocks(n_splits: int = 9, n_blocks: int = 10):
    """Yield (split_idx, test_block, val_block, train_blocks) as used by the CNN."""
    yield from _get_block_indices(n_splits, n_blocks)


def summarise(datasets: List[xr.Dataset]) -> xr.Dataset:
    """Stack per-split datasets and add the across-split median."""
    common = sorted(set.intersection(*[set(d.model.values) for d in datasets]))
    stacked = xr.concat([d.sel(model=common) for d in datasets], dim="split")
    stacked["split"] = [d.attrs["split_idx"] for d in datasets]
    stacked["metric_median"] = stacked["metric_value"].median("split")
    return stacked


def summary_markdown(stacked: xr.Dataset,
                     metrics=("F1", "AUPRC", "AUROC", "Accuracy", "MCC")) -> str:
    """Markdown table of across-split medians (and F1 range) per model."""
    lines = ["| model | " + " | ".join(metrics) + " | F1 range |",
             "|---|" + "---|" * (len(metrics) + 1)]
    present = list(stacked.model.values)
    order = [m for m in BASELINE_FEATURES if m in present] + \
            sorted(m for m in present if m not in BASELINE_FEATURES)
    for m in order:
        row = [f"{float(stacked['metric_median'].sel(metric=k, model=m)):.3f}" for k in metrics]
        f1 = stacked["metric_value"].sel(metric="F1", model=m).values
        lines.append(f"| {m} | " + " | ".join(row) + f" | {f1.min():.2f}–{f1.max():.2f} |")
    return "\n".join(lines)
