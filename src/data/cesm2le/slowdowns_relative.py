"""
slowdowns_relative.py — epoch-free slowdown labels for CESM2-LE.

The original definition scales an observed threshold by the ensemble-mean
trend, so it degenerates wherever the forced trend flattens (≈2005–2020) and
the label base rate becomes strongly year-dependent. Here a slowdown is an
anomaly of the member's decadal trend relative to its forcing group's mean
trend, standardised by the pooled spread over a reference period:

    z(m, t) = (trend(m, t) − trend_group_mean(t)) / σ_pool
    slowdown(m, t) = z > +n_sigma          (RILES: z < −n_sigma)

Demeaning is done per forcing group by default (members 1–50 CMIP6 BB,
51–100 SMBB) so the biomass-burning forcing artifact does not leak between
groups.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import xarray as xr

from .slowdowns import compute_decadal_trends_ensemble

GROUPS = {"cmip6": slice(0, 50), "smbb": slice(50, 100)}


def group_mean_trends(trends_ens: np.ndarray, demean: str = "group") -> np.ndarray:
    """Reference trend per member: group mean (default) or full-ensemble mean, (nens, nyr)."""
    if demean == "all":
        return np.broadcast_to(np.nanmean(trends_ens, axis=0), trends_ens.shape).copy()
    if demean != "group":
        raise ValueError("demean must be 'group' or 'all'")
    if trends_ens.shape[0] != 100:
        raise ValueError("group demeaning assumes 100 members (50 CMIP6 + 50 SMBB)")
    ref = np.empty_like(trends_ens)
    for sl in GROUPS.values():
        ref[sl] = np.nanmean(trends_ens[sl], axis=0)
    return ref


def relative_labels(trends_ens: np.ndarray, trend_years: np.ndarray,
                    n_sigma: float = 1.0, demean: str = "group",
                    pool_years: Tuple[int, int] = (1990, 2040),
                    sigma_mode: str = "pooled") -> Dict[str, np.ndarray]:
    """
    Classify decadal-trend anomalies as slowdown / RILES.

    Args:
        trends_ens: per-member trends (nens, nyr), M km² yr⁻¹.
        trend_years: onset year of each window (nyr,).
        n_sigma: threshold in standard deviations.
        demean: 'group' (per forcing group) or 'all'.
        pool_years: onset-year range used to estimate σ.
        sigma_mode: 'pooled' (one σ) or 'yearly' (σ per onset year, smoothed).
    """
    ref = group_mean_trends(trends_ens, demean)
    anom = trends_ens - ref
    sel = (trend_years >= pool_years[0]) & (trend_years <= pool_years[1])
    if sigma_mode == "pooled":
        sigma = np.full(trend_years.size, np.nanstd(anom[:, sel]))
    elif sigma_mode == "yearly":
        s = np.nanstd(anom, axis=0)
        k = 5
        sigma = np.convolve(np.pad(s, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
    else:
        raise ValueError("sigma_mode must be 'pooled' or 'yearly'")
    z = anom / sigma[None, :]
    return {"trend_anom": anom, "z": z, "sigma": sigma, "reference_trend": ref,
            "slowdown": (z > n_sigma).astype(np.int8),
            "riles": (z < -n_sigma).astype(np.int8)}


def build_relative_dataset(sie: np.ndarray, years: np.ndarray, window: int = 10,
                           start_year: int = 1990, trend_offset: int = 0, **kwargs) -> xr.Dataset:
    """
    Trends + relative labels as a Dataset with the original file's variable names.

    ``trend_offset`` k labels onset year t with the trend of the window t+k … t+k+window−1,
    so predictors at t are strictly before the target (k = 0 is the LB22 convention, where
    the onset value sits inside the fitted window and correlates with the slope by construction).
    """
    trends_ens, trends_mean, trend_years = compute_decadal_trends_ensemble(
        sie, years, window=window, start_year=start_year)
    if trend_offset:
        trends_ens, trends_mean = trends_ens[:, trend_offset:], trends_mean[trend_offset:]
        trend_years = trend_years[trend_offset:] - trend_offset          # onset year = window start − k
    lab = relative_labels(trends_ens, trend_years, **kwargs)
    ds = xr.Dataset(
        {
            "slowdown":           (("nens", "nyr"), lab["slowdown"]),
            "riles":              (("nens", "nyr"), lab["riles"]),
            "linear_trends_ens":  (("nens", "nyr"), trends_ens),
            "linear_trends_mean": (("nyr",), trends_mean),
            "reference_trend":    (("nens", "nyr"), lab["reference_trend"]),
            "trend_anom":         (("nens", "nyr"), lab["trend_anom"]),
            "z":                  (("nens", "nyr"), lab["z"]),
            "sigma":              (("nyr",), lab["sigma"]),
            "threshold_slowdown": (("nyr",), kwargs.get("n_sigma", 1.0) * lab["sigma"]),
        },
        coords={"nens": np.arange(sie.shape[0]), "nyr": trend_years},
    )
    ds.attrs.update({
        "description": "CESM2-LE slowdown labels relative to the forcing-group mean trend",
        "threshold_method": "z = (trend − group_mean_trend) / sigma_pool; slowdown: z > n_sigma",
        "n_sigma": float(kwargs.get("n_sigma", 1.0)),
        "demean": kwargs.get("demean", "group"),
        "sigma_mode": kwargs.get("sigma_mode", "pooled"),
        "pool_years": str(kwargs.get("pool_years", (1990, 2040))),
        "window": window,
        "trend_offset": int(trend_offset),
        "groups": "cmip6: members 0-49, smbb: members 50-99",
    })
    ds["slowdown"].attrs["description"] = "1 = slowdown (trend anomaly > +n_sigma), 0 = normal"
    ds["riles"].attrs["description"] = "1 = rapid ice loss event (trend anomaly < −n_sigma)"
    for v in ("linear_trends_ens", "linear_trends_mean", "reference_trend",
              "trend_anom", "sigma", "threshold_slowdown"):
        ds[v].attrs["units"] = "M km2 yr-1"
    return ds


def frequency_by_year(labels: np.ndarray, years: np.ndarray) -> Dict[str, np.ndarray]:
    """Slowdown frequency per onset year: all members and per forcing group."""
    out = {"all": labels.mean(0)}
    if labels.shape[0] == 100:
        for g, sl in GROUPS.items():
            out[g] = labels[sl].mean(0)
    return out


def frequency_table(labels: np.ndarray, years: np.ndarray, step: int = 10) -> str:
    """Compact text table of slowdown frequency by decade and forcing group."""
    freq = frequency_by_year(labels, years)
    cols = list(freq)
    lines = ["  decade     " + "".join(f"{c:>8s}" for c in cols)]
    for y0 in range(int(years.min()), int(years.max()) + 1, step):
        sel = (years >= y0) & (years < y0 + step)
        if sel.any():
            lines.append(f"  {y0}–{min(y0 + step - 1, int(years.max()))}  "
                         + "".join(f"{freq[c][sel].mean():8.2f}" for c in cols))
    lines.append("  overall    " + "".join(f"{freq[c].mean():8.2f}" for c in cols))
    return "\n".join(lines)
