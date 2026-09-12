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
                           start_year: int = 1990, **kwargs) -> xr.Dataset:
    """Trends + relative labels as a Dataset with the original file's variable names."""
    trends_ens, trends_mean, trend_years = compute_decadal_trends_ensemble(
        sie, years, window=window, start_year=start_year)
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


# =============================================================================
# Diagnostic figure
# =============================================================================

def plot_relative_labels(ds: xr.Dataset, sie: np.ndarray, years: np.ndarray,
                         out_png, original: xr.Dataset = None, member: int = 7,
                         pool_years: Tuple[int, int] = (1990, 2040)) -> None:
    """
    Three-panel diagnostic: group-mean SIE with one member's slowdowns,
    standardised trend anomaly z with ±n_sigma, and slowdown frequency by
    onset year (original vs relative, per forcing group).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#1f2933", "#8a949e", "#e5e8eb"
    c_all, c_cmip6, c_smbb, c_orig = "#0072B2", "#D55E00", "#009E73", "#999999"
    n_sigma, window = float(ds.attrs["n_sigma"]), int(ds.attrs["window"])
    tyrs = ds["nyr"].values
    lab, z = ds["slowdown"].values, ds["z"].values
    sel = (tyrs >= pool_years[0]) & (tyrs <= pool_years[1])

    fig, axes = plt.subplots(3, 1, figsize=(9, 10.5))

    # (a) SIE: members grey, group means, highlighted member + its slowdown windows
    ax = axes[0]
    ax.plot(years, sie.T, color=grid, lw=0.5, zorder=1)
    if sie.shape[0] == 100:
        ax.plot(years, sie[:50].mean(0), color=c_cmip6, lw=1.8, label="CMIP6-BB mean (0–49)")
        ax.plot(years, sie[50:].mean(0), color=c_smbb, lw=1.8, label="SMBB mean (50–99)")
    ax.plot(years, sie.mean(0), color=ink, lw=1.2, ls="--", label="100-member mean")
    ax.plot(years, sie[member], color=c_all, lw=1.2, label=f"member {member}")
    for j, y0 in enumerate(tyrs):
        if lab[member, j]:
            i0 = int(np.where(years == y0)[0][0])
            yy = sie[member, i0:i0 + window]
            xx = years[i0:i0 + window]
            b = np.polyfit(xx, yy, 1)
            ax.plot(xx, np.polyval(b, xx), color="#E69F00", lw=2.2, zorder=5)
    ax.set_xlim(years[0], min(years[-1], 2060))
    ax.set_ylabel("September SIE [M km²]")
    ax.set_title(f"(a) SIE and relative slowdown windows for member {member} "
                 f"(orange = {window}-yr trends flagged as slowdown)", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, ncol=2)

    # (b) z for all members, highlighted member, ±n_sigma
    ax = axes[1]
    ax.plot(tyrs, z.T, color=grid, lw=0.5, zorder=1)
    ax.plot(tyrs, z[member], color=c_all, lw=1.2, zorder=3)
    ax.scatter(tyrs[lab[member] == 1], z[member][lab[member] == 1], s=22, color="#E69F00", zorder=5)
    ax.axhline(n_sigma, color=ink, ls="--", lw=1); ax.axhline(-n_sigma, color=ink, ls="--", lw=1)
    ax.axhline(0, color=muted, lw=0.8)
    ax.axvspan(pool_years[0], pool_years[1], color=grid, alpha=0.35, zorder=0, label="σ pooling / analysis period")
    ax.set_xlim(tyrs[0], min(tyrs[-1], 2060))
    ax.set_ylabel("trend anomaly z = (trend − group mean) / σ")
    ax.set_title(f"(b) standardised trend anomaly, all members (σ_pool = {float(ds['sigma'][0]):.3f} M km² yr⁻¹, "
                 f"threshold ±{n_sigma:g}σ)", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    # (c) frequency by onset year
    ax = axes[2]
    if original is not None:
        o = original["slowdown"].sel(nyr=slice(tyrs[0], tyrs[-1]))
        ax.plot(o["nyr"].values, o.values.mean(0), color=c_orig, lw=2.2, label="original labels (all members)")
    freq = frequency_by_year(lab, tyrs)
    ax.plot(tyrs, freq["all"], color=c_all, lw=2, label="relative — all")
    if "cmip6" in freq:
        ax.plot(tyrs, freq["cmip6"], color=c_cmip6, lw=1.4, label="relative — CMIP6-BB (0–49)")
        ax.plot(tyrs, freq["smbb"], color=c_smbb, lw=1.4, label="relative — SMBB (50–99)")
    ax.axhline(freq["all"][sel].mean(), color=c_all, ls=":", lw=1)
    ax.axvspan(pool_years[0], pool_years[1], color=grid, alpha=0.35, zorder=0)
    ax.set_xlim(tyrs[0], min(tyrs[-1], 2060)); ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of members flagged slowdown")
    ax.set_xlabel("onset year of trend window")
    ax.set_title("(c) slowdown frequency by onset year — a flat line means the labels carry no forced epoch",
                 fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, ncol=2)

    for ax in axes:
        ax.grid(axis="y", color=grid, lw=0.8, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle(f"Relative slowdown labels — window {window} yr, {n_sigma:g}σ, demean={ds.attrs['demean']}",
                 fontsize=11)
    fig.tight_layout()
    from pathlib import Path
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"  figure → {out_png}")


def plot_window_sweep(datasets: Dict[int, xr.Dataset], out_png,
                      pool_years: Tuple[int, int] = (1990, 2040)) -> None:
    """Slowdown frequency by onset year for several window lengths (one line each)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    cols = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for c, (w, ds) in zip(cols, sorted(datasets.items())):
        tyrs = ds["nyr"].values
        ax.plot(tyrs, ds["slowdown"].values.mean(0), color=c, lw=1.6, label=f"{w}-yr window")
    ax.axvspan(pool_years[0], pool_years[1], color="#e5e8eb", alpha=0.35, zorder=0)
    ax.set_xlim(min(d["nyr"].values[0] for d in datasets.values()), 2060); ax.set_ylim(0, 0.6)
    ax.set_ylabel("fraction of members flagged"); ax.set_xlabel("onset year")
    ax.set_title("Relative slowdown frequency by onset year — window sweep", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, ncol=3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200); plt.close(fig)
    print(f"  figure → {out_png}")
