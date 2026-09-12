"""
obs_products.py — compare two observational SST products on the CESM2-LE grid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from src.plotting.forced import arctic_mean


def load_product(path: Path, name: str) -> Dict:
    """Monthly regridded product → dict(sst (nt, nx, ny), years, months, ice or None, name)."""
    with xr.open_dataset(path) as ds:
        sst = ds["sst_obs"].values.astype(np.float32)
        if "time" in ds.coords:
            t = pd.DatetimeIndex(ds["time"].values)
        else:                                            # ERSST file: infer from its date_range attr
            y0, m0 = map(int, ds.attrs["date_range"].split(" to ")[0].split("-"))
            t = pd.date_range(f"{y0}-{m0:02d}-01", periods=sst.shape[0], freq="MS")
        ice = ds["ice_obs"].values.astype(np.float32) if "ice_obs" in ds else None
    return dict(name=name, sst=sst, years=t.year.values, months=t.month.values, ice=ice)


def jja_mean(prod: Dict, y0: int, y1: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """JJA means per year → (years, sst (ny, nx, ny), ice or None)."""
    yrs = np.arange(y0, y1 + 1)
    sel = lambda y: (prod["years"] == y) & np.isin(prod["months"], (6, 7, 8))
    sst = np.stack([np.nanmean(prod["sst"][sel(y)], axis=0) for y in yrs])
    ice = None if prod["ice"] is None else np.stack([np.nanmean(prod["ice"][sel(y)], axis=0) for y in yrs])
    return yrs, sst, ice


def _trend(x: np.ndarray, years: np.ndarray) -> np.ndarray:
    """Least-squares trend per decade along axis 0, NaN-tolerant per cell."""
    t = years - years.mean()
    xm = np.nanmean(x, axis=0)
    num = np.nansum((x - xm) * t[:, None, None], axis=0)
    den = np.sum(t ** 2)
    return 10.0 * num / den


def compare_jja(a: Dict, b: Dict, lat: np.ndarray, y0: int, y1: int, landmask=None,
                clim: Tuple[int, int] = (1991, 2020)) -> Dict:
    """Everything the comparison figure needs, for the overlap years of the two products."""
    y0 = max(y0, int(a["years"].min()), int(b["years"].min()))
    y1 = min(y1, int(a["years"].max()), int(b["years"].max()))
    yrs, sa, _ = jja_mean(a, y0, y1)
    _, sb, ice_b = jja_mean(b, y0, y1)
    if landmask is not None:
        sa = np.where(landmask == 1, np.nan, sa); sb = np.where(landmask == 1, np.nan, sb)
    csel = (yrs >= clim[0]) & (yrs <= clim[1])
    arctic = lat >= 65
    ocean_n = (landmask[arctic] == 0) if landmask is not None else np.ones((arctic.sum(), sa.shape[2]), bool)
    cov = lambda s: np.array([np.isfinite(x[arctic][ocean_n]).mean() for x in s])
    return dict(
        names=(a["name"], b["name"]), years=yrs, clim=clim,
        arctic_a=arctic_mean(sa, lat), arctic_b=arctic_mean(sb, lat),
        clim_diff=np.nanmean(sb[csel] - sa[csel], axis=0),
        trend_diff=_trend(sb, yrs) - _trend(sa, yrs),
        trend_a=_trend(sa, yrs), trend_b=_trend(sb, yrs),
        coverage_a=cov(sa), coverage_b=cov(sb),
        arctic_ice_b=None if ice_b is None else arctic_mean(np.where(landmask == 1, np.nan, ice_b) if landmask is not None else ice_b, lat),
    )


def summary_table(c: Dict) -> str:
    na, nb = c["names"]; lat_note = "Arctic (>65°N) JJA"
    d = c["arctic_b"] - c["arctic_a"]
    lines = [f"  {lat_note}: {nb} − {na}",
             f"    mean difference {c['clim'][0]}–{c['clim'][1]}: {np.nanmean(d[(c['years'] >= c['clim'][0]) & (c['years'] <= c['clim'][1])]):+.3f} °C",
             f"    trend {c['years'][0]}–{c['years'][-1]}: {na} {10*np.polyfit(c['years'], c['arctic_a'], 1)[0]:+.3f}, "
             f"{nb} {10*np.polyfit(c['years'], c['arctic_b'], 1)[0]:+.3f} °C/decade",
             f"    correlation of detrended series: {np.corrcoef(np.polyval(np.polyfit(c['years'], c['arctic_a'], 1), c['years']) - c['arctic_a'], np.polyval(np.polyfit(c['years'], c['arctic_b'], 1), c['years']) - c['arctic_b'])[0, 1]:.2f}",
             f"    valid-cell coverage north of 65°N: {na} {c['coverage_a'].mean():.2f}, {nb} {c['coverage_b'].mean():.2f}"]
    return "\n".join(lines)
