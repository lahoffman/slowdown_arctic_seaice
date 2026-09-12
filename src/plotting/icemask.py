"""
icemask.py — diagnostics for the open-water SST variant (revision step 1.6).
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from . import maps
from . import style as st
from .forced import arctic_mean
from .style import plt


def plot_icemask_summary(ds: xr.Dataset, out_png, years=(1990, 2030), landmask=None) -> None:
    """
    (a) fraction of member-years in ``years`` in which each cell is ice-covered in JJA
        (i.e. has its SST anomaly zeroed in the ``openwater`` inputs);
    (b) ice-covered fraction of the ocean north of 65°N through time, per forcing group.
    """
    lat, lon = ds["lat"].values, ds["lon"].values
    sub = ds.sel(year=slice(*years))
    m = sub["icemask"].values.astype(np.float32)                 # (nens, nyear, nx, ny)
    freq = m.mean(axis=(0, 1))
    if landmask is not None:
        freq = np.where(landmask == 1, np.nan, freq)
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1])
    axa = maps.map_axes(fig, gs[0])
    maps.global_map(axa, lon, lat, freq, "Blues", 0, 1,
                    f"fraction of member-years {years[0]}–{years[1]} with JJA ice cover "
                    f"(aice > {ds.attrs['threshold']:g})")
    if hasattr(axa, "set_extent"):
        try:
            axa.set_extent([-180, 180, 40, 90], crs=maps.ccrs.PlateCarree())
        except Exception:      # pragma: no cover
            pass
    axa.set_title("(a) where the open-water variant zeroes the SST anomaly", loc="left", weight="bold")

    axb = fig.add_subplot(gs[1])
    full = ds["icemask"].values.astype(np.float32)
    ocean = None if landmask is None else np.where(landmask == 1, np.nan, 1.0)
    yrs = ds["year"].values
    for name, sl, c in (("CMIP6-BB (0–49)", slice(0, 50), st.C_CMIP6), ("SMBB (50–99)", slice(50, 100), st.C_SMBB)):
        frac = arctic_mean(full[sl].mean(0) * (1.0 if ocean is None else ocean), lat)
        axb.plot(yrs, frac, color=c, lw=1.8, label=name)
    axb.axvspan(*years, color=st.GRID, alpha=0.5, zorder=0)
    axb.set_ylabel("ice-covered fraction of\nocean north of 65°N (JJA)"); axb.set_xlabel("year")
    axb.set_xlim(yrs[0], yrs[-1]); axb.set_ylim(0, 1)
    axb.legend(frameon=False); st.tidy(axb)
    axb.set_title("(b) masked fraction of the Arctic Ocean", loc="left", weight="bold")
    st.save(fig, out_png)
