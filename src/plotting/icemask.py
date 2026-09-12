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


def plot_openwater_check(sst_before: np.ndarray, sst_after: np.ndarray, icemask: np.ndarray,
                         lat: np.ndarray, lon: np.ndarray, years: np.ndarray, out_png,
                         member: int = 6, year: int = 2010, landmask=None) -> None:
    """
    Did the mask land where it should?  Northern-hemisphere maps for one member-year:
      (a) demeaned SST anomaly as the CNN would see it without masking,
      (b) the same after the open-water mask (zero under ice), mask edge in black,
      (c) mean |anomaly| over all member-years before (left half) vs after (right half)
          is summarised as a zonal-mean curve instead — the masked fraction by latitude.
    """
    iy = int(np.where(years == year)[0][0]) if year in years else 0
    before, after, m = sst_before[member, iy], sst_after[member, iy], icemask[member, iy]
    if landmask is not None:
        before = np.where(landmask == 1, np.nan, before); after = np.where(landmask == 1, np.nan, after)
    v = float(np.nanpercentile(np.abs(before), 99))
    fig = plt.figure(figsize=(15, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.3, 1])
    titles = [f"(a) SST anomaly, member {member}, JJA {years[iy]}",
              "(b) open-water variant (black: ice edge)"]
    for k, (field, title) in enumerate(zip((before, after), titles)):
        ax = maps.map_axes(fig, gs[0, k])
        maps.global_map(ax, lon, lat, field, st.CMAP_SST, -v, v, "SST anomaly [°C]" if k == 1 else None,
                        cbar=(k == 1))
        lon2d, lat2d = np.meshgrid(lon, lat)
        kw = {"transform": maps.ccrs.PlateCarree()} if maps.HAS_CARTOPY else {}
        ax.contour(lon2d, lat2d, m.astype(float), levels=[0.5], colors="k", linewidths=0.8, **kw)
        if maps.HAS_CARTOPY:
            try:
                ax.set_extent([-180, 180, 30, 90], crs=maps.ccrs.PlateCarree())
            except Exception:  # pragma: no cover
                pass
        else:
            ax.set_ylim(30, 90)
        ax.set_title(title, loc="left", weight="bold")

    ax = fig.add_subplot(gs[0, 2])
    ocean = np.ones_like(lat, float)[:, None] * (1.0 if landmask is None else np.where(landmask == 1, np.nan, 1.0))
    frac_masked = np.nanmean(icemask.mean(axis=(0, 1)) * ocean, axis=1)          # by latitude
    nz_before = np.nanmean(np.abs(sst_before).mean(axis=(0, 1)) * ocean, axis=1)
    nz_after = np.nanmean(np.abs(sst_after).mean(axis=(0, 1)) * ocean, axis=1)
    ax.plot(frac_masked, lat, color=st.INK, lw=1.8, label="fraction of member-years masked")
    ax.plot(nz_after / np.where(nz_before > 0, nz_before, np.nan), lat, color=st.C_ALL, lw=1.8,
            label="mean |anomaly| after / before")
    ax.set_ylim(30, 90); ax.set_xlim(0, 1.02); ax.set_ylabel("latitude"); ax.set_xlabel("fraction")
    ax.set_title("(c) by latitude, all member-years", loc="left", weight="bold")
    ax.legend(frameon=False, loc="lower left"); st.tidy(ax)
    st.save(fig, out_png)
