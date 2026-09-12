"""
forced.py — diagnostics for the forced-response removal (revision step 1.3).

Shows what changes when each member is demeaned by its forcing group's mean
instead of the 100-member mean: the SMBB − CMIP6 difference in forced JJA SST
and the Arctic-mean forced SST per group.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from . import maps
from . import style as st
from .style import plt

ARCTIC_LAT = 65.0


def arctic_mean(field: np.ndarray, lat: np.ndarray, lat_min: float = ARCTIC_LAT) -> np.ndarray:
    """Area-weighted mean north of ``lat_min`` over the last two (lat, lon) axes."""
    w = np.cos(np.deg2rad(lat)) * (lat >= lat_min)
    w2 = np.broadcast_to(w[:, None], field.shape[-2:])
    num = np.nansum(field * w2, axis=(-2, -1))
    den = np.nansum(np.where(np.isnan(field), 0.0, w2), axis=(-2, -1))
    return num / den


def plot_group_forced_difference(
    groupmean: np.ndarray,
    names: Sequence[str],
    years: np.ndarray,
    ensmean: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    out_png,
    period: Tuple[int, int] = (2000, 2020),
    landmask: Optional[np.ndarray] = None,
) -> None:
    """
    Three panels:
      (a) map of forced JJA SST, SMBB − CMIP6, averaged over ``period``
      (b) Arctic-mean (>65°N) forced JJA SST: 100-member mean vs each group
      (c) Arctic-mean difference SMBB − CMIP6 through time
    """
    i_c, i_s = names.index("cmip6"), names.index("smbb")
    sel = (years >= period[0]) & (years <= period[1])
    if landmask is not None:                                    # land fill values must not enter any mean
        groupmean = np.where(landmask[None, None] == 1, np.nan, groupmean)
        ensmean = np.where(landmask[None] == 1, np.nan, ensmean)
    diff = groupmean[i_s] - groupmean[i_c]                      # (nyear, nlat, nlon)
    diff_map = np.nanmean(diff[sel], axis=0)
    vmax = float(np.nanpercentile(np.abs(diff_map), 99))

    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1])
    ax_a = maps.map_axes(fig, gs[0, :])
    maps.global_map(ax_a, lon, lat, diff_map, st.CMAP_SST, -vmax, vmax,
                    f"forced JJA SST, SMBB − CMIP6 ({period[0]}–{period[1]}), °C")
    ax_a.set_title("(a) Forced-response difference between forcing groups", loc="left",
                   weight="bold")

    ax_b = fig.add_subplot(gs[1, 0])
    ax_b.plot(years, arctic_mean(ensmean, lat), color=st.C_ALL, label="100-member mean")
    ax_b.plot(years, arctic_mean(groupmean[i_c], lat), color=st.C_CMIP6, label="CMIP6 BB (0–49)")
    ax_b.plot(years, arctic_mean(groupmean[i_s], lat), color=st.C_SMBB, label="SMBB (50–99)")
    ax_b.set_xlabel("year"); ax_b.set_ylabel("Arctic (>65°N) JJA SST, °C")
    ax_b.set_title("(b) Forced Arctic SST by group", loc="left", weight="bold")
    ax_b.legend(frameon=False); st.tidy(ax_b)

    ax_c = fig.add_subplot(gs[1, 1])
    d = arctic_mean(groupmean[i_s] - groupmean[i_c], lat)
    ax_c.axhline(0, color=st.MUTED, lw=0.8)
    ax_c.fill_between(years, 0, d, color=st.C_SMBB, alpha=0.35)
    ax_c.plot(years, d, color=st.INK)
    ax_c.axvspan(*period, color=st.GRID, zorder=0)
    ax_c.set_xlabel("year"); ax_c.set_ylabel("SMBB − CMIP6, °C")
    ax_c.set_title("(c) Arctic forced difference, SMBB − CMIP6", loc="left", weight="bold")
    st.tidy(ax_c)
    st.save(fig, out_png)


def plot_demeaned_arctic_index(
    sst_all: np.ndarray,
    sst_group: np.ndarray,
    years: np.ndarray,
    lat: np.ndarray,
    out_png,
    n_show: int = 100,
) -> None:
    """
    Arctic-mean internal-variability SST per member under the two demeaning
    choices: (a) 100-member mean removed, (b) group mean removed.  With (a)
    the two groups sit on opposite sides of zero during 2000–2020.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    for ax, arr, title in zip(axes, (sst_all, sst_group),
                              ("(a) 100-member mean removed", "(b) forcing-group mean removed")):
        idx = arctic_mean(arr, lat)                              # (nens, nyear)
        for g, sl, c in (("CMIP6 BB", slice(0, 50), st.C_CMIP6), ("SMBB", slice(50, 100), st.C_SMBB)):
            ax.plot(years, idx[sl][: n_show // 2].T, color=c, alpha=0.12, lw=0.7)
            ax.plot(years, idx[sl].mean(0), color=c, lw=2.4, label=f"{g} group mean")
        ax.axhline(0, color=st.MUTED, lw=0.8)
        ax.set_title(title, loc="left", weight="bold"); ax.set_xlabel("year"); st.tidy(ax)
    axes[0].set_ylabel("Arctic (>65°N) JJA SST anomaly, °C")
    axes[0].legend(frameon=False)
    st.save(fig, out_png)
