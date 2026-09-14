"""thickness.py — figure for the thickness-as-predictor look (step 8.12): polar maps + scalar-set skill."""

from __future__ import annotations

import numpy as np

from . import style as st
from .style import plt

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


def polar_axes(fig, pos):
    if HAS_CARTOPY:
        ax = fig.add_subplot(pos, projection=ccrs.NorthPolarStereo())
        ax.set_extent([-180, 180, 60, 90], ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=2); ax.coastlines(lw=0.5, zorder=3)
        return ax
    return fig.add_subplot(pos)


def polar_map(ax, tlat, tlon, data, cmap, vmin, vmax, label=None):
    """pcolormesh on the CICE tripolar grid: project the corners ourselves so cartopy does not try to wrap them."""
    d = np.ma.masked_invalid(np.where(tlat >= 58, data, np.nan))
    if HAS_CARTOPY:
        xyz = ax.projection.transform_points(ccrs.PlateCarree(), np.asarray(tlon), np.asarray(tlat))
        x, y = xyz[..., 0], xyz[..., 1]
        bad = ~np.isfinite(x) | ~np.isfinite(y)
        x = np.where(bad, 0.0, x); y = np.where(bad, 0.0, y); d = np.ma.masked_where(bad, d)
        m = ax.pcolormesh(x, y, d, cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest", rasterized=True)
    else:
        m = ax.pcolormesh(tlon, tlat, d, cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest", rasterized=True)
    if label:
        plt.colorbar(m, ax=ax, orientation="horizontal", pad=0.04, fraction=0.05, label=label)
    return m


def plot_thickness(hi_clim, r_trend, r_resid, eof1, evr, r2_sets, au_sets, tlat, tlon, out_png, month="SEP"):
    """
    (a) mean thickness; (b) corr(hi(t), trend anomaly t+1…t+10); (c) corr(hi(t), residual after SIE);
    (d) EOF1 of the thickness anomaly; (e) test R² per scalar set; (f) test AUROC (slowdown) per scalar set.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.9])
    ax = polar_axes(fig, gs[0, 0]); polar_map(ax, tlat, tlon, hi_clim, "cividis", 0, 3, "mean thickness [m]")
    ax.set_title(f"(a) mean {month} thickness", loc="left", weight="bold")
    ax = polar_axes(fig, gs[0, 1]); polar_map(ax, tlat, tlon, r_trend, st.CMAP_SST, -0.3, 0.3, "correlation")
    ax.set_title("(b) corr(hi(t), trend t+1…t+10)", loc="left", weight="bold")
    ax = polar_axes(fig, gs[0, 2]); polar_map(ax, tlat, tlon, r_resid, st.CMAP_SST, -0.3, 0.3, "correlation")
    ax.set_title("(c) corr(hi(t), residual)", loc="left", weight="bold")
    ax = polar_axes(fig, gs[0, 3]); polar_map(ax, tlat, tlon, eof1, st.CMAP_LRP_SIGNED, -np.nanmax(np.abs(eof1)), np.nanmax(np.abs(eof1)), "EOF 1 loading")
    ax.set_title(f"(d) EOF 1 ({100 * evr[0]:.0f} %)", loc="left", weight="bold")
    for j, (sets, ylabel, title, ref) in enumerate(((r2_sets, "test R² of trend anomaly", "(e) continuous target", 0.0),
                                                     (au_sets, "test AUROC, slowdown (+1σ)", "(f) binary target", 0.5))):
        ax = fig.add_subplot(gs[1, 2 * j:2 * j + 2])
        names = list(sets)
        for i, n in enumerate(names):
            v = sets[n]; c = st.C_ALL if n.startswith("SIE") and "+" not in n else (st.C_ORIG if "vol" in n or "sector" in n else st.C_SMBB)
            ax.bar(i, np.nanmedian(v), color=c, alpha=0.85); ax.scatter(np.full(v.size, i), v, s=10, color=st.INK, alpha=0.5)
        ax.axhline(np.nanmedian(sets[names[0]]), color=st.MUTED, lw=0.8, ls="--")
        ax.axhline(ref, color=st.MUTED, lw=0.8)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=35, ha="right")
        ax.set_ylabel(ylabel); ax.set_title(title, loc="left", weight="bold"); st.tidy(ax)
    fig.tight_layout()
    if out_png is None:
        return fig
    st.save(fig, out_png)
