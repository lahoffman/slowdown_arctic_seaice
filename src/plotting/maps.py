"""
maps.py — global map panels for SST and relevance composites.

Uses cartopy (PlateCarree, central longitude 180) when available and falls
back to a plain lon/lat pcolormesh otherwise, so figures still render on
machines without cartopy.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from matplotlib.patches import Rectangle

from . import style as st
from .style import plt

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:  # pragma: no cover
    HAS_CARTOPY = False


def map_axes(fig, *pos, **kw):
    """Add a subplot with the project map projection (or plain axes)."""
    if HAS_CARTOPY:
        return fig.add_subplot(*pos, projection=ccrs.PlateCarree(central_longitude=180), **kw)
    return fig.add_subplot(*pos, **kw)


def global_map(ax, lon, lat, data, cmap, vmin, vmax, cbar_label: Optional[str] = None,
               cbar: bool = True, coast_lw: float = 0.6):
    """Draw one global field; returns the mappable."""
    lon2d, lat2d = np.meshgrid(lon, lat)
    kw = dict(cmap=cmap, vmin=vmin, vmax=vmax, shading="auto", zorder=0)
    if HAS_CARTOPY:
        ax.set_global()
        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=coast_lw, zorder=2)
        ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.5)
        im = ax.pcolormesh(lon2d, lat2d, data, transform=ccrs.PlateCarree(), **kw)
    else:
        im = ax.pcolormesh(lon2d, lat2d, data, **kw)
        ax.set_facecolor("lightgray")
        ax.set_ylabel("lat")
    if cbar:
        plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, fraction=0.046,
                     label=cbar_label)
    return im


def sst_panel(ax, lon, lat, sst, label="(a)", cbar=True):
    im = global_map(ax, lon, lat, sst, st.CMAP_SST, -1, 1, "SST (normalized)", cbar)
    ax.set_title(label, loc="left", weight="bold")
    return im


def lrp_panel(ax, lon, lat, lrp, label="(b)", signed=True, cbar=True):
    cmap, vmin = (st.CMAP_LRP_SIGNED, -1) if signed else (st.CMAP_LRP_POS, 0)
    im = global_map(ax, lon, lat, lrp, cmap, vmin, 1, "Relevance (normalized)", cbar)
    ax.set_title(label, loc="left", weight="bold")
    return im


def add_boxes(ax, boxes: Dict[str, Dict], colors: Optional[Dict[str, str]] = None, lw=1.8):
    """Outline lat/lon boxes ({name: {'lat': (a, b), 'lon': (c, d)}})."""
    colors = colors or {k: c for k, c in zip(boxes, st.CATEGORICAL * 3)}
    kw = dict(transform=ccrs.PlateCarree()) if HAS_CARTOPY else {}
    for name, b in boxes.items():
        ax.add_patch(Rectangle((b["lon"][0], b["lat"][0]), b["lon"][1] - b["lon"][0],
                               b["lat"][1] - b["lat"][0], fill=False, lw=lw,
                               edgecolor=colors[name], zorder=5, **kw))


def composite_pair(lon, lat, sst, lrp, signed=True, smooth_lrp=None,
                   boxes: Optional[Dict] = None, layout="row", figsize=None):
    """SST + relevance side by side (or stacked). Returns the Figure."""
    if smooth_lrp:
        from src.analysis.composites import smooth_nan
        lrp = smooth_nan(lrp, smooth_lrp)
    nrow, ncol = (1, 2) if layout == "row" else (2, 1)
    fig = plt.figure(figsize=figsize or ((13, 4.6) if layout == "row" else (6.5, 8)))
    ax_a, ax_b = map_axes(fig, nrow, ncol, 1), map_axes(fig, nrow, ncol, 2)
    sst_panel(ax_a, lon, lat, sst, "(a)")
    lrp_panel(ax_b, lon, lat, lrp, "(b)", signed)
    if boxes:
        add_boxes(ax_a, boxes); add_boxes(ax_b, boxes)
    return fig


def composite_grid(lon, lat, fields: Sequence[np.ndarray], labels: Sequence[str],
                   ncol: int = 2, kind="sst", figsize=None):
    """Grid of same-kind maps with one shared colorbar (Fig. S7 style)."""
    n = len(fields); nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=figsize or (6.5 * ncol, 3.4 * nrow + 0.8))
    axes, im = [], None
    for i, (f, lab) in enumerate(zip(fields, labels)):
        ax = map_axes(fig, nrow, ncol, i + 1)
        if kind == "sst":
            im = global_map(ax, lon, lat, f, st.CMAP_SST, -1, 1, cbar=False)
        else:
            im = global_map(ax, lon, lat, f, st.CMAP_LRP_SIGNED, -1, 1, cbar=False)
        ax.set_title(lab, loc="left", weight="bold")
        axes.append(ax)
    fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.04, pad=0.05, shrink=0.7,
                 label="SST (normalized)" if kind == "sst" else "Relevance (normalized)")
    return fig


def regional_relevance_bars(ax, regional: Dict[str, np.ndarray], n_boot: int = 1000):
    """Mean regional relevance across split×seed composites with bootstrap CIs."""
    rng = np.random.default_rng(42)
    names = list(regional)
    means, lo, hi = [], [], []
    for k in names:
        v = regional[k][np.isfinite(regional[k])]
        m = v.mean() if v.size else np.nan
        b = rng.choice(v, (n_boot, v.size), replace=True).mean(1) if v.size else np.array([np.nan])
        means.append(m); lo.append(m - np.percentile(b, 2.5)); hi.append(np.percentile(b, 97.5) - m)
    x = np.arange(len(names))
    ax.bar(x, means, yerr=[lo, hi], color=st.CATEGORICAL * 3, alpha=0.75, capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("mean relevance")
    st.tidy(ax)
