"""
Map plotting utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmocean


def plot_map(data, lat, lon, title='', cmap=None, vmin=None, vmax=None,
             figsize=(12, 8), save_path=None):
    """
    Plot data on a global map.

    Parameters
    ----------
    data : ndarray
        2D data array (lat, lon)
    lat : ndarray
        Latitude values
    lon : ndarray
        Longitude values
    title : str, optional
        Plot title
    cmap : str or colormap, optional
        Colormap to use (default: cmocean.thermal)
    vmin, vmax : float, optional
        Color scale limits
    figsize : tuple, optional
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (fig, ax) matplotlib objects
    """
    if cmap is None:
        cmap = cmocean.cm.thermal

    fig, ax = plt.subplots(
        1, 1, figsize=figsize,
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)}
    )

    ax.set_global()
    ax.coastlines()

    im = ax.pcolormesh(
        lon, lat, data,
        cmap=cmap,
        shading='auto',
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax
    )

    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.xaxis.set_tick_params(rotation=45)
    ax.yaxis.set_tick_params(rotation=45)

    if title:
        ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_comparison_maps(datasets, titles=None, cmap=None, vmin=None, vmax=None,
                         figsize=(20, 10), save_path=None):
    """
    Plot multiple maps side by side for comparison.

    Parameters
    ----------
    datasets : list of dict
        List of datasets, each with keys: 'data', 'lat', 'lon'
    titles : list of str, optional
        Titles for each subplot
    cmap : str or colormap, optional
        Colormap to use
    vmin, vmax : float, optional
        Color scale limits (applied to all subplots)
    figsize : tuple, optional
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (fig, axes) matplotlib objects
    """
    if cmap is None:
        cmap = cmocean.cm.thermal

    n_plots = len(datasets)

    fig, axes = plt.subplots(
        1, n_plots, figsize=figsize,
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)}
    )

    if n_plots == 1:
        axes = [axes]

    for i, (ax, ds) in enumerate(zip(axes, datasets)):
        ax.set_global()
        ax.coastlines()

        im = ax.pcolormesh(
            ds['lon'], ds['lat'], ds['data'],
            cmap=cmap,
            shading='auto',
            transform=ccrs.PlateCarree(),
            vmin=vmin,
            vmax=vmax
        )

        ax.set_xticks(np.arange(-180, 181, 60))
        ax.set_yticks(np.arange(-90, 91, 30))
        ax.xaxis.set_tick_params(rotation=45)
        ax.yaxis.set_tick_params(rotation=45)

        if titles and i < len(titles):
            ax.set_title(titles[i])

    # Shared colorbar
    fig.subplots_adjust(bottom=0.05)
    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.03])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, axes
