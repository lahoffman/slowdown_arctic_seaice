"""
Plotting utilities for publication figures.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# Default color scheme
COLORS = {
    'slowdown': 'green',
    'no_slowdown': 'red',
    'ensemble_bg': 'lightgray',
    'mean': 'black',
    'member': 'darkslateblue',
    'thresh_obs': 'steelblue',
    'thresh_model': 'lightblue'
}


def setup_figure_style():
    """
    Set up publication-quality figure styling.
    """
    mpl.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "figure.titlesize": 22,
        "lines.linewidth": 2.0,
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
    })


def plot_colored_decadal_segments(
    ax,
    years_window_start,
    series,
    slopes,
    slowdown_mask,
    window=10,
    lw=1.0,
    label_slowdown="decadal trend (slowdown)",
    label_noslow="decadal trend (no slowdown)",
    colors=None
):
    """
    Plot decadal trend segments colored by slowdown classification.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    years_window_start : ndarray
        Years array aligned with series (length n)
    series : ndarray
        Time series values (length n)
    slopes : ndarray
        Trend slopes (length n-window)
    slowdown_mask : ndarray
        Binary mask: 1 for slowdown, 0 for no slowdown (length n-window)
    window : int, optional
        Window size (default: 10)
    lw : float, optional
        Line width (default: 1.0)
    label_slowdown : str, optional
        Label for slowdown trends
    label_noslow : str, optional
        Label for non-slowdown trends
    colors : dict, optional
        Custom color scheme
    """
    if colors is None:
        colors = COLORS

    x = np.arange(window)
    first_slow = True
    first_noslow = True

    for j in range(slopes.size):
        y0 = series[j]
        if not np.isfinite(y0) or not np.isfinite(slopes[j]):
            continue

        is_slow = bool(slowdown_mask[j] == 1)
        color = colors['slowdown'] if is_slow else colors['no_slowdown']

        lab = ""
        if is_slow and first_slow:
            lab = label_slowdown
            first_slow = False
        if (not is_slow) and first_noslow:
            lab = label_noslow
            first_noslow = False

        yr = years_window_start[j : j + window]
        ax.plot(yr, slopes[j] * x + y0, color=color, lw=lw, label=lab)


def add_panel_label(ax, label, x=0.02, y=0.98, fontsize=20, fontweight='bold'):
    """
    Add panel label (a), (b), etc. to subplot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to add label to
    label : str
        Label text (e.g., 'a' or '(a)')
    x, y : float, optional
        Position in axes coordinates
    fontsize : int, optional
        Font size
    fontweight : str, optional
        Font weight
    """
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight=fontweight,
            va='top', ha='left')


def create_multi_panel_figure(nrows=2, ncols=3, figsize=None, **kwargs):
    """
    Create a multi-panel figure with consistent styling.

    Parameters
    ----------
    nrows, ncols : int
        Number of rows and columns
    figsize : tuple, optional
        Figure size (width, height)
    **kwargs
        Additional arguments passed to plt.subplots()

    Returns
    -------
    tuple
        (fig, axes) matplotlib objects
    """
    if figsize is None:
        figsize = (ncols * 6, nrows * 4)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)

    return fig, axes


def save_publication_figure(fig, filepath, dpi=300, bbox_inches='tight', **kwargs):
    """
    Save figure with publication-quality settings.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    filepath : str or Path
        Output file path
    dpi : int, optional
        Resolution (default: 300)
    bbox_inches : str, optional
        Bounding box setting (default: 'tight')
    **kwargs
        Additional arguments passed to fig.savefig()
    """
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
    print(f"Figure saved to {filepath}")
