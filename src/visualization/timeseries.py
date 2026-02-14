"""
Time series plotting utilities.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_timeseries(time, data, title='', xlabel='Time', ylabel='Value',
                   figsize=(12, 4), save_path=None):
    """
    Plot a simple time series.

    Parameters
    ----------
    time : array-like
        Time values (can be datetime objects)
    data : ndarray
        Data values
    title : str, optional
        Plot title
    xlabel, ylabel : str, optional
        Axis labels
    figsize : tuple, optional
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (fig, ax) matplotlib objects
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(time, data, color='black', linewidth=1.2)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_enso_phases(time, data, phases, title='', xlabel='Year', ylabel='Niño 3.4 Index',
                    figsize=(12, 4), save_path=None):
    """
    Plot time series with ENSO phase shading.

    Parameters
    ----------
    time : array-like
        Time values
    data : ndarray
        Data values (e.g., Niño 3.4 index)
    phases : ndarray
        Phase labels: 1 (El Niño), -1 (La Niña), 0 (Neutral)
    title : str, optional
        Plot title
    xlabel, ylabel : str, optional
        Axis labels
    figsize : tuple, optional
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (fig, ax) matplotlib objects
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Shade background by phase
    start = 0
    for i in range(1, len(phases)):
        if phases[i] != phases[start]:
            if phases[start] != 0:  # only shade warm/cool
                color = "red" if phases[start] == 1 else "blue"
                ax.axvspan(time[start], time[i-1], color=color, alpha=0.2)
            start = i

    # Last segment
    if phases[start] != 0:
        color = "red" if phases[start] == 1 else "blue"
        ax.axvspan(time[start], time[-1], color=color, alpha=0.2)

    # Plot the time series
    ax.plot(time, data, color="black", linewidth=1.2)

    # Zero line
    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_multiple_timeseries(time, data_dict, title='', xlabel='Time', ylabel='Value',
                             figsize=(12, 6), save_path=None):
    """
    Plot multiple time series on the same axes.

    Parameters
    ----------
    time : array-like
        Time values
    data_dict : dict
        Dictionary of {label: data_array}
    title : str, optional
        Plot title
    xlabel, ylabel : str, optional
        Axis labels
    figsize : tuple, optional
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (fig, ax) matplotlib objects
    """
    fig, ax = plt.subplots(figsize=figsize)

    for label, data in data_dict.items():
        ax.plot(time, data, label=label, linewidth=1.2)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax
