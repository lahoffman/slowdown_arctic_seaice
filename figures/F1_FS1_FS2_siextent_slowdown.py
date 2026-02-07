#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cleaned plotting script for:
Figure 1: panels (a)-(f)
  (a)-(b): NSIDC Sep SIE + decadal trends + trend time series (color by slowdown)
  (c)-(d): CESM2-LE ensemble-mean Sep SIE + decadal trends + trend time series (color by slowdown)
  (e)-(f): Like original FIGURE S3 panels (a)-(b), but:
           - (e) includes ALL ensemble members in light gray background (like (f))
           - slowdown markers/segments are GREEN (not red)

Figure 2:
  (a) = Figure 1(a)
  (b) = Figure 1(e)

Figure 3:
  Top row (a)-(d): like "FS2: PDF of slowdowns" but COMBINE 1990-2039 and 2040-2099 into one group
  Bottom row (e)-(h): duplicate the “FS2: PDF of slowdowns (1990-2040)” style plots as a second row
                      (kept as separate panels even though the periods are now unified above)
  Also: change slowdown color from red -> green everywhere.

Notes:
- "slowdown" definition in your code is: trend > threshold_slowdown  (kept as-is)
- Color rule requested for decadal trend segments:
    green if "yes, slowdown" else red
"""

# ------------------------------------------------------
# ROOT PATH
rootpath = "/cofast/lhoffman/slowdown/"
# ------------------------------------------------------

import sys
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import netCDF4 as nc
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime

# your helper functions (kept)
sys.path.append(rootpath + "functions/")
from functions_general import ncdisp  # noqa
from functions_general import movmean  # noqa


# ======================================================
# Styling / colors
# ======================================================
COLOR_SLOWDOWN = "green"
COLOR_NO_SLOWDOWN = "red"
COLOR_ENSEMBLE_BG = "lightgray"
COLOR_MEAN = "black"
COLOR_MEMBER = "darkslateblue"
COLOR_THRESH_OBS = "steelblue"
COLOR_THRESH_MODEL = "lightblue"

import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 18,            # base font
    "axes.titlesize": 20,        # panel labels like (a), (b)
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


# ======================================================
# Utilities
# ======================================================
def moving_decadal_trend(y: np.ndarray, window: int = 10) -> np.ndarray:
    """
    Compute moving linear trend (slope) over a rolling window.
    Returns slopes of length len(y)-window.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    slopes = np.full(n - window, np.nan)

    x = np.arange(window, dtype=float)
    for j in range(n - window):
        yy = y[j : j + window]
        mask = np.isfinite(yy)
        if mask.sum() < 2:
            continue
        # re-fit on valid indices only
        xx = x[mask]
        yy2 = yy[mask]
        m, b = np.polyfit(xx, yy2, 1)
        slopes[j] = m
    return slopes


def classify_slowdown(slopes: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    """
    Your existing logic: slowdown = slopes > threshold
    slopes: (nt,) or (n_member, nt)
    threshold: broadcastable to slopes shape
    returns int mask: 1 slowdown, 0 no slowdown
    """
    return (slopes > threshold).astype(int)


def plot_colored_decadal_segments(
    ax,
    years_window_start: np.ndarray,
    series: np.ndarray,
    slopes: np.ndarray,
    slowdown_mask: np.ndarray,
    window: int = 10,
    lw: float = 1.0,
    label_slowdown: str = "decadal trend (slowdown)",
    label_noslow: str = "decadal trend (no slowdown)",
):
    """
    Plot decadal trend segments y = m*x + y0 for each window,
    coloring by slowdown_mask[j] (1 => green, 0 => red).
    years_window_start: years array aligned with series (length n)
    series: length n (values)
    slopes: length n-window
    slowdown_mask: length n-window (0/1)
    """
    x = np.arange(window)
    first_slow = True
    first_noslow = True

    for j in range(slopes.size):
        y0 = series[j]
        if not np.isfinite(y0) or not np.isfinite(slopes[j]):
            continue

        is_slow = bool(slowdown_mask[j] == 1)
        color = COLOR_SLOWDOWN if is_slow else COLOR_NO_SLOWDOWN

        lab = ""
        if is_slow and first_slow:
            lab = label_slowdown
            first_slow = False
        if (not is_slow) and first_noslow:
            lab = label_noslow
            first_noslow = False

        yr = years_window_start[j : j + window]
        ax.plot(yr, slopes[j] * x + y0, color=color, lw=lw, label=lab)


# ======================================================
# Load NSIDC Sep SIE and compute decadal trends + thresholds
# ======================================================
def load_nsidc_monthly_sie(file_path: str, current_year: int = 2025):
    months_sheet = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    string_tail = "-NH"

    yearrange = np.arange(1978, current_year + 1)
    nn = current_year - 1978 + 1

    siextenti = []
    timei_all = []

    for i in range(12):
        sie = np.full((nn, 1), np.nan)
        years = np.full((nn, 1), np.nan)

        sheet = months_sheet[i] + string_tail
        df = pd.read_excel(file_path, sheet_name=sheet)

        timei = np.array(df[["Unnamed: 1"]])[9:]
        siei = np.array(df[["Unnamed: 5"]])[9:]

        start = 1979 if i < 10 else 1978
        end = current_year - 1 if i >= 2 else current_year

        start_idx = np.where(yearrange == start)[0][0]
        end_idx = np.where(yearrange == end)[0][0]

        sie[start_idx : end_idx + 1] = siei
        years[start_idx : end_idx + 1] = timei

        siextenti.append(sie)
        timei_all.append(years)

    siextent = np.reshape(np.array(siextenti), (12, nn))
    timeall = np.reshape(np.array(timei_all), (12, nn))

    yearsf = timeall.flatten("F")
    monthsf = np.transpose(np.tile(np.arange(1, 13), (1, nn)))
    ny = yearsf.shape[0]

    years = np.reshape(yearsf.astype(int), (ny,))
    years[:10] = 1978
    years[ny - 12 :] = current_year
    months = np.reshape(monthsf.astype(int), (ny,))

    dates = np.array([datetime(y, m, 1) for y, m in zip(years, months)])

    # monthly series per calendar month (shape 12 x n_years)
    # original code constructed data as months x years; keep consistent
    sie_flat = siextent.flatten("F")  # all months in time order
    monthly = []
    for k in range(1, 13):
        monthly.append(sie_flat[months == k])
    sie_monthly = np.array(monthly)  # (12, n_years)

    # year grid for plotting (12, n_years)
    yearmon = np.reshape(years, (nn, 12)).T  # (12, nn)

    # patch interpolation you had for Dec 1987 & Jan 1988
    # (kept exactly but guarded)
    sie = sie_flat.copy()
    if sie.size > 121:
        tint = np.arange(1, 8)
        sieint = sie[116:123]
        dx = tint.copy().astype(float)
        dy = sieint.copy().astype(float)
        dx2 = tint.copy().astype(float)

        msk = np.isfinite(dy) & np.isfinite(dx)
        if msk.sum() >= 3:
            coeff = np.polyfit(dx[msk], dy[msk], 2)
            poly = np.poly1d(coeff)
            y_fit = poly(dx2)
            sie[119] = y_fit[3]
            sie[120] = y_fit[4]

        # rebuild monthly with patched series
        monthly = []
        for k in range(1, 13):
            monthly.append(sie[months == k])
        sie_monthly = np.array(monthly)

    return sie_monthly, yearmon


def nsidc_decadal_trends_and_thresholds(sie_monthly: np.ndarray, yearmon: np.ndarray):
    """
    Computes moving decadal trends for each month (12 x (n_years-10)),
    and the obs slowdown threshold based on mean+std of obs trends (per month).
    Returns for September (mon=8): needed arrays and monthwise thresholds.
    """
    data = sie_monthly.copy()  # (12, n_years)

    # apply your NaN handling
    i98 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    i25 = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    for j in range(12):
        if i98[j] == 1:
            data[j, 0] = np.nan
        if i25[j] == 1:
            data[j, -1] = np.nan

    # trends per month
    slopes = []
    for m in range(12):
        slopes.append(moving_decadal_trend(data[m, :], window=10))
    linear_trends = np.array(slopes)  # (12, n_years-10)

    # your original: linear_trends[:,12:] due to your year selection quirks
    # here, yearmon is (12, nn) from 1978..current_year; you used 12 offset to start at 1990-ish
    # keep your behavior by slicing 12:
    linear_trends = linear_trends[:, 12:]
    nt = linear_trends.shape[1]

    mean_trend_obs = np.nanmean(linear_trends, axis=1)
    std_trend_obs = np.nanstd(linear_trends, axis=1)

    trend_obs_threshold_slowdown = mean_trend_obs + std_trend_obs
    fraction_obs_threshold_slowdown = trend_obs_threshold_slowdown / mean_trend_obs

    # for completeness (you used it elsewhere)
    trend_obs_threshold_riles = mean_trend_obs - std_trend_obs
    fraction_obs_threshold_riles = trend_obs_threshold_riles / mean_trend_obs

    # x-axis years aligned to linear_trends: yearmon[mon, 12:-10] in your code
    years_for_trends = yearmon[:, 12:-10]  # (12, nt)

    return (
        data,
        linear_trends,
        years_for_trends,
        mean_trend_obs,
        std_trend_obs,
        trend_obs_threshold_slowdown,
        fraction_obs_threshold_slowdown,
        trend_obs_threshold_riles,
        fraction_obs_threshold_riles,
    )


# ======================================================
# Load CESM2-LE Sep SIE and compute ensemble mean trends, thresholds
# ======================================================
def load_cesm2le_sep_sie(rootpath: str):
    mon_names = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    filepathlead = rootpath + "siextent_cesmle_100members_mon_"
    filepathtail = "_185001-210012.nc"

    # only SEP (index 8)
    loadpath = filepathlead + mon_names[8] + filepathtail
    ds = nc.Dataset(loadpath, "r")
    siei = np.array(ds.variables["siextentm"])  # (n_ens, n_time)
    ds.close()

    years = np.arange(1850, 2101)
    y1990 = np.where(years == 1990)[0][0]

    siei90 = siei[:, y1990:]
    years90 = years[y1990:]

    sie_ens_mean = np.nanmean(siei90, axis=0)
    return siei90, years90, sie_ens_mean


def cesm2le_ensemble_trends(siei90: np.ndarray, years90: np.ndarray):
    # ensemble mean trends
    linear_trends_mean = moving_decadal_trend(np.nanmean(siei90, axis=0), window=10)  # (ny-10,)

    # member trends
    n_ens = siei90.shape[0]
    nt = years90.size - 10
    linear_trends_ens = np.full((n_ens, nt), np.nan)
    for i in range(n_ens):
        linear_trends_ens[i, :] = moving_decadal_trend(siei90[i, :], window=10)

    return linear_trends_mean, linear_trends_ens


# ======================================================
# FIGURE 1
# ======================================================
def make_figure1(
    nsidc_data_monthly: np.ndarray,
    nsidc_yearmon: np.ndarray,
    nsidc_trends: np.ndarray,
    nsidc_years_for_trends: np.ndarray,
    mean_trend_obs: np.ndarray,
    std_trend_obs: np.ndarray,
    trend_obs_threshold_slowdown: np.ndarray,
    fraction_obs_threshold_slowdown: np.ndarray,
    siei90: np.ndarray,
    years90: np.ndarray,
    sie_ens_mean: np.ndarray,
    linear_trends_mean: np.ndarray,
    linear_trends_ens: np.ndarray,
    eno: int = 6,
):
    """
    Figure 1: (a)-(f)
    """
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(32, 24), constrained_layout=True)

    # ---------- Panels (a)-(b): NSIDC ----------
    mon = 8  # September (0-based in your plotting usage)
    axa, axb = axes[0, 0], axes[0, 1]

    # Decide slowdown for OBS trends relative to obs threshold (per your mask logic)
    nt_obs = nsidc_trends.shape[1]
    obs_threshold = np.tile(trend_obs_threshold_slowdown[mon], nt_obs)
    obs_slow_mask = classify_slowdown(nsidc_trends[mon, :], obs_threshold)

    # (a) plot NSIDC monthly Sep SIE + decadal trend segments colored by slowdown
    axa.plot(nsidc_yearmon[mon, :], nsidc_data_monthly[mon, :], lw=5, color="black", label="NSIDC SIE")

    # in your original, trend windows start at j+12 and use data[mon, j+12:j+10+12]
    # Here: nsidc_yearmon already matches nsidc_data_monthly; mimic offset=12 and windows of 10.
    offset = 12
    series = nsidc_data_monthly[mon, :]
    years = nsidc_yearmon[mon, :]

    # Build window-start years array aligned to offset
    # Trend slopes nsidc_trends[mon,:] correspond to years[ offset : -10 ]
    years_for_segments = years[offset:]  # length nn-offset
    series_for_segments = series[offset:]  # same

    # Plot each segment j in [0..nt_obs-1] using series_for_segments[j:j+10]
    plot_colored_decadal_segments(
        axa,
        years_window_start=years_for_segments,
        series=series_for_segments,
        slopes=nsidc_trends[mon, :],
        slowdown_mask=obs_slow_mask,
        window=10,
        lw=2.0,
        label_slowdown="decadal trend (yes, slowdown)",
        label_noslow="decadal trend (no slowdown)",
    )

    axa.set_title("(a)")
    axa.set_ylabel(r"September SIE [$\mathrm{M\ km^2}$]")
    axa.legend(loc="best")

    # (b) plot NSIDC decadal trend time series (color by slowdown points)
    yrs_tr = nsidc_years_for_trends[mon, :]  # (nt,)
    # draw line in neutral then overlay colored markers for classification clarity
    axb.plot(yrs_tr, nsidc_trends[mon, :], lw=3, color="gray", label="decadal trend (obs)")
    axb.scatter(
        yrs_tr[obs_slow_mask == 1],
        nsidc_trends[mon, :][obs_slow_mask == 1],
        s=48,
        color=COLOR_SLOWDOWN,
        label="yes, slowdown",
        zorder=3,
    )
    axb.scatter(
        yrs_tr[obs_slow_mask == 0],
        nsidc_trends[mon, :][obs_slow_mask == 0],
        s=48,
        color=COLOR_NO_SLOWDOWN,
        label="no slowdown",
        zorder=3,
    )

    axb.plot(yrs_tr, mean_trend_obs[mon] * np.ones_like(yrs_tr), lw=3, color=COLOR_THRESH_OBS, label="mean decadal trend")
    axb.plot(
        yrs_tr,
        (mean_trend_obs[mon] + std_trend_obs[mon]) * np.ones_like(yrs_tr),
        lw=5,
        color=COLOR_THRESH_OBS,
        ls="--",
        label=f"slowdown threshold ({fraction_obs_threshold_slowdown[mon]:.2f}×μ)",
    )
    axb.set_ylim([-0.35, 0.05])
    axb.set_title("(b)")
    axb.set_ylabel(r"decadal trend [$\mathrm{M\ km^2\ yr^{-1}}$]")
    axb.legend(loc="lower right")

    # ---------- Panels (c)-(d): CESM2-LE ensemble mean ----------
    axc, axd = axes[1, 0], axes[1, 1]

    # model threshold for ensemble-mean (your plot used fraction_obs_threshold * linear_trends_mean)
    # but classification requested is just color by slowdown vs no slowdown;
    # use your own model threshold definition from earlier:
    model_threshold_mean = fraction_obs_threshold_slowdown[8] * linear_trends_mean  # (nt,)
    model_slow_mask_mean = classify_slowdown(linear_trends_mean, model_threshold_mean)

    axc.plot(years90, sie_ens_mean, lw=5, color="black", label="CESM2-LE ensemble mean")
    # decadal segments for ensemble mean colored by slowdown
    # slopes correspond to windows years90[j:j+10] and y0 = sie_ens_mean[j]
    plot_colored_decadal_segments(
        axc,
        years_window_start=years90,
        series=sie_ens_mean,
        slopes=linear_trends_mean,
        slowdown_mask=model_slow_mask_mean,
        window=10,
        lw=3.0,
        label_slowdown="decadal trend (yes, slowdown)",
        label_noslow="decadal trend (no slowdown)",
    )
    axc.set_title("(c)")
    axc.set_ylabel(r"September SIE [$\mathrm{M\ km^2}$]")
    axc.legend(loc="best")

    axd.plot(years90[:-10], linear_trends_mean, lw=4, color="gray", label="decadal trend (ens mean)")
    axd.scatter(
        years90[:-10][model_slow_mask_mean == 1],
        linear_trends_mean[model_slow_mask_mean == 1],
        s=48,
        color=COLOR_SLOWDOWN,
        label="yes, slowdown",
        zorder=3,
    )
    axd.scatter(
        years90[:-10][model_slow_mask_mean == 0],
        linear_trends_mean[model_slow_mask_mean == 0],
        s=48,
        color=COLOR_NO_SLOWDOWN,
        label="no slowdown",
        zorder=3,
    )
    axd.plot(
        years90[:-10],
        trend_obs_threshold_slowdown[8] * np.ones_like(linear_trends_mean),
        lw=4,
        color=COLOR_THRESH_OBS,
        ls="--",
        label="obs slowdown threshold",
    )
    axd.plot(
        years90[:-10],
        model_threshold_mean,
        lw=4,
        color=COLOR_THRESH_MODEL,
        label="model slowdown threshold",
    )
    axd.set_title("(d)")
    axd.set_ylabel(r"decadal trend [$\mathrm{M\ km^2\ yr^{-1}}$]")
    axd.legend(loc="best")

    # ---------- Panels (e)-(f): like your FIGURE S3, but (e) also has ensemble bg ----------
    axe, axf = axes[2, 0], axes[2, 1]

    # build slowdown mask for all members using your same threshold definition
    model_obs_threshold_slowdown = fraction_obs_threshold_slowdown[8] * linear_trends_mean  # (nt,)
    threshold_all = np.tile(model_obs_threshold_slowdown, (siei90.shape[0], 1))  # (n_ens, nt)
    slowdown = classify_slowdown(linear_trends_ens, threshold_all)  # (n_ens, nt)
    linear_trends_slowdown = np.where(slowdown == 1, linear_trends_ens, np.nan)

    # For the member time series points: use siei90[:, :-10] aligned with trend windows
    sie_for_trend_windows = siei90[:, :-10].copy()  # (n_ens, nt) if nt = ny-10
    sie_for_trend_windows[np.isnan(linear_trends_slowdown)] = np.nan  # keep only slowdown windows

    # (e) ensemble members in bg + one highlighted member + its slowdown markers + its slowdown trend segments
    for i in range(siei90.shape[0]):
        axe.plot(years90, siei90[i, :], lw=0.2, color=COLOR_ENSEMBLE_BG)

    # plot slowdown trend segments for selected member eno in GREEN
    for j in range(linear_trends_mean.size):
        if not np.isfinite(sie_for_trend_windows[eno, j]):
            continue
        dx = np.arange(10)
        y0 = siei90[eno, j]
        m = linear_trends_ens[eno, j]
        axe.plot(years90[j : j + 10], m * dx + y0, lw=4.0, color=COLOR_SLOWDOWN)

    axe.plot(years90, sie_ens_mean, lw=5, color=COLOR_MEAN, label="CESM2-LE ensemble mean")
    axe.plot(years90, siei90[eno, :], lw=2.5, color=COLOR_MEMBER, label=f"ensemble no. {eno+1}")
    axe.plot(
        years90[:-10],
        sie_for_trend_windows[eno, :],
        lw=0,
        marker="o",
        ms=4,
        color=COLOR_SLOWDOWN,
        label=f"ensemble no. {eno+1} slowdowns",
    )
    axe.set_title("(e)")
    axe.set_ylabel(r"September SIE [$\mathrm{M\ km^2}$]")
    ensemble_handle = Line2D(
        [], [], color=COLOR_ENSEMBLE_BG, lw=2, label="ensemble members"
    )

    handles, labels = axe.get_legend_handles_labels()
    handles = [ensemble_handle] + handles
    axe.legend(handles=handles, loc="best")


    # (f) decadal trends for all members in bg + mean + thresholds + selected member + slowdown markers
    for i in range(siei90.shape[0]):
        axf.plot(years90[:-10], linear_trends_ens[i, :], lw=0.2, color=COLOR_ENSEMBLE_BG)

    axf.plot(years90[:-10], linear_trends_mean, lw=3, color=COLOR_MEAN, label="ensemble mean")
    axf.plot(
        years90[:-10],
        trend_obs_threshold_slowdown[8] * np.ones_like(linear_trends_mean),
        lw=4,
        color=COLOR_THRESH_OBS,
        ls="--",
        label="obs slowdown threshold",
    )
    axf.plot(
        years90[:-10],
        model_obs_threshold_slowdown,
        lw=4,
        color=COLOR_THRESH_MODEL,
        label="model slowdown threshold",
    )
    axf.plot(years90[:-10], linear_trends_ens[eno, :], lw=4, color=COLOR_MEMBER, label=f"ensemble no. {eno+1}")
    axf.plot(
        years90[:-10],
        linear_trends_slowdown[eno, :],
        lw=0,
        marker="o",
        ms=12,
        color=COLOR_SLOWDOWN,
        label=f"ensemble no. {eno+1} slowdowns",
    )
    axf.set_title("(f)")
    axf.set_ylabel(r"decadal trend [$\mathrm{M\ km^2\ yr^{-1}}$]")
    axf.set_xlim([1990, 2100])
    ensemble_handle = Line2D(
        [], [], color=COLOR_ENSEMBLE_BG, lw=2, label="ensemble members"
    )

    handles, labels = axf.get_legend_handles_labels()
    handles = [ensemble_handle] + handles
    axf.legend(handles=handles, loc="best")


    return fig


# ======================================================
# FIGURE 2 (reuse axes content from Figure 1 by re-plotting cleanly)
# ======================================================
from PIL import Image

from PIL import Image
from matplotlib.lines import Line2D

def add_panel_label(ax, label, x=0.0, y=1.06, fontsize=20):
    """Put panel label slightly ABOVE the axes."""
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False
    )

def make_figure2(fig1_builder_kwargs, png_path, png_width=1.0):
    """
    Layout:
      (a) PNG spans 1 column x 2 rows (left)
      (b) NSIDC panel (top-right)
      (c) Ensemble panel (bottom-right)
    """

    # unpack
    nsidc_data_monthly = fig1_builder_kwargs["nsidc_data_monthly"]
    nsidc_yearmon = fig1_builder_kwargs["nsidc_yearmon"]
    nsidc_trends = fig1_builder_kwargs["nsidc_trends"]
    nsidc_years_for_trends = fig1_builder_kwargs["nsidc_years_for_trends"]
    mean_trend_obs = fig1_builder_kwargs["mean_trend_obs"]
    std_trend_obs = fig1_builder_kwargs["std_trend_obs"]
    trend_obs_threshold_slowdown = fig1_builder_kwargs["trend_obs_threshold_slowdown"]
    fraction_obs_threshold_slowdown = fig1_builder_kwargs["fraction_obs_threshold_slowdown"]

    siei90 = fig1_builder_kwargs["siei90"]
    years90 = fig1_builder_kwargs["years90"]
    sie_ens_mean = fig1_builder_kwargs["sie_ens_mean"]
    linear_trends_mean = fig1_builder_kwargs["linear_trends_mean"]
    linear_trends_ens = fig1_builder_kwargs["linear_trends_ens"]
    eno = fig1_builder_kwargs.get("eno", 6)

    # --- Figure + GridSpec (a spans 2 rows) ---
    fig = plt.figure(figsize=(36, 16))  # tweak to taste; tall enough for 2 rows
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        width_ratios=[2.4, 1.0],
        wspace=0.25, hspace=0.30
    )

    ax_a = fig.add_subplot(gs[:, 0])   # spans both rows
    ax_b = fig.add_subplot(gs[0, 1])   # top-right
    ax_c = fig.add_subplot(gs[1, 1])   # bottom-right

    # -------------------------------------------------
    # (a) PNG panel (spans rows)
    # -------------------------------------------------
    img = Image.open(png_path)
    ax_a.imshow(img, aspect="auto")
    img_w, img_h = img.size
    ax_a.axis("off")
    add_panel_label(ax_a, "(a)")
    ax_a.set_box_aspect(img_h / img_w)

    # -------------------------------------------------
    # (b) NSIDC (same as your Figure 2(b) previously)
    # -------------------------------------------------
    mon = 8
    nt_obs = nsidc_trends.shape[1]
    obs_threshold = np.tile(trend_obs_threshold_slowdown[mon], nt_obs)
    obs_slow_mask = classify_slowdown(nsidc_trends[mon, :], obs_threshold)

    ax_b.plot(nsidc_yearmon[mon, :], nsidc_data_monthly[mon, :],
              lw=3, color="black", label="NSIDC SIE")

    offset = 12
    series = nsidc_data_monthly[mon, :]
    years = nsidc_yearmon[mon, :]

    plot_colored_decadal_segments(
        ax_b,
        years_window_start=years[offset:],
        series=series[offset:],
        slopes=nsidc_trends[mon, :],
        slowdown_mask=obs_slow_mask,
        window=10,
        lw=1.0,
        label_slowdown="decadal trend (yes, slowdown)",
        label_noslow="decadal trend (no slowdown)",
    )

    ax_b.set_ylabel(r"September SIE [$\mathrm{M\ km^2}$]")
    add_panel_label(ax_b, "(b)")
    ax_b.legend(loc="best")

    # -------------------------------------------------
    # (c) Ensemble member + background members
    # -------------------------------------------------
    model_obs_threshold_slowdown = fraction_obs_threshold_slowdown[8] * linear_trends_mean
    threshold_all = np.tile(model_obs_threshold_slowdown, (siei90.shape[0], 1))
    slowdown = classify_slowdown(linear_trends_ens, threshold_all)
    linear_trends_slowdown = np.where(slowdown == 1, linear_trends_ens, np.nan)

    sie_for_trend_windows = siei90[:, :-10].copy()
    sie_for_trend_windows[np.isnan(linear_trends_slowdown)] = np.nan

    # background ensemble members (no labels)
    for i in range(siei90.shape[0]):
        ax_c.plot(years90, siei90[i, :], lw=0.2, color=COLOR_ENSEMBLE_BG)

    # slowdown trend segments for selected member
    for j in range(linear_trends_mean.size):
        if not np.isfinite(sie_for_trend_windows[eno, j]):
            continue
        dx = np.arange(10)
        y0 = siei90[eno, j]
        m = linear_trends_ens[eno, j]
        ax_c.plot(years90[j:j+10], m*dx + y0, lw=1.0, color=COLOR_SLOWDOWN)

    ax_c.plot(years90, sie_ens_mean, lw=3, color=COLOR_MEAN, label="CESM2-LE ensemble mean")
    ax_c.plot(years90, siei90[eno, :], lw=2.5, color=COLOR_MEMBER, label=f"ensemble no. {eno+1}")
    ax_c.plot(years90[:-10], sie_for_trend_windows[eno, :], lw=0, marker="o", ms=4,
              color=COLOR_SLOWDOWN, label=f"ensemble no. {eno+1} slowdowns")

    # one legend entry for ensemble members
    ensemble_handle = Line2D([], [], color=COLOR_ENSEMBLE_BG, lw=2, label="ensemble members")
    handles, labels = ax_c.get_legend_handles_labels()
    ax_c.legend(handles=[ensemble_handle] + handles, loc="best")

    ax_c.set_ylabel(r"September SIE [$\mathrm{M\ km^2}$]")
    add_panel_label(ax_c, "(c)")

    # ensure top margin for labels outside axes
    fig.subplots_adjust(top=0.95)

    return fig



# ======================================================
# FIGURE 3
# ======================================================
def make_figure3(
    siei90: np.ndarray,
    years90: np.ndarray,
    linear_trends_mean: np.ndarray,
    linear_trends_ens: np.ndarray,
    fraction_obs_threshold_slowdown: np.ndarray,
):
    """
    Figure 3:
      Top row (a)-(d): like your first FS2 block BUT combine 1990-2039 and 2040-2099 into one group.
      Bottom row (e)-(h): duplicate your later FS2 block (the one that used [:,:50]) as panels e-h.
                          (Colors: slowdown green, no-slowdown stays blue/gray as appropriate)
    """
    # thresholds per your logic
    model_obs_threshold_slowdown = fraction_obs_threshold_slowdown[8] * linear_trends_mean  # (nt,)
    threshold_all = np.tile(model_obs_threshold_slowdown, (siei90.shape[0], 1))
    slowdown = classify_slowdown(linear_trends_ens, threshold_all)  # (100, nt)

    # ---------- top row uses ALL years (combined group) ----------
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(44, 18), constrained_layout=True)

    axa, axb, axc, axd = axes[0, :]
    axe, axf, axg, axh = axes[1, :]

    # (a) frequency of slowdown events per member (single combined distribution)
    slowdown_per_member = np.nansum(slowdown, axis=1)
    max_val = int(np.nanmax(slowdown_per_member))
    bins = np.arange(0, max_val + 3) - 1
    counts, bin_edges = np.histogram(slowdown_per_member, bins=bins, density=True)
    axa.bar(
        bin_edges[:-1],
        counts,
        width=1.0,
        color=COLOR_THRESH_OBS,
        edgecolor="white",
        alpha=0.7,
        
    )
    axa.set_xlabel("Number of slowdown events per member")
    axa.set_ylabel("Frequency")
    axa.set_xticks(np.arange(0, max_val + 2, 3))
    axa.set_title("(a)")
    axa.legend()

    # (b) overall fraction of events that are slowdown vs no slowdown (combined)
    slowdown_flat = slowdown.reshape(-1)
    axb.hist(
        slowdown_flat,
        bins=[-0.5, 0.5, 1.5],
        weights=np.ones_like(slowdown_flat) / len(slowdown_flat),
        color="grey",
    )
    axb.set_xticks([0, 1], ["no slowdown", "slowdown"])
    axb.set_ylabel("fraction of events")
    axb.set_title("(b)")

    # (c) PDF of SIE during slowdowns vs not (combined)
    sie_windows = siei90[:, :-10]  # aligned with slowdown mask
    mask = slowdown
    data_0 = sie_windows[mask == 0]
    data_1 = sie_windows[mask == 1]
    axc.hist(data_0, bins=30, alpha=0.6, label="no slowdown", density=True, color=COLOR_NO_SLOWDOWN)
    axc.hist(data_1, bins=30, alpha=0.6, label="slowdown", density=True, color=COLOR_SLOWDOWN)
    axc.set_xlabel(r"sea ice extent [M km$^2$]")
    axc.set_ylabel("Density")
    axc.legend()
    axc.set_title("(c)")

    # (d) PDF of SIE anomaly during slowdowns vs not (combined)
    sie_anom = sie_windows - np.nanmean(sie_windows, axis=0)
    data_0 = sie_anom[mask == 0]
    data_1 = sie_anom[mask == 1]
    axd.hist(data_0, bins=30, alpha=0.6, label="no slowdown", density=True, color=COLOR_NO_SLOWDOWN)
    axd.hist(data_1, bins=30, alpha=0.6, label="slowdown", density=True, color=COLOR_SLOWDOWN)
    axd.set_xlabel(r"sea ice extent anomaly [M km$^2$]")
    axd.set_ylabel("Density")
    axd.legend()
    axd.set_title("(d)")

    # ---------- bottom row duplicates your later FS2 (was [:,:50]) ----------
    # Keeping your original “first 50 windows” selection as the duplicate,
    # but now panel (e) should show BOTH PDFs (1990-2039 and 2040-2099).

    subset_9039 = slice(0, 50)    # 1990–2039 (50 windows)
    subset_4099 = slice(50, None) # 2040–2099

    slowdown_9039 = slowdown[:, subset_9039]
    slowdown_4099 = slowdown[:, subset_4099]

    # (e) frequency per member: overlay PDFs for 1990–2039 and 2040–2099
    slowdown_per_member_9039 = np.nansum(slowdown_9039, axis=1)
    slowdown_per_member_4099 = np.nansum(slowdown_4099, axis=1)

    max_val2 = int(np.nanmax(np.concatenate([slowdown_per_member_9039, slowdown_per_member_4099])))
    bins2 = np.arange(0, max_val2 + 3) - 1

    counts2, edges2 = np.histogram(slowdown_per_member_9039, bins=bins2, density=True)
    axe.bar(
        edges2[:-1], counts2, width=1.0,
        color=COLOR_THRESH_OBS, edgecolor="white", alpha=0.7,
        label="1990–2039"
    )

    counts3, edges3 = np.histogram(slowdown_per_member_4099, bins=bins2, density=True)
    axe.bar(
        edges3[:-1], counts3, width=1.0,
        color="red", edgecolor="white", alpha=0.2,   # keep your old styling for the later period
        label="2040–2099"
    )

    axe.set_xlabel("Number of slowdown events per member")
    axe.set_ylabel("Frequency")
    axe.set_xticks(np.arange(0, max_val2 + 2, 3))
    axe.set_title("(e)")
    axe.legend()


    # (f) fraction slowdown vs no slowdown (subset)
    subset = slice(0, 50)
    slowdown_sub = slowdown[:, subset]
    sie_windows_sub_aligned = siei90[:, :-10][:, subset]

    slowdown_flat_sub = slowdown_sub.reshape(-1)
    axf.hist(
        slowdown_flat_sub,
        bins=[-0.5, 0.5, 1.5],
        weights=np.ones_like(slowdown_flat_sub) / len(slowdown_flat_sub),
        color="grey",
    )
    axf.set_xticks([0, 1], ["no slowdown", "slowdown"])
    axf.set_ylabel("fraction of events")
    axf.set_title("(f)")

    # (g) PDF of SIE during events (subset, aligned)
    mask_sub = slowdown_sub
    d0 = sie_windows_sub_aligned[mask_sub == 0]
    d1 = sie_windows_sub_aligned[mask_sub == 1]
    axg.hist(d0, bins=30, alpha=0.6, label="no slowdown", density=True, color=COLOR_NO_SLOWDOWN)
    axg.hist(d1, bins=30, alpha=0.6, label="slowdown", density=True, color=COLOR_SLOWDOWN)
    axg.set_xlabel(r"sea ice extent [M km$^2$]")
    axg.set_ylabel("Density")
    axg.legend()
    axg.set_title("(g)")

    # (h) PDF of anomalies during events (subset, aligned)
    anom_sub = sie_windows_sub_aligned - np.nanmean(sie_windows_sub_aligned, axis=0)
    d0 = anom_sub[mask_sub == 0]
    d1 = anom_sub[mask_sub == 1]
    axh.hist(d0, bins=30, alpha=0.6, label="no slowdown", density=True, color=COLOR_NO_SLOWDOWN)
    axh.hist(d1, bins=30, alpha=0.6, label="slowdown", density=True, color=COLOR_SLOWDOWN)
    axh.set_xlabel(r"sea ice extent anomaly [M km$^2$]")
    axh.set_ylabel("Density")
    axh.legend()
    axh.set_title("(h)")

    return fig


# ======================================================
# Main
# ======================================================
def main():
    # ---------- NSIDC ----------
    nsidc_xlsx = rootpath + "Sea_Ice_Index_Monthly_Data_with_Statistics_G02135_v3.0.xlsx"
    sie_monthly, yearmon = load_nsidc_monthly_sie(nsidc_xlsx, current_year=2025)

    (
        nsidc_data_monthly,
        nsidc_trends,
        nsidc_years_for_trends,
        mean_trend_obs,
        std_trend_obs,
        trend_obs_threshold_slowdown,
        fraction_obs_threshold_slowdown,
        trend_obs_threshold_riles,
        fraction_obs_threshold_riles,
    ) = nsidc_decadal_trends_and_thresholds(sie_monthly, yearmon)

    # ---------- CESM2-LE ----------
    siei90, years90, sie_ens_mean = load_cesm2le_sep_sie(rootpath)
    linear_trends_mean, linear_trends_ens = cesm2le_ensemble_trends(siei90, years90)

    # ---------- Figure 1 ----------
    fig1 = make_figure1(
        nsidc_data_monthly=nsidc_data_monthly,
        nsidc_yearmon=yearmon,
        nsidc_trends=nsidc_trends,
        nsidc_years_for_trends=nsidc_years_for_trends,
        mean_trend_obs=mean_trend_obs,
        std_trend_obs=std_trend_obs,
        trend_obs_threshold_slowdown=trend_obs_threshold_slowdown,
        fraction_obs_threshold_slowdown=fraction_obs_threshold_slowdown,
        siei90=siei90,
        years90=years90,
        sie_ens_mean=sie_ens_mean,
        linear_trends_mean=linear_trends_mean,
        linear_trends_ens=linear_trends_ens,
        eno=6,
    )


    # ---------- Figure 2 ----------
    fig2 = make_figure2(
        dict(
            nsidc_data_monthly=nsidc_data_monthly,
            nsidc_yearmon=yearmon,
            nsidc_trends=nsidc_trends,
            nsidc_years_for_trends=nsidc_years_for_trends,
            mean_trend_obs=mean_trend_obs,
            std_trend_obs=std_trend_obs,
            trend_obs_threshold_slowdown=trend_obs_threshold_slowdown,
            fraction_obs_threshold_slowdown=fraction_obs_threshold_slowdown,
            siei90=siei90,
            years90=years90,
            sie_ens_mean=sie_ens_mean,
            linear_trends_mean=linear_trends_mean,
            linear_trends_ens=linear_trends_ens,
            eno=6,
        ),
        png_path="/home/elic/lhoffman/cnn_slowdown/manuscript/d2/figures/cnn_schematic.png",
        png_width=1.2   # increase if you want PNG panel wider
    )



    # ---------- Figure 3 ----------
    fig3 = make_figure3(
        siei90=siei90,
        years90=years90,
        linear_trends_mean=linear_trends_mean,
        linear_trends_ens=linear_trends_ens,
        fraction_obs_threshold_slowdown=fraction_obs_threshold_slowdown,
    )


    plt.show()


if __name__ == "__main__":
    main()
