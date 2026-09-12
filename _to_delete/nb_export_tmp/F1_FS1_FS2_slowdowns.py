
# ===== MD [0] =====
# # CESM2-LE & NSIDC Sea Ice Slowdown Figures
# 
# Builds the three figures from `x_old/figures/F1_FS1_FS2_siextent_slowdown.py` by loading
# pre-computed data from:
# - **`scripts/01_slowdown_nsidc_sie_sia.py`** → NSIDC thresholds + per-month event files
# - **`scripts/02_cesm2le_slowdowns.py`** → CESM2-LE slowdown classification files
# - **CESM2-LE metrics files** → raw ensemble SIE/SIA time series
# 
# **Toggle the variable and month in Cell 2 (Configuration).**

# ===== CODE [1] =====
import sys
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/lhoffma2/git/slowdown_arctic_seaice")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

%matplotlib inline
print('Imports OK')

# ===== MD [2] =====
# ## Configuration
# Change `VARIABLE` and `MONTH` here to regenerate all figures for a different variable / month.

# ===== CODE [3] =====
from configs import paths

# ─────────────────────────────────────────────────────────────────────────────
# TOGGLE THESE
# ─────────────────────────────────────────────────────────────────────────────
VARIABLE = 'sie'    # 'sie'  or  'sia'
MONTH    = 'SEP'    # 'JAN' | 'FEB' | 'MAR' | 'APR' | 'MAY' | 'JUN'
                    # 'JUL' | 'AUG' | 'SEP' | 'OCT' | 'NOV' | 'DEC'
ENO      = 6        # Ensemble member to highlight in Fig 1 panels (e,f)  [0-based]

# Optional: path to CNN schematic PNG for Figure 2 panel (a)
# Set to None to skip the PNG panel and show only the two data panels.
PNG_PATH = paths.DATA_ROOT / 'models/schematic/cnn_schematic.png'     # e.g. '/path/to/cnn_schematic.png'

# ─────────────────────────────────────────────────────────────────────────────
MONTH_LABELS = ['JAN','FEB','MAR','APR','MAY','JUN',
                'JUL','AUG','SEP','OCT','NOV','DEC']
MONTH_IDX  = MONTH_LABELS.index(MONTH)   # 0-based  (e.g. SEP → 8)
MONTH_NUM  = MONTH_IDX + 1               # 1-based  (e.g. SEP → 9)

VARNAME  = 'SIE' if VARIABLE == 'sie' else 'SIA'
VARLABEL = 'Sea Ice Extent' if VARIABLE == 'sie' else 'Sea Ice Area'
raw_prefix = 'siextentn' if VARIABLE == 'sie' else 'siarean'

print(f'Variable : {VARNAME}')
print(f'Month    : {MONTH}  (0-based index {MONTH_IDX})')
print(f'ENO      : {ENO}')

# ===== MD [4] =====
# ## Paths

# ===== CODE [5] =====
# Locate project root (notebook lives in notebooks/)
NOTEBOOKS_DIR = Path().resolve()
PROJECT_ROOT  = NOTEBOOKS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── NSIDC ────────────────────────────────────────────────────────────────────
nsidc_thresholds_path = (
    paths.NSIDC_SIE_SLOWDOWN_THRESHOLDS if VARIABLE == 'sie'
    else paths.NSIDC_SIA_SLOWDOWN_THRESHOLDS
)
nsidc_events_path = (
    paths.nsidc_sie_slowdown_events(MONTH_NUM) if VARIABLE == 'sie'
    else paths.nsidc_sia_slowdown_events(MONTH_NUM)
)

# ── CESM2-LE slowdown classification files ───────────────────────────────────
cesm_slowdowns_dir = paths.CESM2LE_DIR / 'slowdowns'
cesm_slowdown_path = (
    cesm_slowdowns_dir
    / f'cesm2le_{VARIABLE}_slowdown_riles_{MONTH}_1990-2100.nc'
)

# ── CESM2-LE raw SIE / SIA metrics files (for time-series panels) ────────────
cesm_metrics_dir = paths.CESM2LE_AICE_DIR / 'metrics'

print('NSIDC thresholds :', nsidc_thresholds_path)
print('NSIDC events     :', nsidc_events_path)
print('CESM slowdowns   :', cesm_slowdown_path)
print('CESM metrics dir :', cesm_metrics_dir)

# Quick existence check
for p in [nsidc_thresholds_path, nsidc_events_path, cesm_slowdown_path]:
    status = '✓' if Path(p).exists() else '✗  NOT FOUND'
    print(f'  {status}  {Path(p).name}')

# ===== MD [6] =====
# ## Load NSIDC data

# ===== CODE [7] =====
# ── Thresholds (all 12 months in one file) ────────────────────────────────────
ds_thr = xr.open_dataset(nsidc_thresholds_path)
threshold_slowdown_all = ds_thr['threshold_slowdown'].values   # (12,)  M km² yr⁻¹
fraction_slowdown_all  = ds_thr['fraction_slowdown'].values    # (12,)  threshold / mean
ds_thr.close()

# For the selected month
thr_slow  = float(threshold_slowdown_all[MONTH_IDX])   # mean + std
frac_slow = float(fraction_slowdown_all[MONTH_IDX])    # threshold / mean

# Reconstruct mean and std of obs trends from threshold + fraction
# threshold = mean + std  and  fraction = threshold / mean
# → mean = threshold / fraction,  std = threshold - mean
mean_trend_obs = thr_slow / frac_slow
std_trend_obs  = thr_slow - mean_trend_obs

print(f'NSIDC {MONTH} threshold (slowdown) : {thr_slow:+.4f} M km² yr⁻¹')
print(f'  fraction  : {frac_slow:.4f}')
print(f'  mean trend: {mean_trend_obs:+.4f} M km² yr⁻¹')
print(f'  std  trend: {std_trend_obs:.4f} M km² yr⁻¹')

# ── Per-month events file (selected month) ────────────────────────────────────
ds_ev = xr.open_dataset(nsidc_events_path)

# Trend windows: starting year + slope + slowdown mask
nsidc_trend_years = ds_ev['year'].values.astype(int)         # (n_trends,)
nsidc_trends      = ds_ev['linear_trend'].values             # (n_trends,)
nsidc_slowdown    = ds_ev['slowdown'].values.astype(int)     # (n_trends,)  0/1

# Sea-ice time series starting from 1990
# Note: dim is 'yearsice'; 'yearice' is a separate non-dimension coordinate
nsidc_seaice       = ds_ev['seaice'].values                  # (n_years,)
nsidc_seaice_years = ds_ev.coords['yearice'].values.astype(int)  # (n_years,)

ds_ev.close()

print(f'\nNSIDC trend windows : {nsidc_trend_years[0]}–{nsidc_trend_years[-1]}'
      f'  (n={len(nsidc_trend_years)})')
print(f'NSIDC seaice years  : {nsidc_seaice_years[0]}–{nsidc_seaice_years[-1]}'
      f'  (n={len(nsidc_seaice_years)})')
print(f'Slowdown events     : {nsidc_slowdown.sum()} / {len(nsidc_slowdown)}')

# ===== MD [8] =====
# ## Load CESM2-LE data

# ===== CODE [9] =====
# ── CESM2-LE slowdown classification ─────────────────────────────────────────
ds_cesm = xr.open_dataset(cesm_slowdown_path)

trend_years             = ds_cesm['nyr'].values.astype(int)       # (n_trends,) e.g. 1990–2090
linear_trends_mean      = ds_cesm['linear_trends_mean'].values    # (n_trends,)
linear_trends_ens       = ds_cesm['linear_trends_ens'].values     # (nens, n_trends)
slowdown_ens            = ds_cesm['slowdown'].values.astype(int)  # (nens, n_trends)
threshold_slowdown_cesm = ds_cesm['threshold_slowdown'].values    # (n_trends,) = frac×mean_trend

ds_cesm.close()

n_ens        = linear_trends_ens.shape[0]
n_trends_cesm = linear_trends_mean.shape[0]

print(f'CESM trend windows: {trend_years[0]}–{trend_years[-1]}  (n={n_trends_cesm})')
print(f'Ensemble size     : {n_ens}')
print(f'Slowdowns total   : {slowdown_ens.sum()} / {slowdown_ens.size}'
      f'  ({100 * slowdown_ens.mean():.1f}%)')

# ── CESM2-LE raw SIE/SIA — for time-series panels ────────────────────────────
# Loads siextentn_{group}members_{MONTH}.nc  (or siarean_...)
# Files have dims (ensemble, time), shape ≈ (50, 111) for years 1990–2100
years90 = np.arange(1990, 2101)    # full 111-year time axis

ensemble_arrays = []
for group in ['first50', 'last50']:
    fpath = cesm_metrics_dir / f'{raw_prefix}_{group}members_{MONTH}.nc'
    if fpath.exists():
        ds_raw = xr.open_dataset(fpath)
        arr = ds_raw[raw_prefix].values.astype(np.float32)   # (n_group, nyear)
        ds_raw.close()
        ensemble_arrays.append(arr)
        print(f'  Loaded {fpath.name}  shape={arr.shape}')
    else:
        print(f'  WARNING: not found — {fpath.name}')

if ensemble_arrays:
    siei90       = np.concatenate(ensemble_arrays, axis=0)   # (100, 111)
    sie_ens_mean = np.nanmean(siei90, axis=0)                # (111,)
    print(f'\nCombined SIE shape : {siei90.shape}')
else:
    siei90 = sie_ens_mean = None
    print('WARNING: raw SIE not available — time-series panels will be skipped')

# ===== MD [10] =====
# ## Derived quantities, styling & plot helpers

# ===== CODE [11] =====
# ── Colours ───────────────────────────────────────────────────────────────────
COLOR_SLOWDOWN    = 'green'
COLOR_NO_SLOWDOWN = 'red'
COLOR_ENSEMBLE_BG = 'lightgray'
COLOR_MEAN        = 'black'
COLOR_MEMBER      = 'darkslateblue'
COLOR_THRESH_OBS  = 'steelblue'
COLOR_THRESH_MODEL = 'lightblue'

mpl.rcParams.update({
    'font.size'        : 18,
    'axes.titlesize'   : 20,
    'axes.labelsize'   : 20,
    'xtick.labelsize'  : 16,
    'ytick.labelsize'  : 16,
    'legend.fontsize'  : 16,
    'figure.titlesize' : 22,
    'lines.linewidth'  : 2.0,
    'axes.linewidth'   : 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.major.size' : 6,
    'ytick.major.size' : 6,
})


# ── Axis-label helpers ────────────────────────────────────────────────────────
def ice_ylabel():
    return rf'{MONTH} {VARNAME} [M km$^2$]'

def trend_ylabel():
    return rf'decadal trend [M km$^2$ yr$^{{-1}}$]'


# ── Panel label ───────────────────────────────────────────────────────────────
def add_panel_label(ax, label, x=0.0, y=1.06, fontsize=20):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold',
            va='bottom', ha='left', clip_on=False)


# ── Coloured decadal-trend segment plotter ────────────────────────────────────
def plot_colored_decadal_segments(
    ax, years_window_start, series, slopes, slowdown_mask,
    window=10, lw=1.0,
    label_slowdown='decadal trend (yes, slowdown)',
    label_noslow='decadal trend (no slowdown)',
):
    """
    For each window j, plot the linear segment y = slope[j]*x + series[j]
    in green (slowdown) or red (no slowdown).
    """
    x = np.arange(window)
    first_slow = first_noslow = True
    for j in range(slopes.size):
        y0 = series[j]
        if not np.isfinite(y0) or not np.isfinite(slopes[j]):
            continue
        is_slow = bool(slowdown_mask[j] == 1)
        color   = COLOR_SLOWDOWN if is_slow else COLOR_NO_SLOWDOWN
        lab = ''
        if is_slow and first_slow:
            lab = label_slowdown;  first_slow = False
        elif (not is_slow) and first_noslow:
            lab = label_noslow;    first_noslow = False
        yr = years_window_start[j : j + window]
        ax.plot(yr, slopes[j] * x + y0, color=color, lw=lw, label=lab)


# ── Derived quantities from CESM2-LE ─────────────────────────────────────────
# Classify ensemble *mean* against the time-varying model threshold
# (threshold_slowdown_cesm = fraction_nsidc × linear_trends_mean, loaded from file)
model_slow_mask_mean   = (linear_trends_mean > threshold_slowdown_cesm).astype(int)

# Per-member trends: NaN where not a slowdown
linear_trends_slowdown = np.where(slowdown_ens == 1, linear_trends_ens, np.nan)

# SIE values at trend-window start years, NaN where member is *not* in slowdown
# Shape: (nens, n_trends_cesm)  — aligns siei90[:, :n_trends_cesm] with trend windows
if siei90 is not None:
    sie_for_trend_windows = siei90[:, :n_trends_cesm].copy()   # (nens, n_trends)
    sie_for_trend_windows[np.isnan(linear_trends_slowdown)] = np.nan
else:
    sie_for_trend_windows = None

print('Derived quantities ready.')

# ===== MD [12] =====
# ---
# ## Figure 1 — NSIDC and CESM2-LE SIE with decadal trends (6 panels)
# 
# | Panel | Content |
# |-------|-----------------------------------|
# | (a)   | NSIDC SIE time series + coloured decadal-trend segments |
# | (b)   | NSIDC decadal trend time series coloured by slowdown |
# | (c)   | CESM2-LE ensemble-mean SIE + coloured trend segments |
# | (d)   | CESM2-LE ensemble-mean trend + obs & model thresholds |
# | (e)   | All members (bg) + highlighted member with slowdown segments |
# | (f)   | All member trends (bg) + mean + thresholds + highlighted member |

# ===== CODE [13] =====
# ── Font sizes (GLOBAL CONTROL) ─────────────────────────────────────
large_text = 32
small_text = 24

fig1, axes = plt.subplots(nrows=3, ncols=2, figsize=(32, 24), constrained_layout=True)
axa, axb = axes[0, 0], axes[0, 1]
axc, axd = axes[1, 0], axes[1, 1]
axe, axf = axes[2, 0], axes[2, 1]

def style_axis(ax):
    ax.tick_params(axis='both', labelsize=small_text)

# ── Panel (a): NSIDC SIE + coloured decadal-trend segments ───────────────────
axa.plot(nsidc_seaice_years, nsidc_seaice, lw=5, color='black',
         label=f'NSIDC {VARNAME}')
plot_colored_decadal_segments(
    axa,
    years_window_start=nsidc_seaice_years,
    series=nsidc_seaice,
    slopes=nsidc_trends,
    slowdown_mask=nsidc_slowdown,
    window=10, lw=2.0,
    label_slowdown='decadal trend (yes, slowdown)',
    label_noslow='decadal trend (no slowdown)',
)
axa.set_ylabel(ice_ylabel(), fontsize=large_text)
style_axis(axa)
add_panel_label(axa, '(a)', fontsize=large_text)
axa.legend(loc='best', fontsize=small_text)

# ── Panel (b): NSIDC trend time series coloured by slowdown ──────────────────
axb.plot(nsidc_trend_years, nsidc_trends, lw=3, color='gray',
         label='decadal trend (obs)')
axb.scatter(
    nsidc_trend_years[nsidc_slowdown == 1], nsidc_trends[nsidc_slowdown == 1],
    s=48, color=COLOR_SLOWDOWN, label='yes, slowdown', zorder=3,
)
axb.scatter(
    nsidc_trend_years[nsidc_slowdown == 0], nsidc_trends[nsidc_slowdown == 0],
    s=48, color=COLOR_NO_SLOWDOWN, label='no slowdown', zorder=3,
)
axb.axhline(mean_trend_obs, lw=3, color=COLOR_THRESH_OBS,
            label='mean decadal trend')
axb.axhline(thr_slow, lw=5, color=COLOR_THRESH_OBS, ls='--',
            label=f'slowdown threshold μ+σ')
axb.set_ylabel(trend_ylabel(), fontsize=large_text)
style_axis(axb)
add_panel_label(axb, '(b)', fontsize=large_text)
axb.legend(loc='lower right', fontsize=small_text)

# ── Panels (c)–(d): CESM2-LE ensemble mean ───────────────────────────────────
if siei90 is not None:
    axc.plot(years90, sie_ens_mean, lw=5, color='black',
             label='CESM2-LE ensemble mean')
    plot_colored_decadal_segments(
        axc,
        years_window_start=years90,
        series=sie_ens_mean,
        slopes=linear_trends_mean,
        slowdown_mask=model_slow_mask_mean,
        window=10, lw=3.0,
        label_slowdown='decadal trend (yes, slowdown)',
        label_noslow='decadal trend (no slowdown)',
    )
    axc.set_ylabel(ice_ylabel(), fontsize=large_text)
    style_axis(axc)
    add_panel_label(axc, '(c)', fontsize=large_text)
    axc.legend(loc='best', fontsize=small_text)
else:
    axc.text(0.5, 0.5, 'Raw SIE not available', transform=axc.transAxes,
             ha='center', va='center', fontsize=large_text)

# ── Panel (d): ensemble-mean trend time series ───────────────────────────────
axd.plot(trend_years, linear_trends_mean, lw=4, color='gray',
         label='decadal trend (ens mean)')
axd.scatter(
    trend_years[model_slow_mask_mean == 1], linear_trends_mean[model_slow_mask_mean == 1],
    s=48, color=COLOR_SLOWDOWN, label='yes, slowdown', zorder=3,
)
axd.scatter(
    trend_years[model_slow_mask_mean == 0], linear_trends_mean[model_slow_mask_mean == 0],
    s=48, color=COLOR_NO_SLOWDOWN, label='no slowdown', zorder=3,
)
axd.axhline(thr_slow, lw=4, color=COLOR_THRESH_OBS, ls='--',
            label='obs slowdown threshold')
axd.plot(trend_years, threshold_slowdown_cesm, lw=4, color=COLOR_THRESH_MODEL,
         label='model slowdown threshold')
axd.set_ylabel(trend_ylabel(), fontsize=large_text)
style_axis(axd)
add_panel_label(axd, '(d)', fontsize=large_text)
axd.legend(loc='best', fontsize=small_text)

# ── Panels (e)–(f): all members + highlighted member ─────────────────────────
if siei90 is not None:
    for i in range(n_ens):
        axe.plot(years90, siei90[i, :], lw=0.2, color=COLOR_ENSEMBLE_BG)

    for j in range(n_trends_cesm):
        if sie_for_trend_windows is not None and not np.isfinite(sie_for_trend_windows[ENO, j]):
            continue
        dx = np.arange(10)
        y0 = siei90[ENO, j]
        m  = linear_trends_ens[ENO, j]
        axe.plot(years90[j : j + 10], m * dx + y0, lw=4.0, color=COLOR_SLOWDOWN)

    axe.plot(years90, sie_ens_mean, lw=5, color=COLOR_MEAN,
             label='CESM2-LE ensemble mean')
    axe.plot(years90, siei90[ENO, :], lw=2.5, color=COLOR_MEMBER,
             label=f'ensemble no. {ENO + 1}')
    if sie_for_trend_windows is not None:
        axe.plot(trend_years, sie_for_trend_windows[ENO, :], lw=0, marker='o', ms=4,
                 color=COLOR_SLOWDOWN, label=f'ensemble no. {ENO + 1} slowdowns')

    axe.set_ylabel(ice_ylabel(), fontsize=large_text)
    style_axis(axe)
    add_panel_label(axe, '(e)', fontsize=large_text)

    ens_handle = Line2D([], [], color=COLOR_ENSEMBLE_BG, lw=2, label='ensemble members')
    handles, labels = axe.get_legend_handles_labels()
    axe.legend(handles=[ens_handle] + handles, loc='best', fontsize=small_text)
else:
    axe.text(0.5, 0.5, 'Raw SIE not available', transform=axe.transAxes,
             ha='center', va='center', fontsize=large_text)

# ── Panel (f): all member trends as bg + mean + thresholds + highlighted member
for i in range(n_ens):
    axf.plot(trend_years, linear_trends_ens[i, :], lw=0.2, color=COLOR_ENSEMBLE_BG)

axf.plot(trend_years, linear_trends_mean, lw=3, color=COLOR_MEAN,
         label='ensemble mean')
axf.axhline(thr_slow, lw=4, color=COLOR_THRESH_OBS, ls='--',
            label='obs slowdown threshold')
axf.plot(trend_years, threshold_slowdown_cesm, lw=4, color=COLOR_THRESH_MODEL,
         label='model slowdown threshold')
axf.plot(trend_years, linear_trends_ens[ENO, :], lw=4, color=COLOR_MEMBER,
         label=f'ensemble no. {ENO + 1}')
axf.plot(trend_years, linear_trends_slowdown[ENO, :], lw=0, marker='o', ms=12,
         color=COLOR_SLOWDOWN, label=f'ensemble no. {ENO + 1} slowdowns')
axf.set_xlim([1990, 2100])
axf.set_ylabel(trend_ylabel(), fontsize=large_text)
style_axis(axf)
add_panel_label(axf, '(f)', fontsize=large_text)

ens_handle = Line2D([], [], color=COLOR_ENSEMBLE_BG, lw=2, label='ensemble members')
handles, labels = axf.get_legend_handles_labels()
axf.legend(handles=[ens_handle] + handles, loc='best', fontsize=small_text)

# fig1.suptitle(f'Figure 1 — {MONTH} {VARNAME}', fontsize=large_text, y=1.01)
plt.show()

# ===== MD [14] =====
# ---
# ## Figure 2 — Schematic + NSIDC + Ensemble member
# 
# - **Panel (a):** CNN schematic PNG (`PNG_PATH`; skipped if `None`)
# - **Panel (b):** NSIDC SIE time series with coloured trend segments  *(same as Fig 1a)*
# - **Panel (c):** CESM2-LE all-member background + highlighted member  *(same as Fig 1e)*

# ===== CODE [15] =====
from pathlib import Path

# ── Font sizes (GLOBAL CONTROL) ─────────────────────────────────────
large_text = 32
small_text = 20

# Decide layout depending on whether a PNG path is provided
_png_available = PNG_PATH is not None and Path(PNG_PATH).exists()

if _png_available:
    from PIL import Image
    fig2 = plt.figure(figsize=(36, 16))
    gs = fig2.add_gridspec(
        nrows=2, ncols=2,
        width_ratios=[2.1, 1.0],
        wspace=0.15, hspace=0.30
    )

    ax_a = fig2.add_subplot(gs[:, 0])
    ax_b = fig2.add_subplot(gs[0, 1])
    ax_c = fig2.add_subplot(gs[1, 1])

    img = Image.open(PNG_PATH)
    ax_a.imshow(img, aspect='auto')
    ax_a.axis('off')
    ax_a.set_box_aspect(img.size[1] / img.size[0])

    add_panel_label(ax_a, '(a)', fontsize=large_text)

else:
    if PNG_PATH is not None:
        print(f'PNG not found at {PNG_PATH} — showing 2-panel layout')

    fig2, (ax_b, ax_c) = plt.subplots(
        1, 2, figsize=(28, 10),
        constrained_layout=True
    )

# ── Panel (b): NSIDC SIE + coloured segments ────────────────────────
ax_b.plot(
    nsidc_seaice_years, nsidc_seaice,
    lw=3, color='black',
    label=f'NSIDC {VARNAME}'
)

plot_colored_decadal_segments(
    ax_b,
    years_window_start=nsidc_seaice_years,
    series=nsidc_seaice,
    slopes=nsidc_trends,
    slowdown_mask=nsidc_slowdown,
    window=10, lw=1.0,
    label_slowdown='decadal trend (yes, slowdown)',
    label_noslow='decadal trend (no slowdown)',
)

ax_b.set_ylabel(ice_ylabel(), fontsize=large_text)
ax_b.tick_params(axis='both', labelsize=small_text)

add_panel_label(ax_b, '(b)', fontsize=large_text)

ax_b.legend(loc='best', fontsize=small_text)

# ── Panel (c): Ensemble ─────────────────────────────────────────────
if siei90 is not None:

    for i in range(n_ens):
        ax_c.plot(
            years90, siei90[i, :],
            lw=0.2, color=COLOR_ENSEMBLE_BG
        )

    for j in range(n_trends_cesm):
        if (
            sie_for_trend_windows is not None and
            not np.isfinite(sie_for_trend_windows[ENO, j])
        ):
            continue

        dx = np.arange(10)
        y0 = siei90[ENO, j]
        m  = linear_trends_ens[ENO, j]

        ax_c.plot(
            years90[j : j + 10],
            m * dx + y0,
            lw=1.0,
            color=COLOR_SLOWDOWN
        )

    ax_c.plot(
        years90, sie_ens_mean,
        lw=3, color=COLOR_MEAN,
        label='CESM2-LE ensemble mean'
    )

    ax_c.plot(
        years90, siei90[ENO, :],
        lw=2.5, color=COLOR_MEMBER,
        label=f'ensemble no. {ENO + 1}'
    )

    if sie_for_trend_windows is not None:
        ax_c.plot(
            trend_years,
            sie_for_trend_windows[ENO, :],
            lw=0, marker='o', ms=4,
            color=COLOR_SLOWDOWN,
            label=f'ensemble no. {ENO + 1} slowdowns'
        )

    ens_handle = Line2D(
        [], [], color=COLOR_ENSEMBLE_BG,
        lw=2, label='ensemble members'
    )

    handles, labels = ax_c.get_legend_handles_labels()

    ax_c.legend(
        handles=[ens_handle] + handles,
        loc='best',
        fontsize=small_text
    )

else:
    ax_c.text(
        0.5, 0.5,
        'Raw SIE not available',
        transform=ax_c.transAxes,
        ha='center', va='center',
        fontsize=large_text
    )

ax_c.set_ylabel(ice_ylabel(), fontsize=large_text)
ax_c.tick_params(axis='both', labelsize=small_text)

add_panel_label(ax_c, '(c)', fontsize=large_text)

# Optional title
# fig2.suptitle(f'Figure 2 — {MONTH} {VARNAME}', fontsize=large_text, y=1.01)

if _png_available:
    fig2.subplots_adjust(top=0.95)

plt.show()

# ===== MD [16] =====
# ---
# ## Figure 3 — PDF / distribution of slowdown events (8 panels)
# 
# **Left column (a–d):** combined 1990–2090  
# **Right column (e–h):** split 1990–2039 vs 2040–2090
# 
# | Panel | Content |
# |-------|─────────|
# | (a)   | Histogram of slowdown events per member (all years combined) |
# | (b)   | Fraction of trend windows classified as slowdown vs normal |
# | (c)   | PDF of SIE values during slowdown vs normal windows |
# | (d)   | PDF of SIE anomalies during slowdown vs normal windows |
# | (e–h) | Same as (a–d) but overlaying 1990–2039 and 2040–2090 subsets |

# ===== CODE [17] =====
from matplotlib.ticker import FormatStrFormatter

# ── Font sizes (GLOBAL CONTROL) ─────────────────────────────────────
large_text = 36
small_text = 36

if siei90 is None:
    print('Raw SIE not available — Figure 3 requires siei90. Skipping.')
else:
    # Helper for consistent tick formatting
    def format_ticks(ax, xfmt='%.1f', yfmt='%.2f', size=small_text):
        ax.tick_params(axis='both', labelsize=size)
        ax.xaxis.set_major_formatter(FormatStrFormatter(xfmt))
        ax.yaxis.set_major_formatter(FormatStrFormatter(yfmt))

    # Data prep
    sie_windows = siei90[:, :n_trends_cesm]

    idx_split = int(np.where(trend_years == 2040)[0][0]) if 2040 in trend_years else 50
    subset_9039 = slice(0, idx_split)
    subset_4099 = slice(idx_split, None)

    fig3, axes3 = plt.subplots(
        nrows=4, ncols=2, figsize=(34, 44),
        constrained_layout=True
    )

    axa3, axe3 = axes3[0, 0], axes3[0, 1]
    axb3, axf3 = axes3[1, 0], axes3[1, 1]
    axc3, axg3 = axes3[2, 0], axes3[2, 1]
    axd3, axh3 = axes3[3, 0], axes3[3, 1]

    # ── (a)
    slowdown_per_member = slowdown_ens.sum(axis=1)
    max_val = int(slowdown_per_member.max())
    bins = np.arange(-1, max_val + 3)

    counts, edges = np.histogram(slowdown_per_member, bins=bins, density=True)

    axa3.bar(edges[:-1], counts, width=1.0,
             color=COLOR_THRESH_OBS, edgecolor='white', alpha=0.7)

    axa3.set_xlabel('Number of slowdown events per member', fontsize=large_text)
    axa3.set_ylabel('Frequency', fontsize=large_text)
    axa3.set_xticks(np.arange(0, max_val + 2, max(1, max_val // 10)))
    format_ticks(axa3, xfmt='%.0f', yfmt='%.2f', size=large_text)
    add_panel_label(axa3, '(a)', fontsize=large_text)

    # ── (b)
    flat = slowdown_ens.ravel()
    axb3.hist(flat, bins=[-0.5, 0.5, 1.5],
              weights=np.ones_like(flat) / len(flat), color='grey')

    axb3.set_xticks([0, 1])
    axb3.set_xticklabels(['no slowdown', 'slowdown'], fontsize=large_text)
    axb3.set_ylabel('fraction of events', fontsize=large_text)

    axb3.tick_params(axis='y', labelsize=large_text)
    axb3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    add_panel_label(axb3, '(b)', fontsize=large_text)

    # ── (c)
    mask_all = slowdown_ens
    d0 = sie_windows[mask_all == 0]
    d1 = sie_windows[mask_all == 1]

    axc3.hist(d0, bins=30, alpha=0.6, density=True,
              color=COLOR_NO_SLOWDOWN, label='no slowdown')
    axc3.hist(d1, bins=30, alpha=0.6, density=True,
              color=COLOR_SLOWDOWN, label='slowdown')

    axc3.set_xlabel(rf'{MONTH} {VARNAME} [M km$^2$]', fontsize=large_text)
    axc3.set_ylabel('Density', fontsize=large_text)
    axc3.legend(fontsize=large_text)

    format_ticks(axc3, size=small_text)
    add_panel_label(axc3, '(c)', fontsize=large_text)

    # ── (d)
    sie_anom = sie_windows - np.nanmean(sie_windows, axis=0)
    d0a = sie_anom[mask_all == 0]
    d1a = sie_anom[mask_all == 1]

    axd3.hist(d0a, bins=30, alpha=0.6, density=True,
              color=COLOR_NO_SLOWDOWN, label='no slowdown')
    axd3.hist(d1a, bins=30, alpha=0.6, density=True,
              color=COLOR_SLOWDOWN, label='slowdown')

    axd3.set_xlabel(rf'{MONTH} {VARNAME} anomaly [M km$^2$]', fontsize=large_text)
    axd3.set_ylabel('Density', fontsize=large_text)
    axd3.legend(fontsize=large_text)

    format_ticks(axd3, size=small_text)
    add_panel_label(axd3, '(d)', fontsize=large_text)

    # ── Right column prep
    slow_9039 = slowdown_ens[:, subset_9039]
    slow_4099 = slowdown_ens[:, subset_4099]
    sie_win_9039 = sie_windows[:, subset_9039]

    yr_start_9039 = trend_years[subset_9039][0]
    yr_end_9039 = trend_years[subset_9039][-1]

    # ── (e)
    spm_9039 = slow_9039.sum(axis=1)
    spm_4099 = slow_4099.sum(axis=1)

    max_val2 = int(np.concatenate([spm_9039, spm_4099]).max())
    bins2 = np.arange(-1, max_val2 + 3)

    cnts9039, edg9039 = np.histogram(spm_9039, bins=bins2, density=True)
    cnts4099, edg4099 = np.histogram(spm_4099, bins=bins2, density=True)

    axe3.bar(edg9039[:-1], cnts9039, width=1.0,
             color=COLOR_THRESH_OBS, edgecolor='white', alpha=0.7,
             label=f'{yr_start_9039}–{yr_end_9039}')
    axe3.bar(edg4099[:-1], cnts4099, width=1.0,
             color='red', edgecolor='white', alpha=0.2)

    axe3.set_xlabel('Number of slowdown events per member', fontsize=large_text)
    axe3.set_ylabel('Frequency', fontsize=large_text)
    axe3.set_xticks(np.arange(0, max_val2 + 2, max(1, max_val2 // 8)))
    axe3.legend(fontsize=large_text)

    format_ticks(axe3, xfmt='%.0f', size=small_text)
    add_panel_label(axe3, '(e)', fontsize=large_text)

    # ── (f)
    flat_sub = slow_9039.ravel()

    axf3.hist(flat_sub, bins=[-0.5, 0.5, 1.5],
              weights=np.ones_like(flat_sub) / len(flat_sub), color='grey')

    axf3.set_xticks([0, 1])
    axf3.set_xticklabels(['no slowdown', 'slowdown'], fontsize=small_text)
    axf3.set_ylabel('fraction of events', fontsize=large_text)
    axf3.set_title(f'({yr_start_9039}–{yr_end_9039})', fontsize=large_text)

    axf3.tick_params(axis='y', labelsize=small_text)
    axf3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    add_panel_label(axf3, '(f)', fontsize=large_text)

    # ── (g)
    d0g = sie_win_9039[slow_9039 == 0]
    d1g = sie_win_9039[slow_9039 == 1]

    axg3.hist(d0g, bins=30, alpha=0.6, density=True,
              color=COLOR_NO_SLOWDOWN, label='no slowdown')
    axg3.hist(d1g, bins=30, alpha=0.6, density=True,
              color=COLOR_SLOWDOWN, label='slowdown')

    axg3.set_xlabel(rf'{MONTH} {VARNAME} [M km$^2$]', fontsize=large_text)
    axg3.set_ylabel('Density', fontsize=large_text)
    axg3.set_title(f'({yr_start_9039}–{yr_end_9039})', fontsize=large_text)
    axg3.legend(fontsize=large_text)

    format_ticks(axg3, size=small_text)
    add_panel_label(axg3, '(g)', fontsize=large_text)

    # ── (h)
    anom_sub = sie_win_9039 - np.nanmean(sie_win_9039, axis=0)
    d0h = anom_sub[slow_9039 == 0]
    d1h = anom_sub[slow_9039 == 1]

    axh3.hist(d0h, bins=30, alpha=0.6, density=True,
              color=COLOR_NO_SLOWDOWN, label='no slowdown')
    axh3.hist(d1h, bins=30, alpha=0.6, density=True,
              color=COLOR_SLOWDOWN, label='slowdown')

    axh3.set_xlabel(rf'{MONTH} {VARNAME} anomaly [M km$^2$]', fontsize=large_text)
    axh3.set_ylabel('Density', fontsize=large_text)
    axh3.set_title(f'({yr_start_9039}–{yr_end_9039})', fontsize=large_text)
    axh3.legend(fontsize=large_text)

    format_ticks(axh3, size=small_text)
    add_panel_label(axh3, '(h)', fontsize=large_text)

    plt.show()

# ===== MD [18] =====
# ---
# ## Slowdown frequency diagnostics

# ===== CODE [19] =====
# ── Slowdown frequency ─────────────────────────────────────────────────────────
# Same metric as plotted in panels (b) and (f) above: fraction of
# trend windows classified as slowdown.

if siei90 is not None:
    # Overall frequency
    total_events  = slowdown_ens.sum()
    total_windows = slowdown_ens.size
    overall_freq  = total_events / total_windows
    print(f'Overall slowdown frequency: {total_events}/{total_windows} '
          f'= {overall_freq:.4f} ({overall_freq*100:.2f}%)')
    print()

    # Per-member frequency
    slowdown_per_member = slowdown_ens.sum(axis=1)   # (nens,)
    n_trend_windows = slowdown_ens.shape[1]
    freq_per_member = slowdown_per_member / n_trend_windows

    print(f'Per-member slowdown counts (out of {n_trend_windows} trend windows):')
    print(f'  Mean:   {slowdown_per_member.mean():.1f} events '
          f'({freq_per_member.mean()*100:.2f}%)')
    print(f'  Median: {np.median(slowdown_per_member):.0f} events')
    print(f'  Min:    {slowdown_per_member.min()} events')
    print(f'  Max:    {slowdown_per_member.max()} events')
    print(f'  Std:    {slowdown_per_member.std():.2f} events')
    print()

    # Early vs late period
    idx_split = int(np.where(trend_years == 2040)[0][0]) if 2040 in trend_years else 50
    early = slowdown_ens[:, :idx_split]
    late  = slowdown_ens[:, idx_split:]
    print(f'Early period ({trend_years[0]}–{trend_years[idx_split-1]}):')
    print(f'  Frequency: {early.sum()}/{early.size} = {early.mean()*100:.2f}%')
    print(f'Late period ({trend_years[idx_split]}–{trend_years[-1]}):')
    print(f'  Frequency: {late.sum()}/{late.size} = {late.mean()*100:.2f}%')
else:
    print('Raw SIE not available — cannot compute frequency.')


# ===== MD [20] =====
# ---
# ## SIE vs GMT diagnostic
# 
# Scatter plot of September SIE vs. global mean surface temperature (TREFHT)
# from CESM2-LE, using all ensemble members and all years.  Includes
# a hexbin density representation and a linear regression line.

# ===== CODE [21] =====
# ── SIE vs GMT scatter ─────────────────────────────────────────────────────────
# GMT data produced by scripts/01_cesm2le_preprocessing.py

GMT_DIR = paths.CESM2LE_TREF_DIR / 'gmt'

# Load GMT (same pattern as raw SIE: first50 + last50)
gmt_arrays = []
for group in ['first50', 'last50']:
    fpath = GMT_DIR / f'trefht_gmt_cesmle_{group}members_1990-2100.nc'
    if fpath.exists():
        ds_gmt = xr.open_dataset(fpath)
        gmt_arr = ds_gmt['trefht_gmt'].values.astype(np.float32)   # (nens, nyear)
        ds_gmt.close()
        gmt_arrays.append(gmt_arr)
        print(f'  Loaded {fpath.name}  shape={gmt_arr.shape}')
    else:
        print(f'  WARNING: not found — {fpath.name}')

if gmt_arrays and siei90 is not None:
    gmt_all = np.concatenate(gmt_arrays, axis=0)   # (100, 111)
    print(f'Combined GMT shape: {gmt_all.shape}')
    print(f'Combined SIE shape: {siei90.shape}')

    # Flatten for scatter
    gmt_flat = gmt_all.ravel()
    sie_flat = siei90.ravel()

    # Remove NaN
    valid = np.isfinite(gmt_flat) & np.isfinite(sie_flat)
    gmt_v = gmt_flat[valid]
    sie_v = sie_flat[valid]

    # Linear regression
    coeffs = np.polyfit(gmt_v, sie_v, 1)
    r = np.corrcoef(gmt_v, sie_v)[0, 1]
    fit_x = np.linspace(gmt_v.min(), gmt_v.max(), 100)
    fit_y = np.polyval(coeffs, fit_x)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Hexbin density
    hb = ax.hexbin(gmt_v, sie_v, gridsize=40, cmap='YlOrRd',
                   mincnt=1, linewidths=0.2, edgecolors='gray')
    cb = plt.colorbar(hb, ax=ax, label='Count')

    # Ensemble-mean relationship
    gmt_ens_mean = np.nanmean(gmt_all, axis=0)
    ax.plot(gmt_ens_mean, sie_ens_mean, color='black', linewidth=2.5,
            label='Ensemble mean', zorder=3)

    # Regression line
    ax.plot(fit_x, fit_y, color='steelblue', linewidth=2, linestyle='--',
            label=f'Linear fit (r = {r:.3f})', zorder=4)

    ax.set_xlabel(r'Global Mean Temperature [K]')
    ax.set_ylabel(ice_ylabel())
    ax.legend(loc='upper right')
    ax.set_title(f'{MONTH} {VARNAME} vs Global Mean Temperature — CESM2-LE')

    plt.tight_layout()
    plt.show()

    print(f'\nLinear regression: slope = {coeffs[0]:.4f}, intercept = {coeffs[1]:.2f}')
    print(f'Pearson r = {r:.4f}, R² = {r**2:.4f}')
    print(f'N = {len(gmt_v)} data points')
else:
    if not gmt_arrays:
        print('GMT data not found. Run scripts/01_cesm2le_preprocessing.py first.')
    if siei90 is None:
        print('Raw SIE not available.')

