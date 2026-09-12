
# ===== MD [0] =====
# # FS10 — SIE vs GMT Slowdown Comparison
# 
# Compare slowdowns in September sea ice extent (SIE) versus yearly mean
# global mean temperature (GMT) across the CESM2-LE ensemble.
# 
# **Prerequisites**
# - `scripts/02_cesm2le_slowdowns.py` — SIE slowdown classification
# - `scripts/02_cesm2le_slowdowns_gmt.py` — GMT slowdown classification

# ===== CODE [1] =====
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt

PROJECT_ROOT = Path.cwd().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths

# ===== CODE [2] =====
# ── Configuration ──────────────────────────────────────────────────────────────
SIE_MONTH     = 'SEP'
START_YEAR    = 1990
END_YEAR      = 2100

mpl.rcParams.update({
    'font.size'        : 18,
    'axes.titlesize'   : 20,
    'axes.labelsize'   : 18,
    'xtick.labelsize'  : 16,
    'ytick.labelsize'  : 16,
    'legend.fontsize'  : 16,
    'figure.titlesize' : 22,
})

# ===== MD [3] =====
# ## Load SIE and GMT slowdown data

# ===== CODE [4] =====
# ── Load SIE slowdowns ─────────────────────────────────────────────────────────
sie_path = paths.cesm2le_slowdown_file('sie', SIE_MONTH, START_YEAR, END_YEAR)
print(f'SIE slowdown file: {sie_path}')
if not sie_path.exists():
    raise FileNotFoundError(
        f'SIE slowdown file not found: {sie_path}\n'
        f'Run scripts/02_cesm2le_slowdowns.py first.'
    )

ds_sie = xr.open_dataset(sie_path)
sie_slowdown = ds_sie['slowdown'].values      # (nens, n_trends)
sie_years    = ds_sie['nyr'].values            # trend window start years
ds_sie.close()

print(f'  SIE slowdown shape: {sie_slowdown.shape}')
print(f'  SIE year range    : {sie_years[0]}-{sie_years[-1]}')
print(f'  SIE slowdown freq : {sie_slowdown.sum()} / {sie_slowdown.size} '
      f'({100 * sie_slowdown.mean():.1f}%)')

# ===== CODE [5] =====
# ── Load GMT slowdowns ─────────────────────────────────────────────────────────
gmt_path = paths.cesm2le_gmt_slowdown_file(START_YEAR, END_YEAR)
print(f'GMT slowdown file: {gmt_path}')
if not gmt_path.exists():
    raise FileNotFoundError(
        f'GMT slowdown file not found: {gmt_path}\n'
        f'Run scripts/02_cesm2le_slowdowns_gmt.py first.'
    )

ds_gmt = xr.open_dataset(gmt_path)
gmt_slowdown = ds_gmt['slowdown'].values      # (nens, n_trends)
gmt_years    = ds_gmt['nyr'].values            # trend window start years
ds_gmt.close()

print(f'  GMT slowdown shape: {gmt_slowdown.shape}')
print(f'  GMT year range    : {gmt_years[0]}-{gmt_years[-1]}')
print(f'  GMT slowdown freq : {gmt_slowdown.sum()} / {gmt_slowdown.size} '
      f'({100 * gmt_slowdown.mean():.1f}%)')

# ===== MD [6] =====
# ## Align dimensions and compute BOTH

# ===== CODE [7] =====
# ── Align SIE and GMT on the same trend-window start years ─────────────────────
# Both should use the same window length and start year, but let's be safe.
common_years = np.intersect1d(sie_years, gmt_years)
print(f'Common year range: {common_years[0]}-{common_years[-1]} '
      f'({len(common_years)} windows)')

# Index into each array
sie_idx = np.isin(sie_years, common_years)
gmt_idx = np.isin(gmt_years, common_years)

sie_aligned = sie_slowdown[:, sie_idx]   # (nens, n_common)
gmt_aligned = gmt_slowdown[:, gmt_idx]   # (nens, n_common)

# Sanity check: ensemble and time dims must match
assert sie_aligned.shape == gmt_aligned.shape, (
    f'Shape mismatch after alignment: SIE {sie_aligned.shape} vs GMT {gmt_aligned.shape}'
)
nens, n_common = sie_aligned.shape
print(f'Aligned shape: ({nens}, {n_common})')

# ── BOTH: SIE slowdown AND GMT slowdown in same member/year ───────────────────
both_aligned = (sie_aligned == 1) & (gmt_aligned == 1)

# ── Count number of ensemble members with a slowdown per year ─────────────────
sie_count  = sie_aligned.sum(axis=0)    # (n_common,)
gmt_count  = gmt_aligned.sum(axis=0)    # (n_common,)
both_count = both_aligned.sum(axis=0)   # (n_common,)

print(f'\nPer-year member counts (mean +/- std):')
print(f'  SIE  : {sie_count.mean():.1f} +/- {sie_count.std():.1f}')
print(f'  GMT  : {gmt_count.mean():.1f} +/- {gmt_count.std():.1f}')
print(f'  BOTH : {both_count.mean():.1f} +/- {both_count.std():.1f}')

# ===== MD [8] =====
# ## Figure: SIE vs GMT slowdown frequency

# ===== CODE [9] =====
# ── Font sizes (GLOBAL CONTROL) ─────────────────────────────────────
label_text  = 42
title_text  = 46
tick_text   = 34
legend_text = 34

fig, ax = plt.subplots(figsize=(10*4, 6*4))

ax.plot(common_years, sie_count, color='lightblue', linewidth=10,
        label='September SIE slowdowns')
ax.plot(common_years, gmt_count, color='tomato', linewidth=10,
        label='Yearly GMT slowdowns')
ax.plot(common_years, both_count, color='silver', linewidth=10,
        label='Both (SIE & GMT)')

# Labels + title
ax.set_ylabel('Number of ensemble members with slowdowns', fontsize=label_text)

# Legend
ax.legend(loc='best', framealpha=0.9, fontsize=legend_text)

# Ticks
ax.tick_params(axis='both', labelsize=tick_text)

# Limits
ax.set_xlim(common_years[0], common_years[-1])
ax.set_ylim(bottom=0)

# Clean styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

# ===== MD [10] =====
# ## Summary statistics

# ===== CODE [11] =====
# Correlation between SIE and GMT slowdown counts
from scipy import stats

r, p = stats.pearsonr(sie_count, gmt_count)
print(f'Pearson r (SIE vs GMT counts): {r:.3f}  (p = {p:.2e})')

# What fraction of SIE slowdowns also have GMT slowdowns?
n_sie_total  = int(sie_aligned.sum())
n_both_total = int(both_aligned.sum())
print(f'\nSIE slowdowns that are also GMT slowdowns: '
      f'{n_both_total} / {n_sie_total} = {100 * n_both_total / n_sie_total:.1f}%')

# What fraction of GMT slowdowns also have SIE slowdowns?
n_gmt_total = int(gmt_aligned.sum())
print(f'GMT slowdowns that are also SIE slowdowns: '
      f'{n_both_total} / {n_gmt_total} = {100 * n_both_total / n_gmt_total:.1f}%')

# ===== MD [12] =====
# ## Joint PDF: GMT trend vs SIE trend

# ===== CODE [13] =====
from scipy.stats import gaussian_kde

# ── Load trend arrays from the slowdown files ─────────────────────────────────
ds_sie = xr.open_dataset(sie_path)
sie_trends = ds_sie['linear_trends_ens'].values   # (nens, n_trends), M km² yr⁻¹
sie_trend_years = ds_sie['nyr'].values
ds_sie.close()

ds_gmt = xr.open_dataset(gmt_path)
gmt_trends = ds_gmt['gmt_trends_ens'].values      # (nens, n_trends), K yr⁻¹
gmt_trend_years = ds_gmt['nyr'].values
ds_gmt.close()

# Align on common years
sie_tidx = np.isin(sie_trend_years, common_years)
gmt_tidx = np.isin(gmt_trend_years, common_years)
sie_tr = sie_trends[:, sie_tidx].ravel()
gmt_tr = gmt_trends[:, gmt_tidx].ravel()

# Drop NaNs
valid = ~(np.isnan(sie_tr) | np.isnan(gmt_tr))
x = gmt_tr[valid]   # GMT trend (K/yr)
y = sie_tr[valid]    # SIE trend (M km²/yr)
print(f'Joint sample size: {len(x):,}')

# ── Gaussian KDE ──────────────────────────────────────────────────────────────
xy = np.vstack([x, y])
kde = gaussian_kde(xy)

# Evaluate on a regular grid
xmin, xmax = np.percentile(x, [0.5, 99.5])
ymin, ymax = np.percentile(y, [0.5, 99.5])
xg = np.linspace(xmin, xmax, 200)
yg = np.linspace(ymin, ymax, 200)
Xg, Yg = np.meshgrid(xg, yg)
Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))

cf = ax.contourf(Xg, Yg, Z, levels=20, cmap='Blues')
ax.contour(Xg, Yg, Z, levels=20, colors='steelblue', linewidths=0.4, alpha=0.6)
plt.colorbar(cf, ax=ax, label='Density')

ax.axhline(0, color='k', linewidth=0.5, linestyle='--', alpha=0.4)
ax.axvline(0, color='k', linewidth=0.5, linestyle='--', alpha=0.4)

ax.set_xlabel('GMT decadal trend (K yr$^{-1}$)')
ax.set_ylabel('SIE decadal trend (M km$^2$ yr$^{-1}$)')
ax.set_title('Joint PDF of decadal GMT and September SIE trends')

# Annotate correlation
r_joint, p_joint = stats.pearsonr(x, y)
ax.text(0.03, 0.97, f'r = {r_joint:.3f}',
        transform=ax.transAxes, fontsize=14, va='top',
        bbox=dict(boxstyle='round', fc='white', alpha=0.8))

plt.tight_layout()
plt.show()

