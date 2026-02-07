#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------
#------------------------------------------------------
# ROOT PATH
rootpath = '/cofast/lhoffman/slowdown/'
#------------------------------------------------------
#------------------------------------------------------

#set up environment
#------------------------------------------------------
#------------------------------------------------------

#system
#------------------
import sys
import os
import csv
import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc
from netCDF4 import Dataset
import datetime

#data processing
#------------------
#scipy
from scipy import stats, odr
from scipy.io import netcdf
from scipy.stats import norm
from scipy.ndimage import uniform_filter1d

#plotting
#------------------
#matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cmocean

#import functions
#------------------
sys.path.append(rootpath+'functions/')
from functions_general import ncdisp
from functions_general import movmean

#------------------------------------------------------
#------------------------------------------------------

#------------------------------------------------------
# 0. LANDMASK
#------------------------------------------------------
#load landmask: zeros for ocean, ones for land [192,288]
loadpath = rootpath+'cnn_cesm2le_landmask.nc'
dataset = nc.Dataset(loadpath,'r')
landmask = dataset.variables['landmask'][:]
landmask_3d = landmask[None,:,:]
#------------------------------------------------------

#------------------------------------------------------
#------------------------------------------------------
# 0. LAT, LON
#------------------------------------------------------
#------------------------------------------------------
filepath = rootpath+'b.e21.BHISTcmip6.f09_g17.LE2-1001.001.cam.h1.Z200.18500101-18591231.nc'
dataset = nc.Dataset(filepath,'r')
lat = np.array(dataset.variables['lat'])
lon = np.array(dataset.variables['lon'])
#------------------------------------------------------

#------------------------------------------------------
#------------------------------------------------------
# I. SST
#------------------------------------------------------
#------------------------------------------------------

# Load SST obs regridded to cesm2le
loadpath = rootpath+'D1_ersstv5_sst_regridded_to_cesm2_1854-2024.nc'
dataset = nc.Dataset(loadpath,'r')
sst_obs = np.array(dataset.variables['sst_obs'])

# Create monthly date array
dates = pd.date_range(start='1854-01-01', end='2024-12-31', freq='MS')

# Convert to numpy arrays
dates_np = dates.to_numpy()
years = dates.year.to_numpy()
months = dates.month.to_numpy()
#------------------------------------------------------
#------------------------------------------------------



def _area_mean_monthly(sst, lat, lon, latmin, latmax, lonmin, lonmax):
    """
    sst: (time, nlat, nlon)
    lat, lon: 1D
    returns: (time,) monthly area-mean over box
    """
    # subset box
    lat_inds = np.where((lat >= latmin) & (lat <= latmax))[0]
    lon_inds = np.where((lon >= lonmin) & (lon <= lonmax))[0]
    sst_subset = sst[:, lat_inds][:, :, lon_inds]

    # cos(lat) weights (normalize over chosen lats)
    wlat = np.cos(np.deg2rad(lat[lat_inds]))
    wlat = wlat / np.nansum(wlat)
    w2d = wlat[:, None]  # (nlat_sub, 1)

    # weighted over lat → (time, nlon_sub)
    lat_weighted = np.nansum(sst_subset * w2d[None, :, :], axis=1)
    # mean over lon → (time,)
    ts = np.nanmean(lat_weighted, axis=1)
    return ts


def _monthly_anoms(ts, years, baseline=(1990, 2020)):
    """
    ts: (time,) monthly series
    years: (time,) year for each month
    returns: anomalies (time,) with monthly climatology removed over baseline
    """
    ntime = len(ts)
    months = np.arange(ntime) % 12  # assumes starts in Jan

    mask_base = (years >= baseline[0]) & (years <= baseline[1])

    clim = np.full(12, np.nan)
    for m in range(12):
        sel = mask_base & (months == m)
        if np.any(sel):
            clim[m] = np.nanmean(ts[sel])

    anoms = ts - clim[months]
    return anoms


def compute_enso_indices(sst_obs, lat, lon, years, baseline=(1990, 2020)):
    """
    Compute Niño3, Niño4, Niño3.4, and the 'cold tongue' (N_CT) and 'warm pool'
    (N_WP) indices defined as:

        N_CT = N3 - α N4
        N_WP = N4 - α N3

    where
        α = 2/5  if  N3 * N4 > 0
        α = 0    otherwise

    Everything is monthly, detrended the same way as your original fn.
    """
    # 1) (optional) trend removal on full field – you had sst_internal = sst_obs
    sst_internal = sst_obs  # keep as-is

    # 2) get monthly area means for the three standard boxes
    # Niño3.4: 5S–5N, 170W–120W  ->  (lat -5..5, lon 190..240E)
    n34_ts = _area_mean_monthly(sst_internal, lat, lon,
                                latmin=-5, latmax=5, lonmin=190, lonmax=240)

    # Niño3: 5S–5N, 150W–90W -> (lat -5..5, lon 210..270E)
    n3_ts = _area_mean_monthly(sst_internal, lat, lon,
                               latmin=-5, latmax=5, lonmin=210, lonmax=270)

    # Niño4: 5S–5N, 160E–150W -> (lat -5..5, lon 160..210E)
    n4_ts = _area_mean_monthly(sst_internal, lat, lon,
                               latmin=-5, latmax=5, lonmin=160, lonmax=210)

    # 3) monthly anomalies over baseline
    n34 = _monthly_anoms(n34_ts, years, baseline=baseline)
    n3  = _monthly_anoms(n3_ts,  years, baseline=baseline)
    n4  = _monthly_anoms(n4_ts,  years, baseline=baseline)

    # 4) linear-detrend each (like your original fn)
    t = np.arange(sst_obs.shape[0])  # monthly index
    def _detrend(x):
        coeffs = np.polyfit(t, x, 1)
        return x - np.polyval(coeffs, t)

    n34 = _detrend(n34)
    n3  = _detrend(n3)
    n4  = _detrend(n4)

    # 5) N_CT and N_WP:
    # α depends on the sign of N3 * N4 *at each time step*
    alpha = np.where(n3 * n4 > 0, 2.0/5.0, 0.0)  # (time,)

    n_ct = n3 - alpha * n4
    n_wp = n4 - alpha * n3
    
    # --- 5-month moving mean ---
    def smooth(x):
        return uniform_filter1d(x, size=5, mode="nearest")
    n34_s, n3_s, n4_s, nct_s, nwp_s = map(smooth, (n34, n3, n4, n_ct, n_wp))

    # --- normalize by baseline std ---
    base_mask = (years >= baseline[0]) & (years <= baseline[1])
    def norm(x):
        sigma = np.nanstd(x[base_mask])
        return x / sigma
    n34, n3, n4, n_ct, n_wp = map(norm, (n34_s, n3_s, n4_s, nct_s, nwp_s))

    return n3, n4, n34, n_ct, n_wp


def enso_phase_from_std(index_ts, thresh=0.4):
    """
    index_ts : 1D array (time,) already normalized (σ≈1)
    thresh   : number of std devs for El Niño / La Niña
    returns  : labels array of ints in {-1, 0, +1}
    """
    labels = np.zeros_like(index_ts, dtype=int)
    labels[index_ts >= thresh] = 1
    labels[index_ts <= -thresh] = -1
    return labels


nino3, nino4, nino34, nino_ct, nino_wp = compute_enso_indices(sst_obs,lat,lon,years)

nino3_labels = enso_phase_from_std(nino3)
nino4_labels = enso_phase_from_std(nino4)
nino34_labels = enso_phase_from_std(nino34)
nino_ct_labels = enso_phase_from_std(nino_ct)
nino_wp_labels = enso_phase_from_std(nino_wp)



'''
# SAVE
savepath = rootpath+'X1_ersstv_nino_indices_labels_1threshold_1854-2024.nc'
ds = xr.Dataset(
    {
        "nino34":(("nt",),nino34),
        "labels":(("nt",),labels),
        "nino34_jja":(("nyr","njja"),nino34_jja),
        "nino34_months":(("nyr","nm"),nino34_months),
        "nino_labels":(("nyear","nm"),nino_labels),
        "nino_labels_jja":(("nyear","njja"),nino_labels_jja),

    },
    coords={
        "nt":np.arange(nino34.shape[0]),
        "nyr":np.arange(nino34_months.shape[0]), #years
        "nm":np.arange(nino34_months.shape[1]), #months
        "njja":np.arange(nino34_jja.shape[1]), #jja = 3
    },
)

#save to NetCDF file
encoding = {var: {"zlib":True,"coJmplevel":4} for var in ds.data_vars}
ds.to_netcdf(savepath,format='NETCDF4')
print(f'NetCDF file saved to {savepath}')
'''

y1990 = np.where(years==1990)[0][0]
series = nino_ct[y1990:]    # time series
phases = nino_ct_labels[y1990:]         # -1, 0, +1
date = dates[y1990:]

fig, ax = plt.subplots(figsize=(12,4))

# Shade background by phase
start = 0
for i in range(1, len(phases)):
    if phases[i] != phases[start]:
        if phases[start] != 0:  # only shade warm/cool
            color = "red" if phases[start] == 1 else "blue"
            ax.axvspan(date[start], date[i-1], color=color, alpha=0.2)
        start = i
# last segment
if phases[start] != 0:
    color = "red" if phases[start] == 1 else "blue"
    ax.axvspan(date[start], date[-1], color=color, alpha=0.2)

# Plot the time series
ax.plot(date, series, color="black", linewidth=1.2)

# Zero line
ax.axhline(0, color="black", linewidth=0.8)

ax.set_xlabel("Year")
ax.set_ylabel("Niño3.4 index")
plt.show()
