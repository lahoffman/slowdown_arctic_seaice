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


def compute_nino34_index(sst_obs,lat,lon,years,baseline=(1990,2020)):
    
    # 1. Remove trend
    sst_internal = sst_obs
    
    # 2. Subset nino3.4 region [5S-5N, 170W-120W (190-240E)]
    lat_inds = np.where((lat >= -5) & (lat <= 5))[0]
    lon_inds = np.where((lon >= 190) & (lon <= 240))[0]
    sst_subset = sst_internal[:,lat_inds,:][:,:,lon_inds]
    
    # ---------- 2) Cos(latitude) weights (normalize over latitude only) ----------
    wlat = np.cos(np.deg2rad(lat[lat_inds]))                                   # (nlat*,)
    wlat = wlat / np.nansum(wlat)                                              # sum to 1 over selected lats
    w2d  = wlat[:, None]                                                       # (nlat*, 1) broadcast over lon

    # ---------- 3) Area-weighted regional mean ----------
    # For obs: weighted sum over lat, then simple mean over lon
    obs_lat_sum = np.nansum(sst_subset * w2d[None, :, :], axis=1)          # (ntime, nlon*)
    sst_weighted_obs = np.nanmean(obs_lat_sum, axis=1)                         # (ntime,)

    # ---------- 4) Monthly climatology over the baseline period ----------
    # NOTE: with (start, end) inclusive; change the comparison if you want (start, end] etc.
    mask_base = (years >= baseline[0]) & (years <= baseline[1])

    ntime = years.shape[0]
    months = (np.arange(ntime) % 12)  # 0..11; assumes series is monthly & starts in January
    # If your series doesn't start in January, build `months` from real timestamps.

    # Obs monthly climatology: (12,)
    clim_obs = np.full(12, np.nan)


    for m in range(12):
        sel = mask_base & (months == m)
        if np.any(sel):
            clim_obs[m] = np.nanmean(sst_weighted_obs[sel])

    # ---------- 5) Subtract monthly climatology -> anomalies ----------
    obs_clim_by_time   = clim_obs[months]                  # (ntime,)

    sst_m_clim   = sst_weighted_obs - obs_clim_by_time           # (ntime,)

    # 7. Compute linear trend
    coefficients = np.polyfit(np.arange(sst_obs.shape[0]), sst_m_clim, 1)
    polynomial = np.poly1d(coefficients)
    x_fit = np.arange(sst_obs.shape[0])
    y_fit = polynomial(x_fit)
            
    # 8. Subtract linear trend
    nino34 = sst_m_clim - y_fit
    
    return nino34

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
        
# Calculate nino3.4 index
nino34 = compute_nino34_index(sst_obs,lat,lon,years)

# 5-month moving mean
nino34_smooth = uniform_filter1d(nino34, size=5, axis=0, mode='nearest')

# Normalize by standard deviation
baseline_mask = (years >= 1990) & (years <= 2020)
sigma_climatological = np.nanstd(nino34_smooth[baseline_mask], axis=0, keepdims=True)
nino34 = nino34_smooth / sigma_climatological

# time indices
nyear = np.arange(1854,2025).shape[0]
nmonths = 12
ntime = dates_np.shape[0]

# label months as El Nino, La Nina, Neutral
nino34_threshold = 0.4
min_length = 6
labels = np.zeros_like(nino34,dtype=int)


#El Nino
mask_pos = nino34 >= nino34_threshold
i = 0
while i < ntime:
    if mask_pos[i]:
        j = i
        while j < ntime and mask_pos[j]:
            j += 1
        if j - i >= min_length:
            labels[i:j] = 1
        i = j
    else: 
        i += 1
        
#La Nina
mask_neg = nino34 <= -nino34_threshold
i = 0
while i < ntime:
    if mask_neg[i]:
        j = i
        while j < ntime and mask_neg[j]:
            j += 1
        if j - 1 >= min_length:
            labels[i:j] = -1
        i = j
    else: 
        i += 1

#reshape labels
nino_labels = labels.reshape(nyear,nmonths)
nino_labels_jja = nino_labels[:,5:8]

# subset JJA
yearmonths = years.reshape(nyear,nmonths)
nino34_months = nino34.reshape(nyear,nmonths)
nino34_jja = nino34_months[:,5:8]

'''
# SAVE
savepath = rootpath+'X1_ersstv_nino34_index_labels_0.4threshold_1854-2024.nc'
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
series = nino34[y1990:]    # time series
phases = labels[y1990:]         # -1, 0, +1
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
