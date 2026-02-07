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


def compute_ipo_index(sst_obs,lat,lon,years,baseline=(1990,2020)):
    
    # 1. Remove trend
    sst_internal = sst_obs
    nlat,nlon = sst_internal.shape[1],sst_internal.shape[2]
    
    # ---------- 4) Monthly climatology over the baseline period ----------
    # NOTE: with (start, end) inclusive; change the comparison if you want (start, end] etc.
    mask_base = (years >= baseline[0]) & (years <= baseline[1])

    ntime = years.shape[0]
    months = (np.arange(ntime) % 12) 

    # Obs monthly climatology: (12,)
    clim_obs = np.full((12,nlat,nlon), np.nan)

    for m in range(12):
        sel = mask_base & (months == m)
        if np.any(sel):
            clim_obs[m,:,:] = np.nanmean(sst_obs[sel],axis=0,keepdims=True)

    # ---------- 5) Subtract monthly climatology -> anomalies ----------
    obs_clim_by_time   = clim_obs[months]                  # (ntime,)
    sst_m_trend   = sst_obs - obs_clim_by_time           # (ntime,)
            
    
    def box_mean(lat_range, lon_range):
        lat_inds = np.where((lat >= lat_range[0]) & (lat <= lat_range[1]))[0]
        lon_inds = np.where((lon >= lon_range[0]) & (lon <= lon_range[1]))[0]
        subset = sst_m_trend[ :, lat_inds, :][ :, :, lon_inds]
        weights = np.cos(np.deg2rad(lat[lat_inds]))
        weights = weights / weights.sum()
        bm = np.nanmean(subset * weights[None, None, :, None], axis=(2, 3))
        return bm
    
    trop = box_mean((-10, 10), (170, 270))   # 170E–90W
    npac = box_mean((25, 45), (140, 215))   # 140E–145W
    spac = box_mean((-50, -15), (150, 200))
    
    ipo = trop - 0.5 * (npac + spac)
    
    return ipo

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
ipo = compute_ipo_index(sst_obs,lat,lon,years)

# 13-year Chebyshev Type-I low-pass filter (Henly et al. 2015)
from scipy.signal import cheby1, filtfilt

def chebyshev_lowpass(tpi, cutoff_years=13, order=4, rp=0.05):
    # sampling frequency
    f_sampling = 12.0
    f_nyquist = f_sampling/2.0
    
    # cutoff frequency (cycles per year --> cycles per month)
    f_cutoff = 1.0 / cutoff_years  #cycles per year
    fc_norm = (f_cutoff) / f_nyquist #normalized cutoff
    
    # Chebyshev type I low-pass filter
    b, a = cheby1(order, rp, fc_norm, btype='low')
    
    tpi_filtered = filtfilt(b, a, tpi, axis=1)
    
    return tpi_filtered

ipo_filtered = chebyshev_lowpass(ipo,cutoff_years=13)

#label months as cool / warm 
labels_filtered = np.zeros_like(ipo_filtered,dtype=int)
labels_filtered[ipo_filtered > 0] = 1
labels_filtered[ipo_filtered < 0] = -1

labels = np.zeros_like(ipo,dtype=int)
labels[ipo > 0] = 1
labels[ipo < 0] = -1

'''
# SAVE
savepath = rootpath+'X1_ersstv_ipo_index_labels_0threshold_1854-2024.nc'
ds = xr.Dataset(
    {

        "ipo":(("n1","nt"),ipo),
        "ipo_filtered":(("n1","nt"),ipo_filtered),
        "labels":(("n1","nt"),labels),
        "labels_filtered":(("n1","nt"),labels_filtered),

    },
    coords={
        "n1":np.arange(labels.shape[0]), #ensemble members
        "nt":np.arange(labels.shape[1]), #years
    },
)

#save to NetCDF file
encoding = {var: {"zlib":True,"coJmplevel":4} for var in ds.data_vars}
ds.to_netcdf(savepath,format='NETCDF4')
print(f'NetCDF file saved to {savepath}')
'''

y1990 = np.where(years==1990)[0][0]
date = dates[y1990:]

#------------------------------------------------------
# PLOT: timeseries: filtered IPO +/- phases
#------------------------------------------------------
series = ipo_filtered[0,y1990:]    # time series
phases = labels_filtered[0,y1990:]         # -1, 0, +1

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
ax.set_ylabel("IPO index")
plt.show()


#------------------------------------------------------
# PLOT: timeseries: un-filtered IPO +/- phases
#------------------------------------------------------
series = ipo[0,y1990:]    # time series
phases = labels[0,y1990:]         # -1, 0, +1


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
ax.set_ylabel("IPO index")
plt.show()

#------------------------------------------------------
# PLOT: map, SST composites in filtered IPO +/- phases
#------------------------------------------------------
sst_anomaly = sst_obs - np.nanmean(sst_obs,axis=0,keepdims=True)

sst_pos = np.nanmean(sst_anomaly[labels_filtered[0,:] ==  1], axis=0)
sst_neg = np.nanmean(sst_anomaly[labels_filtered[0,:] == -1], axis=0)


datasets = [
    {"data": sst_pos[:, :], "lat": lat, "lon": lon, "title": "SST in IPO+"},
    {"data": sst_neg[:, :],        "lat": lat, "lon": lon, "title": "SST in IPO-"},
]

# Create figure and axes
fig, axes = plt.subplots(1, 2, figsize=(20, 10),
                         subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

# Loop through datasets and plot
for ax, ds in zip(axes, datasets):
    ax.set_global()
    ax.coastlines()

    im = ax.pcolormesh(ds["lon"], ds["lat"], ds["data"],
                       cmap=cmocean.cm.balance,
                       shading='auto',
                       transform=ccrs.PlateCarree())
    
    im.set_clim(-1,1)

    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.xaxis.set_tick_params(rotation=45)
    ax.yaxis.set_tick_params(rotation=45)
    ax.set_title(ds["title"])

# Adjust layout manually to make space for colorbar
fig.subplots_adjust(bottom=0.05)

# Shared horizontal colorbar below both plots
cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.03])  # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', pad=0.05)
cbar.set_label('SST')
#------------------------------------------------------