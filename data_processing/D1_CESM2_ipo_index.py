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


def compute_ipo_index(sst,lat,lon,years,baseline=(1981,2010)):
    
    # 1. Remove forced signal
    sst_internal = sst - sst.mean(axis=0,keepdims=True)
    
    # 2. Subtract monthly climatology over baseline period
    mask_base = (years > baseline[0]) & (years <= baseline[1])
    clim = sst_internal[:, mask_base].mean(axis=1) 
    
    # 3. Area-weighted mean
    def box_mean(lat_range, lon_range):
        lat_inds = np.where((lat >= lat_range[0]) & (lat <= lat_range[1]))[0]
        lon_inds = np.where((lon >= lon_range[0]) & (lon <= lon_range[1]))[0]
        subset = sst_internal[:, :, lat_inds, :][:, :, :, lon_inds]
        weights = np.cos(np.deg2rad(lat[lat_inds]))
        weights = weights / weights.sum()
        return (subset * weights[None, None, :, None]).mean(axis=(2, 3))
    
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
filepathlead1 = '/cofast/lhoffman/cesmle/sst/mon/sst_cesmle_first50members_mon_'
filepathlead2 = '/cofast/lhoffman/cesmle/sst/mon/sst_cesmle_last50members_mon_'
filepathtail = '_185001-210012.nc'
mon = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']

years = np.arange(1850,2101)
y1990 = np.where(years==1990)[0][0]

sst_all = []
ipo_all = []
for i in range(12):
    
    #load & concatenate first and last 50 members
    loadpath1 = filepathlead1+mon[i]+filepathtail
    loadpath2 = filepathlead2+mon[i]+filepathtail

    with nc.Dataset(loadpath1, 'r') as ds1:
        ds1.set_auto_mask(False)   # avoid masked arrays
        sst1 = ds1.variables['ssthm'][:].astype('f4')  # whole variable into memory
        
    with nc.Dataset(loadpath2, 'r') as ds2:
        ds2.set_auto_mask(False)
        sst2 = ds2.variables['ssthm'][:].astype('f4')
    
    ssti = np.concatenate([sst1, sst2], axis=0)
    
    #calculate ipo index
    ipo = compute_ipo_index(ssti,lat,lon,years)
    
    sst_all.append(ssti)
    ipo_all.append(ipo)

ipo_unfiltered = np.array(ipo_all)
nmonths,nens,nyear = ipo_unfiltered.shape

# reshape
ntime = nmonths*nyear
ipo_transpose = ipo_unfiltered.transpose(1,2,0)
ipo_reshaped = ipo_transpose.reshape(nens,ntime)


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

ipo_filtered = chebyshev_lowpass(ipo_reshaped,cutoff_years=13)
ipo_filt_reshape = ipo_filtered.reshape(nens,nyear,nmonths)

#label months as cool / warm 
labels = np.zeros_like(ipo_filt_reshape,dtype=int)
labels[ipo_filt_reshape > 0] = 1
labels[ipo_filt_reshape < 0] = -1



# SAVE
savepath = rootpath+'D2_cesm2le_ipo_index_1850-2100.nc'
ds = xr.Dataset(
    {

        "ipo_unfiltered":(("nmon","nens","nyr"),ipo_unfiltered),
        "ipo_filt_reshape":(("nens","nyr","nmon"),ipo_filt_reshape),
        "labels":(("nens","nyr","nmon"),labels),

    },
    coords={
        "nens":np.arange(labels.shape[0]), #ensemble members
        "nyr":np.arange(labels.shape[1]), #years
        "nmon":np.arange(labels.shape[2]), #mon = 12
    },
)

#save to NetCDF file
encoding = {var: {"zlib":True,"coJmplevel":4} for var in ds.data_vars}
ds.to_netcdf(savepath,format='NETCDF4')
print(f'NetCDF file saved to {savepath}')


'''
import matplotlib.pyplot as plt
import numpy as np

years  = np.arange(1850, 2101)      # (251,)
series = nino34_months[0, :, 0]     # time series
phases = labels[0, :, 0]            # -1, 0, +1

fig, ax = plt.subplots(figsize=(12,4))

# Shade background by phase
start = 0
for i in range(1, len(phases)):
    if phases[i] != phases[start]:
        if phases[start] != 0:  # only shade warm/cool
            color = "red" if phases[start] == 1 else "blue"
            ax.axvspan(years[start], years[i-1], color=color, alpha=0.2)
        start = i
# last segment
if phases[start] != 0:
    color = "red" if phases[start] == 1 else "blue"
    ax.axvspan(years[start], years[-1], color=color, alpha=0.2)

# Plot the time series
ax.plot(years, series, color="black", linewidth=1.2)

# Zero line
ax.axhline(0, color="black", linewidth=0.8)

ax.set_xlabel("Year")
ax.set_ylabel("Niño3.4 index")
plt.show()

'''
