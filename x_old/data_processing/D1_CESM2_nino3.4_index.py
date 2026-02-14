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


def compute_nino34_index(sst,lat,lon,years,baseline=(1990,2020)):
    
    # 1. Remove forced signal
    sst_internal = sst - sst.mean(axis=0,keepdims=True)
    
    # 2. Subset nino3.4 region [5S-5N, 170W-120W (190-240E)]
    lat_inds = np.where((lat >= -5) & (lat <= 5))[0]
    lon_inds = np.where((lon >= 190) & (lon <= 240))[0]
    sst_subset = sst_internal[:,:,lat_inds,:][:,:,:,lon_inds]
    
    # 3. Compute weights: cosine of latitude
    weights = np.cos(np.deg2rad(lat[lat_inds]))
    weights = weights / weights.sum()
    
    # 4. Area-weighted regional mean
    sst_weighted = (sst_subset * weights[None,None,:,None]).mean(axis=(2,3))
    
    # 5. Compute monthly climatology over fixed baseline
    mask_base = (years > baseline[0]) & (years <= baseline[1])
    clim = sst_weighted[:, mask_base].mean(axis=1) 
    
    # 6. Subtract baseline climatology
    nino34 = sst_weighted - clim[:,None]
            
    
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
filepathlead1 = '/cofast/lhoffman/cesmle/sst/mon/sst_cesmle_first50members_mon_'
filepathlead2 = '/cofast/lhoffman/cesmle/sst/mon/sst_cesmle_last50members_mon_'
filepathtail = '_185001-210012.nc'
mon = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']


sst_all = []
nino34_all = []
for i in range(12):
    
    #load & concatenate first and last 50 members
    loadpath1 = filepathlead1+mon[i]+filepathtail
    loadpath2 = filepathlead2+mon[i]+filepathtail

    dataset = nc.Dataset(loadpath1,'r')
    sst1 = dataset.variables['ssthm']
    unique_years = dataset.variables['unique_years']
    years = np.array(unique_years)
    y1990 = np.where(years==1990)[0][0]

    dataset = nc.Dataset(loadpath2,'r')
    sst2 = dataset.variables['ssthm']

    ssti12 = np.concatenate((sst1,sst2),axis=0)

    #calculate nino3.4 index
    nino34 = compute_nino34_index(ssti12,lat,lon,years)
    
    sst_all.append(ssti12)
    nino34_all.append(nino34)


year = np.arange(1850,2101)
yearmon = np.tile(year,(12,1))
yearmonreshape = yearmon.transpose(1,0)
yearmonrr = yearmonreshape.reshape(12*251,)

nino34a = np.array(nino34_all)
nmonths,nens,nyear = nino34a.shape

# reshape
ntime = nmonths*nyear
nino34_transpose = nino34a.transpose(1,2,0)
nino34_reshaped = nino34_transpose.reshape(nens,ntime)

# 5-month moving mean
nino34_smooth = uniform_filter1d(nino34_reshaped, size=5, axis=1, mode='nearest')

# normalize by standard deviation
baseline_mask = (yearmonrr >= 1990) & (yearmonrr <= 2020)
sigma_climatological = np.nanstd(nino34_smooth[:, baseline_mask], axis=1, keepdims=True)
nino34 = nino34_smooth / sigma_climatological

# subset JJA
yearmonths = yearmonrr.reshape(nyear,nmonths)
nino34_months = nino34.reshape(nens,nyear,nmonths)
nino34_jja = nino34_months[:,:,5:8]


# label months as El Nino, La Nina, Neutral
nino34_threshold = 1
min_length = 6
labels = np.zeros_like(nino34,dtype=int)

# for all months & years
for m in range(nens):
    ts = nino34[m,:]
    
    #El Nino
    mask_pos = ts >= nino34_threshold
    i = 0
    while i < ntime:
        if mask_pos[i]:
            j = i
            while j < ntime and mask_pos[j]:
                j += 1
            if j - i >= min_length:
                labels[m,i:j] = 1
            i = j
        else: 
            i += 1
            
    #La Nina
    mask_neg = ts <= -nino34_threshold
    i = 0
    while i < ntime:
        if mask_neg[i]:
            j = i
            while j < ntime and mask_neg[j]:
                j += 1
            if j - 1 >= min_length:
                labels[m,i:j] = -1
            i = j
        else: 
            i += 1

#reshape labels
nino_labels = labels.reshape(nens,nyear,nmonths)
nino_labels_jja = nino_labels[:,:,5:8]



# SAVE
savepath = rootpath+'D2_cesm2le_nino34_index_labels_1850-2100.nc'
ds = xr.Dataset(
    {

        "nino34_jja":(("nens","nyr","njja"),nino34_jja),
        "nino34_months":(("nens","nyr","nm"),nino34_months),
        "nino_labels":(("nens","nyear","nm"),nino_labels),
        "nino_labels_jja":(("nens","nyear","njja"),nino_labels_jja),

    },
    coords={
        "nens":np.arange(nino34_months.shape[0]), #ensemble members
        "nyr":np.arange(nino34_months.shape[1]), #years
        "nm":np.arange(nino34_months.shape[2]), #months
        "njja":np.arange(nino34_jja.shape[2]), #jja = 3
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
phases = nino_labels[0, :, 0]            # -1, 0, +1

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