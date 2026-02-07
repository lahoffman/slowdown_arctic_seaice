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

loadpath = rootpath+'D1_ersstv5_sst_regridded_to_cesm2_1854-2024.nc'
dataset = nc.Dataset(loadpath,'r')
sst_obs = np.array(dataset.variables['sst_obs'])

# Create monthly date array
dates = pd.date_range(start='1854-01-01', end='2024-12-31', freq='MS')

# Convert to numpy arrays
dates_np = dates.to_numpy()
years = dates.year.to_numpy()
months = dates.month.to_numpy()

# grab 1990 - 2024
y1990 = np.where(years==1990)[0][0]
ssti = sst_obs[y1990:,:,:]
dates1990 = dates[y1990:]

# jja mean
ntime,nlat,nlon = ssti.shape
nmon = 12
nyear = ntime//12
sst_reshape = ssti.reshape(nyear,nmon,nlat,nlon)
sst_JJA = np.nanmean(sst_reshape[1:,5:8,:,:],axis=1)

# mam mean
sst_MAM = np.nanmean(sst_reshape[1:,2:5,:,:],axis=1)

# djf mean
sst_JF = sst_reshape[1:,:2,:,:]
sst_D = sst_reshape[:-1,11,:,:][:,None,:,:]
sst_djf = np.concatenate((sst_D,sst_JF),axis=1)
sst_DJF = np.nanmean(sst_djf,axis=1)

#son mean
sst_SON = np.nanmean(sst_reshape[:-1,8:11,:,:],axis=1)

sst_seasonal = np.concatenate((sst_JJA[None,:,:,:],sst_MAM[None,:,:,:],sst_DJF[None,:,:,:],sst_SON[None,:,:,:]),axis=0)
years_seasonal = np.concatenate((np.arange(1991,2025)[None,:],np.arange(1991,2025)[None,:],np.arange(1991,2025)[None,:],np.arange(1990,2024)[None,:]),axis=0)
seasons = ['JJA','MAM','DJF','SON']

# remove forced (linear)
dx = np.arange(1990,2024)
sst_residual = np.empty_like(sst_seasonal)
for k in range(4):
    for i in range(nlat):
        for j in range(nlon):
            dy = sst_seasonal[k,:,:,:]
            coeff = np.polyfit(dx, dy[:, i, j], 1)
            y_fit = np.poly1d(coeff)(dx)
            sst_residual[k, :, i, j] = dy[:, i, j] - y_fit


# global standardization
sst_masked = np.where(landmask_3d == 0, sst_residual, np.nan)
miu = np.nanmean(sst_residual,axis=(1, 2, 3),keepdims=True)
sigma = np.nanstd(sst_residual,axis=(1, 2, 3),keepdims=True)
del sst_masked

sst_standardized = np.divide((sst_residual-miu),sigma)
sst_year = np.array(sst_standardized)
del sst_residual, sst_standardized


nt = 26
datasets = [
    {"data": sst_year[0,nt,:, :], "lat": lat, "lon": lon, "title": "SST in JJA+"},
    {"data": sst_year[1,nt,:, :], "lat": lat, "lon": lon, "title": "SST in MAM"},
    {"data": sst_year[2,nt,:, :], "lat": lat, "lon": lon, "title": "SST in DJF"},
    {"data": sst_year[3,nt,:, :], "lat": lat, "lon": lon, "title": "SST in SON"},
]

# Create figure and axes
fig, axes = plt.subplots(1, 4, figsize=(40, 10),
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


savepath = rootpath+'D2_ersstv5_sst_seasonal_TVT_1979-2024.nc'
ds = xr.Dataset(
    {

        "sst":(("nse","ntr","nx","ny"),sst_year),

    },
    coords={
        "nse":np.arange(sst_year.shape[0]), #seasons
        "ntr":np.arange(sst_year.shape[1]), #samples
        "nx":np.arange(sst_year.shape[2]), #lat index
        "ny":np.arange(sst_year.shape[3]), #lon index

    },
)

#save to NetCDF file
encoding = {var: {"zlib":True,"coJmplevel":4} for var in ds.data_vars}
ds.to_netcdf(savepath,format='NETCDF4')
print(f'NetCDF file saved to {savepath}')
