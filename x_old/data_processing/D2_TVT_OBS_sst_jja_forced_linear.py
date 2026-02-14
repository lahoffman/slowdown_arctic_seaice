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
sst_JJA = np.nanmean(sst_reshape[:,5:8,:,:],axis=1)

# remove forced (linear)
dx = np.arange(1990,2025)
dy = sst_JJA
sst_residual = np.empty_like(dy)
for i in range(nlat):
    for j in range(nlon):
        coeff = np.polyfit(dx, dy[:, i, j], 1)
        y_fit = np.poly1d(coeff)(dx)
        sst_residual[:, i, j] = dy[:, i, j] - y_fit


# global standardization
sst_masked = np.where(landmask_3d == 0, sst_residual, np.nan)
miu = np.nanmean(sst_residual,axis=(0, 1, 2),keepdims=True)
sigma = np.nanstd(sst_residual,axis=(0, 1, 2),keepdims=True)
del sst_masked

sst_standardized = np.divide((sst_residual-miu),sigma)
sst_year = np.array(sst_standardized)
del sst_residual, sst_standardized

savepath = rootpath+'D2_ersstv5_sstJJA_TVT_1979-2024.nc'
ds = xr.Dataset(
    {

        "sst":(("ntr","nx","ny"),sst_year),

    },
    coords={
        "ntr":np.arange(sst_year.shape[0]), #samples
        "nx":np.arange(sst_year.shape[1]), #lat index
        "ny":np.arange(sst_year.shape[2]), #lon index

    },
)

#save to NetCDF file
encoding = {var: {"zlib":True,"coJmplevel":4} for var in ds.data_vars}
ds.to_netcdf(savepath,format='NETCDF4')
print(f'NetCDF file saved to {savepath}')
