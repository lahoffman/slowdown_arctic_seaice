#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------
#------------------------------------------------------
#set up environment
#------------------------------------------------------
#------------------------------------------------------

#modules
#------------------


#system
#------------------
import sys
import os
import csv
import numpy as np
import netCDF4 as nc
import pandas as pd
import xarray as xr
from netCDF4 import Dataset

#data processing
#------------------
#scipy
from scipy import stats, odr
from scipy.io import netcdf
from scipy.stats import norm
from datetime import datetime

#import functions
#------------------
sys.path.append('/home/elic/lhoffman/functions/')
from functions_general import ncdisp
from functions_general import movmean


#------------------------------------------------------
#------------------------------------------------------

#load data
# https://data-osdf.rda.ucar.edu/ncar/rda/d651056/CESM2-LE/ice/proc/tseries/month_1/aice/b.e21.BHISTcmip6.f09_g17.LE2-1001.001.cice.h.aice.185001-185912.nc

start_year = '1850'
end_year = '1859'
filepath = '/cofast/lhoffman/cesmle/aice/raw/b.e21.BHISTcmip6.f09_g17.LE2-1001.001.cice.h.aice.185001-185912.nc'
dataset = nc.Dataset(filepath,'r')
lat = np.array(dataset.variables['TLAT'])
lon = np.array(dataset.variables['TLON'])
aice = np.array(dataset.variables['aice'])
tarea = np.array(dataset.variables['tarea'])
datef = np.array(dataset.variables['time'])

#time
start_year = 1850
end_year = 1859
dates = pd.date_range(start=f'{start_year}-01', end=f'{end_year}-12', freq='MS')
dates_array = dates.to_pydatetime()
years = np.array([d.year for d in dates_array])
months = np.array([d.month for d in dates_array])
unique_years = np.unique(years)

fill_value = 1e+29
aice = np.where(np.abs(aice) > fill_value, np.nan, aice)

#------------------------------------------------------
#------------------------------------------------------
#AICE, MONTHLY
#------------------------------------------------------
#------------------------------------------------------
#save monthly
savepath = '/cofast/lhoffman/cesmle/aice/mon/BHISTcmip6_mon_1001.001_aice_185001-185912.nc'

# Create an xarray Dataset
ds = xr.Dataset(
    {
        "aice": (("nm", "nx", "ny"),aice),
        "lat":(("nx","ny"),lat),
        "lon":(("nx","ny"),lon),
    },
    coords={
        "nm": np.arange(aice.shape[0]), # months
        "nx": np.arange(aice.shape[1]),  # Latitude index
        "ny": np.arange(aice.shape[2]),  # Longitude index
        
    },
)

# Save to a NetCDF file
encoding = {var: {"zlib": True, "complevel": 4} for var in ds.data_vars}
ds.to_netcdf(savepath, format="NETCDF4", encoding=encoding)

print(f"NetCDF file saved to {savepath}")

#------------------------------------------------------

'''
#------------------------------------------------------
#------------------------------------------------------
#sea ice extent
#------------------------------------------------------
#------------------------------------------------------
ice_mask = aice >= 0.15
arctic_mask = lat >= 50
antarctic_mask = lat <= 50

arctic_mask_3d = np.broadcast_to(arctic_mask, aice.shape)
antarctic_mask_3d = np.broadcast_to(antarctic_mask, aice.shape)

arctic_ice = ice_mask & arctic_mask_3d     # shape: (time, nj, ni)
antarctic_ice = ice_mask & antarctic_mask_3d
tarea_3d = np.broadcast_to(tarea, aice.shape)

siextentn = (arctic_ice * tarea_3d).sum(axis=(1, 2)) / 1e12     # million km²
siextents = (antarctic_ice * tarea_3d).sum(axis=(1, 2)) / 1e12

#save siextent
savepath = '/cofast/lhoffman/cesmle/aice/siextent/BHISTcmip6_year_1001.001_siextent_185001-185912.nc'

# Create an xarray Dataset
ds = xr.Dataset(
    {
        "siextentn": (("nm", ),siextentn),
        "siextents": (("nm",),siextents),
    },
    coords={
        "nm": np.arange(siextentn.shape[0]), #months
        
    },
)

# Save to a NetCDF file
encoding = {var: {"zlib": True, "complevel": 4} for var in ds.data_vars}
ds.to_netcdf(savepath, format="NETCDF4", encoding=encoding)

print(f"NetCDF file saved to {savepath}")
#------------------------------------------------------
#------------------------------------------------------

#------------------------------------------------------
#------------------------------------------------------
#AICE, YEARLY
#------------------------------------------------------
#------------------------------------------------------

y0 = unique_years[0]
yf = unique_years[-1]+1

aice_year = []
time_year = []
for k in range(y0,yf):
    aicey = np.nanmean(aice[years==k,:,:],axis=0)[np.newaxis,:,:]
    timey = dates[years==k]
    
    if k == y0:
        aice_year = aicey
    else:
        aice_year = np.append(aice_year,aicey,axis=0)

    time_year.append(timey)
aice_yearly = np.array(aice_year)

#save yearly
savepath = '/cofast/lhoffman/cesmle/aice/year/BHISTcmip6_year_1001.001_aice_185001-185912.nc'

# Create an xarray Dataset
ds = xr.Dataset(
    {
        "aice_yearly": (("nm", "nx", "ny"),aice_yearly),
        "unique_years": (("nm",), unique_years),
        "lat":(("nx","ny"),lat),
        "lon":(("nx","ny"),lon),
    },
    coords={
        "nm": np.arange(aice_yearly.shape[0]), #months
        "nx": np.arange(aice_yearly.shape[1]),  # Latitude index
        "ny": np.arange(aice_yearly.shape[2]),  # Longitude index
        
    },
)

# Save to a NetCDF file
encoding = {var: {"zlib": True, "complevel": 4} for var in ds.data_vars}
ds.to_netcdf(savepath, format="NETCDF4", encoding=encoding)

print(f"NetCDF file saved to {savepath}")

#------------------------------------------------------
#------------------------------------------------------
'''

