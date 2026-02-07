

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

'''
# --- first 50 members ---
loadpath = '/cofast/lhoffman/cesmle/aice/mon_combined/aice_cesmle_first50members_mon_199001-210012.nc'
dataset = nc.Dataset(loadpath,'r')
aicei = np.array(dataset.variables['aice'])

time = np.arange(np.datetime64('1990-01'),np.datetime64('2101-01'), np.timedelta64(1,'M'))
months = np.array([t.astype(object).month for t in time])

mon_labels = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
savepath_lead = '/cofast/lhoffman/cesmle/aice/mon/aice_cesmle_first50members_mon_'
savepath_tail = '_199001-210012.nc'

nems, ntime, nlat, nlon = aicei.shape

for i in range(1,13):
    
    idx = np.where(months == i)[0]
    aice_mon = aicei[:,idx,:,:]
    time_mon = time[idx]
    
    savepath = f"{savepath_lead}{mon_labels[i-1]}{savepath_tail}"

    # Create an xarray Dataset
    ds = xr.Dataset(
        {
            "aice_mon": (("nem", "nm", "nx", "ny"), aice_mon),
        },
        coords={
            "nem": np.arange(aice_mon.shape[0]),  # Ensemble members index
            "nm": np.arange(aice_mon.shape[1]),  #time: monthly, 1990-2100
            "nx": np.arange(aice_mon.shape[2]),  # Latitude index
            "ny": np.arange(aice_mon.shape[3]),  # Longitude index

        },
    )

    # Save to a NetCDF file
    encoding = {var: {"zlib": True, "complevel": 4} for var in ds.data_vars}
    ds.to_netcdf(savepath, format="NETCDF4", encoding=encoding)

    print(f"NetCDF file saved to {savepath}")
    
'''
 
    
# --- last 50 members ---
loadpath = '/cofast/lhoffman/cesmle/aice/mon_combined/aice_cesmle_last50members_mon_199001-210012.nc'
dataset = nc.Dataset(loadpath,'r')
aicei = np.array(dataset.variables['aice'])

time = np.arange(np.datetime64('1990-01'),np.datetime64('2101-01'), np.timedelta64(1,'M'))
months = np.array([t.astype(object).month for t in time])

mon_labels = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
savepath_lead = '/cofast/lhoffman/cesmle/aice/mon/aice_cesmle_last50members_mon_'
savepath_tail = '_199001-210012.nc'

nems, ntime, nlat, nlon = aicei.shape

for i in range(1,13):
    
    idx = np.where(months == i)[0]
    aice_mon = aicei[:,idx,:,:]
    time_mon = time[idx]
    
    savepath = f"{savepath_lead}{mon_labels[i-1]}{savepath_tail}"

    # Create an xarray Dataset
    ds = xr.Dataset(
        {
            "aice_mon": (("nem", "nm", "nx", "ny"), aice_mon),
        },
        coords={
            "nem": np.arange(aice_mon.shape[0]),  # Ensemble members index
            "nm": np.arange(aice_mon.shape[1]),  #time: monthly, 1990-2100
            "nx": np.arange(aice_mon.shape[2]),  # Latitude index
            "ny": np.arange(aice_mon.shape[3]),  # Longitude index

        },
    )

    # Save to a NetCDF file
    encoding = {var: {"zlib": True, "complevel": 4} for var in ds.data_vars}
    ds.to_netcdf(savepath, format="NETCDF4", encoding=encoding)

    print(f"NetCDF file saved to {savepath}")