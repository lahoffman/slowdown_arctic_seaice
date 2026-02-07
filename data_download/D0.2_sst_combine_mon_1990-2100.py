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
# LOAD DATA
#------------------------------------------------------

loadpath = '/cofast/lhoffman/cesmle/sst/raw/b.e21.BHISTcmip6.f09_g17.LE2-1001.001.cam.h0.SST.190001-190912.nc'
start_years = ['1990','2000','2010','2015','2025','2035','2045','2055','2065','2075','2085','2095']
end_years = ['1999','2009','2014','2024','2034','2044','2054','2064','2074','2084','2094','2100']
unique_years = np.arange(1990,2101)

#members 1-50
cmip6_members=[1001.001, 1021.002, 1041.003, 1061.004, 1081.005, 1101.006, 1121.007, 1141.008, 1161.009, '1181.010',
              1231.001, 1231.002, 1231.003, 1231.004, 1231.005, 1231.006, 1231.007, 1231.008, 1231.009, '1231.010',
              1251.001, 1251.002, 1251.003, 1251.004, 1251.005, 1251.006, 1251.007, 1251.008, 1251.009, '1251.010', 
              1281.001, 1281.002, 1281.003, 1281.004, 1281.005, 1281.006, 1281.007, 1281.008, 1281.009, '1281.010', 
              1301.001, 1301.002, 1301.003, 1301.004, 1301.005, 1301.006, 1301.007, 1301.008, 1301.009, '1301.010']

#members 51-100
smbb_members=[1231.011, 1231.012, 1231.013, 1231.014, 1231.015, 1231.016, 1231.017, 1231.018, 1231.019, '1231.020',
              1251.011, 1251.012, 1251.013, 1251.014, 1251.015, 1251.016, 1251.017, 1251.018, 1251.019, '1251.020',
              1281.011, 1281.012, 1281.013, 1281.014, 1281.015, 1281.016, 1281.017, 1281.018, 1281.019, '1281.020',
              1301.011, 1301.012, 1301.013, 1301.014, 1301.015, 1301.016, 1301.017, 1301.018, 1301.019, '1301.020',
              1011.001, 1031.002, 1051.003, 1071.004, 1091.005, 1111.006, 1131.007, 1151.008, 1171.009, '1191.010']


#------------------------------------------------------
#------------------------------------------------------
# --- members 1-50 ---
#------------------------------------------------------
#------------------------------------------------------
sstij = []
for i in range(50):
    for j in range(12):
        if j < 3:
            filepath = '/cofast/lhoffman/cesmle/sst/raw/b.e21.BHISTcmip6.f09_g17.LE2-{}.cam.h0.SST.{}01-{}12.nc'.format(cmip6_members[i],start_years[j],end_years[j])
        else: 
            filepath = '/cofast/lhoffman/cesmle/sst/raw/b.e21.BSSP370cmip6.f09_g17.LE2-{}.cam.h0.SST.{}01-{}12.nc'.format(cmip6_members[i],start_years[j],end_years[j])
        
        dataset = nc.Dataset(filepath,'r')
        ssti = np.array(dataset.variables['sst'])
        
        if j == 0:
            sstj = ssti
        else:
            sstj = np.concatenate((sstj,ssti),axis=0)
    sstij.append(sstj)
sst = np.array(sstij)



#------------------------------------------------------
# save to .nc file
#------------------------------------------------------
savepath = '/cofast/lhoffman/cesmle/sst/mon_combined/sst_cesmle_first50members_mon_199001-210012.nc'

# Create an xarray Dataset
ds = xr.Dataset(
    {
        "sst": (("nem", "nm", "nx", "ny"), sst),
        "unique_years": (("nyr",), unique_years),
    },
    coords={
        "nem": np.arange(sst.shape[0]),  # Ensemble members index
        "nm": np.arange(sst.shape[1]),  #time: monthly, 1990-2100
        "nx": np.arange(sst.shape[2]),  # Latitude index
        "ny": np.arange(sst.shape[3]),  # Longitude index
        "nyr": np.arange(unique_years.shape[0]), #years
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
# --- members 51-100 ---
#------------------------------------------------------
#------------------------------------------------------

sstij = []
for i in range(50):
    for j in range(12):
        if j < 3:
            filepath = '/cofast/lhoffman/cesmle/sst/raw/b.e21.BHISTsmbb.f09_g17.LE2-{}.cam.h0.SST.{}01-{}12.nc'.format(cmip6_members[i],start_years[j],end_years[j])
        else: 
            filepath = '/cofast/lhoffman/cesmle/sst/raw/b.e21.BSSP370smbb.f09_g17.LE2-{}.cam.h0.SST.{}01-{}12.nc'.format(cmip6_members[i],start_years[j],end_years[j])
        
        dataset = nc.Dataset(filepath,'r')
        ssti = np.array(dataset.variables['sst'])
        
        if j == 0:
            sstj = ssti
        else:
            sstj = np.concatenate((sstj,ssti),axis=0)
    sstij.append(sstj)
sst = np.array(sstij)


#------------------------------------------------------
# save to .nc file
#------------------------------------------------------
savepath = '/cofast/lhoffman/cesmle/sst/mon_combined/sst_cesmle_last50members_mon_199001-210012.nc'

# Create an xarray Dataset
ds = xr.Dataset(
    {
        "sst": (("nem", "nm", "nx", "ny"), sst),
        "unique_years": (("nyr",), unique_years),
    },
    coords={
        "nem": np.arange(sst.shape[0]),  # Ensemble members index
        "nm": np.arange(sst.shape[1]),  #time: monthly, 1990-2100
        "nx": np.arange(sst.shape[2]),  # Latitude index
        "ny": np.arange(sst.shape[3]),  # Longitude index
        "nyr": np.arange(unique_years.shape[0]), #years
    },
)

# Save to a NetCDF file
encoding = {var: {"zlib": True, "complevel": 4} for var in ds.data_vars}
ds.to_netcdf(savepath, format="NETCDF4", encoding=encoding)

print(f"NetCDF file saved to {savepath}")
#------------------------------------------------------
#------------------------------------------------------