#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------
#------------------------------------------------------
# ROOT PATH, SST composites
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

'''
#------------------------------------------------------
#------------------------------------------------------
# I. SST
#------------------------------------------------------
#------------------------------------------------------

#model file paths
#------------------------------------------------------

filepathlead1 = '/cofast/lhoffman/cesmle/sst/mon/sst_cesmle_first50members_mon_'
filepathlead2 = '/cofast/lhoffman/cesmle/sst/mon/sst_cesmle_last50members_mon_'
filepathtail = '_199001-210012.nc'
mon = ['JUN','JUL','AUG','MAR','APR','MAY','DEC','JAN','FEB','SEP','OCT','NOV']
mon_index = np.reshape(np.arange(12),(4,3))

sst_season = []
for j in range(4):
    sst_season_i = []
    for i in range(3): 
        
        mi = mon[mon_index[j,i]]
        
        #load & concatenate first and last 50 members
        loadpath1 = filepathlead1+mi+filepathtail
        loadpath2 = filepathlead2+mi+filepathtail

        with nc.Dataset(loadpath1, 'r') as ds1, nc.Dataset(loadpath2, 'r') as ds2:
            years  = np.arange(1990,2101)
            sst1   = ds1.variables['sst_mon'][:, :, :, :]           # (nens1, nyear, nlat, nlon)
            sst2   = ds2.variables['sst_mon'][:, :, :, :]           # (nens2, nyear, nlat, nlon)

        y1990 = int(np.where(years == 1990)[0][0])
        y2041 = int(np.where(years == 2041)[0][0])

        ssti12 = np.concatenate((sst1[:, :y2041, ...], sst2[:, :y2041, ...]), axis=0)  # (nens, nyear, nlat, nlon)

        if (j == 2) and (i == 0):   #DEC
            ssti = ssti12[:,y1990:-1,:,:]
            
        elif (j == 2) and (i in (1, 2)): #JAN, FEB
            ssti = ssti12[:,y1990+1:,:,:]
        
        elif j == 3: #SON
            ssti = ssti12[:,y1990:-1,:,:]
            
        else: #JJA, MAM
            ssti = ssti12[:,y1990+1:,:,:]
            
        sst_season_i.append(ssti)
        sst_season_j = np.nanmean(sst_season_i,axis=0)
    
    sst_season.append(sst_season_j) # [nseason,nens,nyears,nx,ny]
    
    del sst_season_i, sst_season_j, ssti, ssti12, sst2, sst1


# remove ensemble mean
sst_ensemble_mean = np.nanmean(sst_season,axis=1,keepdims=True)
sst_demeaned = sst_season-sst_ensemble_mean

del sst_season, sst_ensemble_mean

#reshape for ensemble member grouping with different forcing
nt = sst_demeaned.shape[2]
nx = sst_demeaned.shape[3]
ny = sst_demeaned.shape[4]
sst = sst_demeaned.reshape(4,10,10,nt,nx,ny)

del sst_demeaned

#train-test split(80-10-10 for t-v-t)
#training
sst_tri = sst[:,:,2:,:,:,:]
sst_trr = sst_tri.reshape(4,80,nt,nx,ny)
sst_tr = sst_trr.reshape(4,80*nt,nx,ny)

del sst_tri, sst_trr

#validation
sst_vali =  sst[:,:,1,:,:,:]
sst_var = sst_vali.reshape(4,10,nt,nx,ny)
sst_va = sst_var.reshape(4,10*nt,nx,ny)

del sst_vali, sst_var

#testing
sst_tei =  sst[:,:,0,:,:,:]
sst_ter = sst_tei.reshape(4,10,nt,nx,ny)
sst_te = sst_ter.reshape(4,10*nt,nx,ny)

del sst_tei, sst_ter, sst

#training statistics
sst_tr_masked = np.where(landmask_3d == 0, sst_tr, np.nan)
miu_train = np.nanmean(sst_tr_masked,axis=(1, 2, 3),keepdims=True)
sigma_train = np.nanstd(sst_tr_masked,axis=(1, 2, 3),keepdims=True)

del sst_tr_masked

sst_tr_standardized = np.divide((sst_tr-miu_train),sigma_train)
sst_va_standardized = np.divide((sst_va-miu_train),sigma_train)
sst_te_standardized = np.divide((sst_te-miu_train),sigma_train)

del sst_tr,sst_te,sst_va

sst_tr_year = np.array(sst_tr_standardized)
sst_va_year = np.array(sst_va_standardized)
sst_te_year = np.array(sst_te_standardized)

del sst_tr_standardized,sst_te_standardized, sst_va_standardized

savepath = rootpath+'D2_cesm2le_sst_seasonal_TVT_1990-2040.nc'
ds = xr.Dataset(
    {

        "sst_tr":(("nse","ntr","nx","ny"),sst_tr_year),
        "sst_va":(("nse","nva","nx","ny"),sst_va_year),
        "sst_te":(("nse","nte","nx","ny"),sst_te_year),

    },
    coords={
        "nse":np.arange(sst_tr_year.shape[0]), #seasons
        "ntr":np.arange(sst_tr_year.shape[1]), #training samples
        "nva":np.arange(sst_va_year.shape[1]), #validation samples
        "nte":np.arange(sst_te_year.shape[1]), #test samples
        "nx":np.arange(sst_tr_year.shape[2]), #lat index
        "ny":np.arange(sst_tr_year.shape[3]), #lon index

    },
)

#save to NetCDF file
encoding = {var: {"zlib":True,"coJmplevel":4} for var in ds.data_vars}
ds.to_netcdf(savepath,format='NETCDF4')
print(f'NetCDF file saved to {savepath}')

#del sst_tr_year, sst_te_year, sst_va_year


#------------------------------------------------------
#------------------------------------------------------


#------------------------------------------------------
#------------------------------------------------------
# IV. SLOWDOWN
#------------------------------------------------------
#------------------------------------------------------

loadpath = rootpath+'sie_nsidc_linear_decadal_slowdown_1990-2024.nc'
dataset = nc.Dataset(loadpath,'r')
sieij = np.array(dataset.variables['slowdown'])

# grab 1990 - 2040
years = np.arange(1990,2091)
y2041 = np.where(years==2041)[0][0]
sie = sieij[:,1:y2041]
nt = sie.shape[1]

del sieij

#train-test split, (80-10-10 for t-v-t)
siei = sie.reshape(10,10,nt)
sie_tri = siei[:,2:,:]
sie_vai = siei[:,1,:]
sie_tei = siei[:,0,:]

#reshape
sie_tr = sie_tri.reshape(-1)
sie_va = sie_vai.reshape(-1)
sie_te = sie_tei.reshape(-1)

#------------------------------------------------------
#------------------------------------------------------
# V. time
#------------------------------------------------------
#------------------------------------------------------
#load & concatenate first and last 50 members
loadpath1 = rootpath+'z200_cesmle_first50members_year_190001-210012.nc'
dataset = nc.Dataset(loadpath1,'r')
time = np.array(dataset.variables['time'])

years = np.arange(1990,2101)
y2041 = np.where(years==2041)[0][0]
time = time[1:y2041]
nt = time.shape[0]
timeti = np.tile(time[:, np.newaxis], (1, 100)).T
timet = timeti.reshape(10,10,nt)

time_tei = timet[:,0,:]
time_vai = timet[:,1,:]
time_tri = timet[:,2:,:]

time_tr = time_tri.reshape(-1)
time_va = time_vai.reshape(-1)
time_te = time_tei.reshape(-1)

#------------------------------------------------------
# save
#------------------------------------------------------

savepath = rootpath+'D2_cnn_cesm2le_sieTREND_seasons_TVT_1990-2100.nc'
ds = xr.Dataset(
    {
        "time_tr_slowdown":(("ntr",),time_tr),
        "time_va_slowdown":(("nva",),time_va),
        "time_te_slowdown":(("nte",),time_te),
        "sie_tr_slowdown":(("ntr",),sie_tr),
        "sie_va_slowdown":(("nva",),sie_va),
        "sie_te_slowdown":(("nte",),sie_te),

    },
    coords={
        "ntr":np.arange(sie_tr.shape[0]), #training samples
        "nva":np.arange(sie_va.shape[0]), #validation samples
        "nte":np.arange(sie_te.shape[0]), #test samples

    },
)

#save to NetCDF file
encoding = {var: {"zlib":True,"coJmplevel":4} for var in ds.data_vars}
ds.to_netcdf(savepath,format='NETCDF4')
print(f'NetCDF file saved to {savepath}')
'''




#------------------------------------------------------
#------------------------------------------------------
# VI. siextent
#------------------------------------------------------
#------------------------------------------------------

loadpath = rootpath+'siextent_cesmle_100members_mon_SEP_185001-210012.nc'
dataset = nc.Dataset(loadpath,'r')
siextent = np.array(dataset.variables['siextentm'])

sie_ensemble_mean = np.nanmean(siextent,axis=0,keepdims=True)
sieij = siextent - sie_ensemble_mean
# grab 1990 - 2040
years = np.arange(1850,2101)
y1990 = np.where(years==1991)[0][0]
y2041 = np.where(years==2041)[0][0]
sie = sieij[:,y1990:y2041]
nt = sie.shape[1]

del sieij

#train-test split, (80-10-10 for t-v-t)
siei = sie.reshape(10,10,nt)
sie_tri = siei[:,2:,:]
sie_vai = siei[:,1,:]
sie_tei = siei[:,0,:]

#reshape
sie_tr = sie_tri.reshape(-1)
sie_va = sie_vai.reshape(-1)
sie_te = sie_tei.reshape(-1)

#------------------------------------------------------
# save
#------------------------------------------------------

savepath = rootpath+'D2_cnn_cesm2le_siextent_seasons_TVT_1990-2100.nc'
ds = xr.Dataset(
    {
        "sie_tr":(("ntr",),sie_tr),
        "sie_va":(("nva",),sie_va),
        "sie_te":(("nte",),sie_te),

    },
    coords={
        "ntr":np.arange(sie_tr.shape[0]), #training samples
        "nva":np.arange(sie_va.shape[0]), #validation samples
        "nte":np.arange(sie_te.shape[0]), #test samples

    },
)

#save to NetCDF file
encoding = {var: {"zlib":True,"coJmplevel":4} for var in ds.data_vars}
ds.to_netcdf(savepath,format='NETCDF4')
print(f'NetCDF file saved to {savepath}')