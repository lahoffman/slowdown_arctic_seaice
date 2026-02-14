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

#data processing
#------------------
#scipy
from scipy import stats, odr
from scipy.io import netcdf
from scipy.stats import norm

#other
from datetime import datetime
from datetime import timedelta

#import functions
#------------------
sys.path.append(rootpath+'functions/')
from functions_general import ncdisp
from functions_general import movmean

#plotting
#------------------
import matplotlib.pyplot as plt

#------------------------------------------------------
#------------------------------------------------------

#------------------------------------------------------
#------------------------------------------------------
# I. siextent, NSIDC
#------------------------------------------------------
#------------------------------------------------------

loadpath = rootpath+'/sie_nsidc_linear_decadal_trend_fraction_of_mean_monthly_1990-2024.nc'
dataset = nc.Dataset(loadpath,'r')
fraction_obs_threshold_riles = np.array(dataset.variables['fraction_obs_threshold_riles'])
fraction_obs_threshold_slowdown = np.array(dataset.variables['fraction_obs_threshold_slowdown'])
trend_obs_threshold_riles = np.array(dataset.variables['trend_obs_threshold_riles'])
trend_obs_threshold_slowdown = np.array(dataset.variables['trend_obs_threshold_slowdown'])


#------------------------------------------------------
#------------------------------------------------------
# II. siextent, CESM2-LE
#------------------------------------------------------
#------------------------------------------------------

#model file paths
#------------------------------------------------------
mon = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']

filepathlead = rootpath+'siextent_cesmle_100members_mon_'
filepathtail = '_185001-210012.nc'

for i in range(8,9):
    loadpath = filepathlead+mon[i]+filepathtail
    dataset = nc.Dataset(loadpath,'r')
    siei = np.array(dataset.variables['siextentm'])

#ensemble mean
sie_ens_mean = np.nanmean(siei,axis=0)
years = np.arange(1850,2101)
y1990 = np.where(years==1990)[0][0]

sie_ens_mean = sie_ens_mean[y1990:]
years = years[y1990:]

#10-year moving linear trends
ny = years.shape[0]

slopej = []
for j in range(ny-10):
    dx = np.arange(10)
    dy = sie_ens_mean[j:j+10]
    
    #remove NaNs
    dy = dy[~np.isnan(dx)]
    dx = dx[~np.isnan(dx)]
    
    dx = dx[~np.isnan(dy)]
    dy = dy[~np.isnan(dy)]

    x = dx
    y = dy
    
    # Perform a quadratic fit (degree = 2)
    coefficients = np.polyfit(x, y, 1)
    polynomial = np.poly1d(coefficients)
    slope = coefficients[0]
    intercept = coefficients[1]
    
    
    slopej.append(slope)
    
linear_trends = np.array(slopej)
nt = linear_trends.shape[0]

#plot SIE and trends

plt.plot(years,sie_ens_mean,linewidth = 3, color='black')
for j in range(nt):
    dx = np.arange(10)
    dy = sie_ens_mean[j:j+10]
    m = linear_trends[j]
    plt.plot(years[j:j+10],m*dx+dy[0],linewidth=1,color='red')
plt.show()


#plot SIE and trends
plt.plot(years[:-10],linear_trends[:],linewidth=2,color='black',label='ensemble mean')
plt.plot(years[:-10],trend_obs_threshold_slowdown[8]*np.ones(nt),linewidth=2,color='steelblue',linestyle='dashed',label='slowdown observed threshold')
plt.plot(years[:-10],trend_obs_threshold_riles[8]*np.ones(nt),linewidth=2,color='green',linestyle='dashed',label='rile observed threshold')
plt.plot(years[:-10],fraction_obs_threshold_slowdown[8]*linear_trends[:],linewidth=2,color='skyblue',label='slowdown model threshold')
plt.plot(years[:-10],fraction_obs_threshold_riles[8]*linear_trends[:],linewidth=2,color='lightgreen',label='rile model threshold')
plt.ylabel('decadal trend in September SIE [$\\mathrm{M\\ km^2\\ yr^{-1}}$]')
plt.legend()
plt.show()

#plot SIE and trends
plt.plot(years[:-10],linear_trends[:],linewidth=2,color='black',label='ensemble mean')
plt.plot(years[:-10],trend_obs_threshold_slowdown[8]*np.ones(nt),linewidth=2,color='steelblue',linestyle='dashed',label='slowdown observed threshold')
plt.plot(years[:-10],fraction_obs_threshold_slowdown[8]*linear_trends[:],linewidth=2,color='skyblue',label='slowdown model threshold')
plt.ylabel('decadal trend in September SIE [$\\mathrm{M\\ km^2\\ yr^{-1}}$]')
plt.legend()
plt.show()


#plot SIE and trends
plt.plot(years[:-10],linear_trends[:],linewidth=2,color='black',label='ensemble mean')
plt.plot(years[:-10],trend_obs_threshold_riles[8]*np.ones(nt),linewidth=2,color='green',linestyle='dashed',label='rile observed threshold')
plt.plot(years[:-10],fraction_obs_threshold_riles[8]*linear_trends[:],linewidth=2,color='lightgreen',label='rile model threshold')
plt.ylabel('decadal trend in September SIE [$\\mathrm{M\\ km^2\\ yr^{-1}}$]')
plt.legend()
plt.show()


#calculate climate model treshold (slowdown & riles)
# threshold = fraction of mean from obs * ensemble mean trend
model_obs_threshold_slowdown = fraction_obs_threshold_slowdown[8]*linear_trends[:]
model_obs_threshold_riles = fraction_obs_threshold_riles[8]*linear_trends[:]

#categorize decadal trends in ensemble members as 'slowdown' or 'no slowdown'
# slowdown = trend is below climate model threshold
# no slowdown = trend is above climate model threshold

# a .linear trends for each ensemble member

siei90 = siei[:,y1990:]
ny = siei90.shape[1]

slopei = []
for i in range (100):
    slopej = []
    for j in range(ny-10):
        dx = np.arange(10)
        dy = siei90[i,j:j+10]
        
        #remove NaNs
        dy = dy[~np.isnan(dx)]
        dx = dx[~np.isnan(dx)]
        
        dx = dx[~np.isnan(dy)]
        dy = dy[~np.isnan(dy)]

        x = dx
        y = dy
        
        # Perform a quadratic fit (degree = 2)
        coefficients = np.polyfit(x, y, 1)
        polynomial = np.poly1d(coefficients)
        slope = coefficients[0]
        intercept = coefficients[1]
        
        
        slopej.append(slope)
    slopei.append(slopej)
        
linear_trends_ens = np.array(slopei)
nt = linear_trends_ens.shape[0]


#---------------------------------------------------------
#---------------------------------------------------------
# slowdown
#---------------------------------------------------------
#---------------------------------------------------------

threshold_slowdown = np.tile(model_obs_threshold_slowdown,(100,1))
slowdown = (linear_trends_ens > threshold_slowdown).astype(int)
linear_trends_slowdown = np.where(linear_trends_ens > threshold_slowdown, 
                                  linear_trends_ens, np.nan)

#plot SIE and trends
nt1 = linear_trends.shape[0]
eno = 6
for i in range(100):    
    plt.plot(years[:-10],linear_trends_ens[i,:],color='gray',linewidth='.2')
plt.plot(years[:-10],linear_trends[:],linewidth=3,color='black',label='ensemble mean')
plt.plot(years[:-10],trend_obs_threshold_slowdown[8]*np.ones(nt1),linewidth=2,color='steelblue',linestyle='dashed')
plt.plot(years[:-10],fraction_obs_threshold_slowdown[8]*linear_trends[:],linewidth=2,color='lightblue',label='slowdown threshold')
plt.plot(years[:-10],linear_trends_ens[eno,:],color='teal',linewidth='2',label=f'ensemble no. {eno-1}')
plt.plot(years[:-10],linear_trends_slowdown[eno,:],color='red',linewidth='2',marker='o', linestyle='none',label = f'ensemble no. {eno-1} slowdowns')
plt.ylabel('Decadal Trends in SIE')
plt.xlim([1990,2100])
plt.legend()
plt.show()

# Plot the theoretical PDF
slowdown_flat = np.reshape(slowdown, (-1,))
plt.hist(slowdown_flat,
         bins=[-0.5, 0.5, 1.5],  # to center bins on 0 and 1
         weights=np.ones_like(slowdown_flat) / len(slowdown_flat),  # normalize to fraction
         color='grey')
plt.xticks([0, 1], ['no slowdown', 'slowdown'])
plt.ylabel('Fraction of events')
plt.title('Distribution of Slowdown Events')

plt.tight_layout()
plt.show()

#frequency of slowdown events per member
slowdown_per_member = np.nansum(slowdown,axis=1)
max_val = slowdown_per_member.max()
bins = np.arange(0, max_val + 3) - 1
counts, bin_edges = np.histogram(slowdown_per_member, bins=bins, density=True)
plt.bar(bin_edges[:-1], counts, width=1.0, color='steelblue', edgecolor='white', alpha=0.8)
plt.xlabel('Number of slowdown events per member')
plt.ylabel('Frequency')
plt.xticks(np.arange(0, max_val + 2, 3))
plt.tight_layout()
plt.show()

#frequency of slowdown events per member
slowdown_per_member_9039 = np.nansum(slowdown[:,:50],axis=1)
slowdown_per_member_4090 = np.nansum(slowdown[:,50:],axis=1)
max_val = slowdown_per_member.max()
bins = np.arange(0, max_val + 3) - 1
counts, bin_edges = np.histogram(slowdown_per_member_9039, bins=bins, density=True)
plt.bar(bin_edges[:-1], counts, width=1.0, color='steelblue', edgecolor='white', alpha=0.7,label='1990-2039')
counts, bin_edges = np.histogram(slowdown_per_member_4090, bins=bins, density=True)
plt.bar(bin_edges[:-1], counts, width=1.0, color='red', edgecolor='white', alpha=0.2,label='2040-2099')
plt.xlabel('Number of slowdown events per member')
plt.ylabel('Frequency')
plt.xticks(np.arange(0, max_val + 2, 3))
plt.legend()
plt.tight_layout()
plt.show()


# slowdown: Create an xarray Dataset
savepath = rootpath+'sie_nsidc_linear_decadal_slowdown_1990-2024.nc'
ds = xr.Dataset(
    {
        "slowdown": (("nm","nt"), slowdown),
    },
    coords={
        "nm": np.arange(slowdown.shape[0]),  # ensemble no.
        "nt": np.arange(slowdown.shape[1]),  # time
    },
)

# Save to a NetCDF file
encoding = {var: {"zlib": True, "complevel": 4} for var in ds.data_vars}
ds.to_netcdf(savepath, format="NETCDF4", encoding=encoding)

print(f"NetCDF file saved to {savepath}")







#---------------------------------------------------------
#---------------------------------------------------------
#riles
#---------------------------------------------------------
#---------------------------------------------------------

threshold_riles = np.tile(model_obs_threshold_riles,(100,1))
riles = (linear_trends_ens < threshold_riles).astype(int)
linear_trends_riles = np.where(linear_trends_ens < threshold_riles, 
                                  linear_trends_ens, np.nan)

# Plot the theoretical PDF
riles_flat = np.reshape(riles, (-1,))
plt.hist(riles_flat,
         bins=[-0.5, 0.5, 1.5],  # to center bins on 0 and 1
         weights=np.ones_like(riles_flat) / len(riles_flat),  # normalize to fraction
         color='grey')
plt.xticks([0, 1], ['no riles', 'riles'])
plt.ylabel('Fraction of events')
plt.title('Distribution of RILES Events')

plt.tight_layout()
plt.show()

#plot SIE and trends
nt1 = linear_trends.shape[0]
eno = 6
for i in range(100):    
    plt.plot(years[:-10],linear_trends_ens[i,:],color='gray',linewidth='.2')
plt.plot(years[:-10],linear_trends[:],linewidth=3,color='black',label='ensemble mean')
plt.plot(years[:-10],trend_obs_threshold_riles[8]*np.ones(nt1),linewidth=2,color='green',linestyle='dashed')
plt.plot(years[:-10],fraction_obs_threshold_riles[8]*linear_trends[:],linewidth=2,color='lightgreen',label='riles threshold')
plt.plot(years[:-10],linear_trends_ens[eno,:],color='teal',linewidth='2',label=f'ensemble no. {eno-1}')
plt.plot(years[:-10],linear_trends_riles[eno,:],color='red',linewidth='2',marker='o', linestyle='none',label = f'ensemble no. {eno-1} riles')
plt.ylabel('Decadal Trends in SIE')
plt.xlim([1990,2100])
plt.legend()
plt.show()

#frequency of riles events per member
riles_per_member = np.nansum(riles,axis=1)
riles_per_member_9039 = np.nansum(riles[:,:50],axis=1)
riles_per_member_4090 = np.nansum(riles[:,50:],axis=1)
max_val = riles_per_member.max()
bins = np.arange(0, max_val + 3) - 1
counts, bin_edges = np.histogram(riles_per_member_9039, bins=bins, density=True)
plt.bar(bin_edges[:-1], counts, width=1.0, color='steelblue', edgecolor='white', alpha=0.5,label='1990-2039')
counts, bin_edges = np.histogram(riles_per_member_4090, bins=bins, density=True)
plt.bar(bin_edges[:-1], counts, width=1.0, color='red', edgecolor='white', alpha=0.5,label='2040-2099')
plt.xlabel('Number of riles events per member')
plt.ylabel('Frequency')
plt.xticks(np.arange(0, max_val + 2, 3))
plt.legend()
plt.tight_layout()
plt.show()

'''
# riles: Create an xarray Dataset
savepath = rootpath+'sie_nsidc_linear_decadal_riles_1990-2024.nc'
ds = xr.Dataset(
    {
        "riles": (("nm","nt"), riles),
    },
    coords={
        "nm": np.arange(riles.shape[0]),  # ensemble no.
        "nt": np.arange(riles.shape[1]),  # time
    },
)

# Save to a NetCDF file
encoding = {var: {"zlib": True, "complevel": 4} for var in ds.data_vars}
ds.to_netcdf(savepath, format="NETCDF4", encoding=encoding)

print(f"NetCDF file saved to {savepath}")
'''
