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

#plot
#------------------
import matplotlib.pyplot as plt

#------------------------------------------------------
#------------------------------------------------------

#------------------------------------------------------
#------------------------------------------------------
# I. siextent, NSIDC
#------------------------------------------------------
#------------------------------------------------------

loadpath = rootpath+'sie_nsidc_linear_decadal_trend_fraction_of_mean_monthly_1990-2024.nc'
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

# sie file paths
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


nrows = 1 # no. datasets
dataset_indices = range(nrows)
titles = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(d)']

fig,axes = plt.subplots(nrows=nrows,ncols=2,figsize=(16,8))

if nrows ==1:
    axes = np.array([axes])

# === First dataset ===
idx = 0
ax1 = axes[idx, 0]
ax2 = axes[idx, 1]
title_idx = 2 * idx
eno = 6

sie_slowdown = siei90[:,:-10].copy()
sie_slowdown[np.isnan(linear_trends_slowdown)] = np.nan

siej = siei[:,y1990:].copy()

#plot SIE and trends
#for j in range(100):
    #ax1.plot(years,siei90[j,:],linewidth=.2,color='gray')
for j in range(nt):
    dx = np.arange(10)
    dy = siej[eno,j:j+10]
    m = linear_trends_ens[eno,j]
    if j == 10 and not np.isnan(sie_slowdown[eno,j]):
        ax1.plot(years[j:j+10],m*dx+dy[0],linewidth=1,color='red',label=f'decadal trend for slowdowns in ensemble no. {eno+1}')
    elif not np.isnan(sie_slowdown[eno,j]): 
        ax1.plot(years[j:j+10],m*dx+dy[0],linewidth=1,color='red',label = '')
ax1.plot(years,sie_ens_mean,linewidth = 3, color='black',label = 'CESM2-LE ensemble mean')
ax1.plot(years,siei[eno,y1990:],linewidth=3,color='teal',label=f'ensemble no. {eno+1}')
ax1.plot(years[:-10],sie_slowdown[eno,:],color='red',linewidth='2',marker='o', linestyle='none',label = f'ensemble no. {eno+1} slowdowns')
ax1.set_title(titles[title_idx])
ax1.legend()
ax1.set_ylabel(r'September SIE [$\mathrm{M\ km^2}$]')



#plot SIE and trends
nt1 = linear_trends.shape[0]
eno = 6
for i in range(100):    
    ax2.plot(years[:-10],linear_trends_ens[i,:],color='gray',linewidth='.2')
ax2.plot(years[:-10],linear_trends[:],linewidth=3,color='black',label='ensemble mean')
ax2.plot(years[:-10],trend_obs_threshold_slowdown[8]*np.ones(nt1),linewidth=2,color='steelblue',linestyle='dashed')
ax2.plot(years[:-10],fraction_obs_threshold_slowdown[8]*linear_trends[:],linewidth=2,color='lightblue',label='slowdown threshold')
ax2.plot(years[:-10],linear_trends_ens[eno,:],color='teal',linewidth='2',label=f'ensemble no. {eno+1}')
ax2.plot(years[:-10],linear_trends_slowdown[eno,:],color='red',linewidth='2',marker='o', linestyle='none',label = f'ensemble no. {eno+1} slowdowns')
ax2.set_ylabel(r'decadal trens in SIE [$\mathrm{M\ km^2\ yr^{-1}}$]')
ax2.set_xlim([1990,2100])
ax2.set_title('(b)')
ax2.legend()










