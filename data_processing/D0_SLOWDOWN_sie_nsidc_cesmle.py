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
file_path = rootpath+'Sea_Ice_Index_Monthly_Data_with_Statistics_G02135_v3.0.xlsx'
months = ['January','February','March','April','May','June','July','August','September','October','November','December']
string_tail = '-NH'

current_date = datetime.now()
current_year = current_date.year
current_year = 2025
current_month = current_date.month
day = current_date.day
yearrange = np.arange(1978,current_year+1)
nn = current_year-1978+1

siextenti = []
time = []

for i in range(12):
    sie = np.full((nn,1), np.nan)
    years = np.full((nn,1), np.nan)
    sheet = months[i]+string_tail
    data = pd.read_excel(file_path,sheet_name=sheet)
    
    timei = np.array(data[['Unnamed: 1']])[9:]
    siei = np.array(data[['Unnamed: 5']])[9:]
    
    if i < 10:  
        start = 1979
    else:
        start = 1978
    
    #if i >= current_month-4:
    if i >= 2:
        end = current_year-1
    else: 
        end = current_year
   
    startyearindex = np.where(yearrange==start)[0][0]
    endyearindex = np.where(yearrange==end)[0][0]    
    sie[startyearindex:endyearindex+1,]=siei[:,]
    years[startyearindex:endyearindex+1,]=timei[:,]
    
    #siey = sie[1:-1]
    #yeary = years[1:-1]
    
    siey = sie
    yeary = years
    
    siextenti.append(siey)
    time.append(yeary)

siextent = np.reshape(np.array(siextenti),[12,nn])
timeall = np.reshape(np.array(time),[12,nn])

yearsf = timeall.flatten('F')
monthsf = np.transpose(np.tile(np.arange(1,13),[1,nn]))
ny = yearsf.shape[0]

years =np.reshape(yearsf.astype(int),(ny,))
years[:10] = 1978
years[ny-12:] = current_year
months = np.reshape(monthsf.astype(int),(ny,))
time= [datetime(year, month, 1) for year, month in zip(years, months)]

sie_yearly_meani = np.nanmean(siextent,axis=0)
sie=siextent.flatten('F')
year = np.arange(1979,current_year+1)

#fill in Dec 1987 and Jan 1988 with linear interpolation using SEP87-MAR88
tint = np.arange(1,8)
sieint = sie[116:123]
dx = tint
dy = sieint
dx2 = tint
 
#remove NaNs
dy = dy[~np.isnan(dx)]
dx = dx[~np.isnan(dx)]
 
dx = dx[~np.isnan(dy)]
dy = dy[~np.isnan(dy)]

x = dx
y = dy

# Perform a quadratic fit (degree = 2)
coefficients = np.polyfit(x, y, 2)
polynomial = np.poly1d(coefficients)

# Generate x values for plotting the fitted curve
x_fit = dx2
y_fit = polynomial(x_fit)

#replace sie[107:108] with extrapolated values
sie[119] = y_fit[3]
sie[120] = y_fit[4]

#a. separate into desired time frames
#------------------
#yearly mean
dates = np.array(time)
yearstf = np.array([date.year for date in dates])
unique_years = np.unique(yearstf)
sie_yearly_mean = np.array([np.nanmean(sie[years==year],axis=0) for year in unique_years])

#months
months = np.array([date.month for date in dates])
unique_months = np.unique(months)

#monthly sie
sie_mon = []
time_mon = []
for k in range(1,13):
    siem = sie[months==k]
    timem = dates[months==k]
    sie_mon.append(siem)
    time_mon.append(timem)

sie_monthly = np.transpose(np.array(sie_mon))
time_monthly = np.array(time_mon)


#b. take moving mean for T = 1:10 years
#------------------
#data = np.concatenate([sie_yearly_mean[:,np.newaxis],sie_monthly],axis=1) 
data = sie_monthly
nd = np.shape(data)[1]

#set NaNs for appropriate means in 1978 and 2024; 1==NaN
i98 = [1,1,1,1,1,1,1,1,1,1,0,0]
i25 = [0,0,1,1,1,1,1,1,1,1,1,1]

sieoj = []
for j in range(12):
    sieo = data[:,j]
    if i98[j] == 1:
        sieo[0] = np.nan 
    if i25[j] == 1:
        sieo[-1] = np.nan
    sieoj.append(sieo)
data = np.array(sieoj)

#10-year moving linear trends
yearmon = np.reshape(years,(48,12)).T
slopei = []
for i in range (12):
    slopej = []
    for j in range(48-10):
        dx = np.arange(10)
        dy = data[i,j:j+10]
        
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

linear_trends = np.array(slopei)[:,12:]
nt = linear_trends.shape[1]


#plot SIE and trends
for i in range(8,9):
    plt.plot(yearmon[i,:],data[i,:],linewidth = 3, color='black')
    for j in range(nt):
        dx = np.arange(10)
        dy = data[i,j+12:j+10+12]
        m = linear_trends[i,j]
        plt.plot(yearmon[i,j+12:j+10+12],m*dx+dy[0],linewidth=1,color='red')
plt.show()
        

#defining a threshold for a 'slowdown' event
mean_trend_obs = np.nanmean(linear_trends,axis=1)
std_trend_obs = np.nanstd(linear_trends,axis=1)

trend_obs_threshold_slowdown = mean_trend_obs+std_trend_obs
tot_rep_slowdown = np.tile(trend_obs_threshold_slowdown,(nt,1)).T
tot_mask_slowdown = linear_trends > tot_rep_slowdown
#fraction_obs_threshold_slowdown = np.sum(tot_mask_slowdown,axis=1)/nt
fraction_obs_threshold_slowdown = trend_obs_threshold_slowdown/mean_trend_obs

trend_obs_threshold_riles = mean_trend_obs-std_trend_obs
tot_rep_riles = np.tile(trend_obs_threshold_riles,(nt,1)).T
tot_mask_riles = linear_trends < tot_rep_riles
#fraction_obs_threshold_riles = np.sum(tot_mask_riles,axis=1)/nt
fraction_obs_threshold_riles = trend_obs_threshold_riles/mean_trend_obs

#plot SIE and trends
for i in range(8,9):
    plt.plot(yearmon[i,12:-10],linear_trends[i,:],linewidth=2,color='red')
    plt.plot(yearmon[i,12:-10],mean_trend_obs[i,]*np.ones(nt),linewidth=1,color='black')
    plt.plot(yearmon[i,12:-10],(mean_trend_obs[i,]+std_trend_obs[i,])*np.ones(nt),linewidth=3,color='steelblue',label=f'slowdown threshold ({fraction_obs_threshold_slowdown[i]:.2f} x $\mu$)')
    plt.plot(yearmon[i,12:-10],(mean_trend_obs[i,]-std_trend_obs[i,])*np.ones(nt),linewidth=3,color='green',label=f'riles threshold ({fraction_obs_threshold_riles[i]:.2f} x $\mu$)')
    plt.legend()
    plt.ylabel('Observed decadal trend in September SIE [$\\mathrm{M\\ km^2\\ yr^{-1}}$]')
plt.show()

x = np.arange(1979,2025)
y = data[8,1:-1]
coefficients = np.polyfit(x, y, 1)
slope = coefficients[0]
intercept = coefficients[1]
yy = slope*x+intercept

#plot SIE ANOMALY and -ve TREND
#HIGH SIE ANOMALY --> HIGH -ve TREND
fig, ax1 = plt.subplots()
for i in range(8,9):
    ax1.plot(yearmon[i,1:-1],data[i,1:-1]-yy,linewidth = 3, color='black')
    ax2 = ax1.twinx()
    ax2.plot(yearmon[i,12:-10],-linear_trends[i,:],linewidth=2,color='red')

'''
# Create an xarray Dataset
savepath = rootpath+'sie_nsidc_linear_decadal_trend_fraction_of_mean_monthly_1990-2024.nc'
ds = xr.Dataset(
    {
        "fraction_obs_threshold_riles": (("nm",), fraction_obs_threshold_riles),
        "fraction_obs_threshold_slowdown": (("nm",), fraction_obs_threshold_slowdown),
        "trend_obs_threshold_riles": (("nm",), trend_obs_threshold_riles),
        "trend_obs_threshold_slowdown": (("nm",), trend_obs_threshold_slowdown),
    },
    coords={
        "nm": np.arange(fraction_obs_threshold_riles.shape[0]),  # Months index
    },
)

# Save to a NetCDF file
encoding = {var: {"zlib": True, "complevel": 4} for var in ds.data_vars}
ds.to_netcdf(savepath, format="NETCDF4", encoding=encoding)

print(f"NetCDF file saved to {savepath}")
'''


# September slowdowns in OBS
sep_slowdown = tot_mask_slowdown[8,:].astype(int)
sep_linear_trend = linear_trends[8,:]

'''
# Create an xarray Dataset
savepath = rootpath+'sie_nsidc_slowdowns_1990-2015.nc'
ds = xr.Dataset(
    {
        "sep_slowdown": (("ny",), sep_slowdown),
        "sep_linear_trend": (("ny",), sep_linear_trend),

    },
    coords={
        "ny": np.arange(sep_slowdown.shape[0]),  # years
    },
)

# Save to a NetCDF file
encoding = {var: {"zlib": True, "complevel": 4} for var in ds.data_vars}
ds.to_netcdf(savepath, format="NETCDF4", encoding=encoding)

print(f"NetCDF file saved to {savepath}")
'''