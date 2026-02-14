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



#----------------------------------------------------------------
#----------------------------------------------------------------
# FIGURE S1
#----------------------------------------------------------------
#----------------------------------------------------------------

mon = 8 # SEP = 8
nrows = 2 # no. datasets
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

#plot siextent w/ decadal trends
ax1.plot(yearmon[mon,:],data[mon,:],linewidth = 3, color='black',label = 'NSIDC SIE')
for j in range(nt):
    dx = np.arange(10)
    dy = data[mon,j+12:j+10+12]
    m = linear_trends[mon,j]
    if j == 0:
        ax1.plot(yearmon[mon,j+12:j+10+12],m*dx+dy[0],linewidth=1,color='red',label = 'decadal trend')
    else: 
        ax1.plot(yearmon[mon,j+12:j+10+12],m*dx+dy[0],linewidth=1,color='red',label = '')
ax1.legend()    
ax1.set_title(titles[title_idx])
ax1.set_ylabel(r'September SIE [$\mathrm{M\ km^2}$]')

#plot decadal trends
ax2.plot(yearmon[mon,12:-10],linear_trends[mon,:],linewidth=2,color='red',label='decadal trend, obs')
ax2.plot(yearmon[mon,12:-10],mean_trend_obs[mon,]*np.ones(nt),linewidth=1,color='black',label = 'mean decadal trend')
ax2.plot(yearmon[mon,12:-10],(mean_trend_obs[mon,]+std_trend_obs[mon,])*np.ones(nt),linewidth=3,color='steelblue',linestyle='dashed',label=f'slowdown threshold ({fraction_obs_threshold_slowdown[mon]:.2f} x $\mu$)')
#ax2.plot(yearmon[mon,12:-10],(mean_trend_obs[mon,]-std_trend_obs[mon,])*np.ones(nt),linewidth=3,color='green',label=f'riles threshold ({fraction_obs_threshold_riles[mon]:.2f} x $\mu$)')
ax2.set_ylim([-0.35,0.05])
ax2.set_title(titles[title_idx+1])
ax2.legend(loc='lower right')
ax2.set_ylabel('decadal trend in September SIE [$\\mathrm{M\\ km^2\\ yr^{-1}}$]')

#----------------------------------------------------------------
 
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


#----------------------------------------------------------------
#----------------------------------------------------------------
# FIGURE S1 (c) and (d)
#----------------------------------------------------------------
#----------------------------------------------------------------

# === Second dataset ===
idx = 1
ax3 = axes[idx, 0]
ax4 = axes[idx, 1]
title_idx = 2 * idx

#plot SIE and trends
ax3.plot(years,sie_ens_mean,linewidth = 3, color='black',label = 'CESM2-LE ensemble mean')
for j in range(nt):
    dx = np.arange(10)
    dy = sie_ens_mean[j:j+10]
    m = linear_trends[j]
    if j == 0:
        ax3.plot(years[j:j+10],m*dx+dy[0],linewidth=1,color='red',label='decadal trend')
    else: 
        ax3.plot(years[j:j+10],m*dx+dy[0],linewidth=1,color='red',label = '')
ax3.set_title(titles[title_idx])
ax3.legend()
ax3.set_ylabel(r'September SIE [$\mathrm{M\ km^2}$]')


#plot SIE and trends
ax4.plot(years[:-10],linear_trends[:],linewidth=2,color='red',label='decadal trend, ensemble mean')
ax4.plot(years[:-10],trend_obs_threshold_slowdown[8]*np.ones(nt),linewidth=2,color='steelblue',linestyle='dashed',label='slowdown observed threshold')
#ax4.plot(years[:-10],trend_obs_threshold_riles[8]*np.ones(nt),linewidth=2,color='green',linestyle='dashed',label='rile observed threshold')
ax4.plot(years[:-10],fraction_obs_threshold_slowdown[8]*linear_trends[:],linewidth=2,color='skyblue',label='slowdown model threshold')
#ax4.plot(years[:-10],fraction_obs_threshold_riles[8]*linear_trends[:],linewidth=2,color='lightgreen',label='rile model threshold')
ax4.set_title(titles[title_idx+1])
ax4.set_ylabel('decadal trend in September SIE [$\\mathrm{M\\ km^2\\ yr^{-1}}$]')
ax4.legend()

#----------------------------------------------------------------
#----------------------------------------------------------------




#----------------------------------------------------------------
#----------------------------------------------------------------
# FS2: PDF of slowdowns
#----------------------------------------------------------------
#----------------------------------------------------------------


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


nrows = 1
fig,axes = plt.subplots(nrows=nrows,ncols=4,figsize=(32,8))

if nrows ==1:
    axes = np.array([axes])

# === First dataset ===
idx = 0
ax1 = axes[idx, 0]
ax2 = axes[idx, 1]
ax3 = axes[idx, 2]
ax4 = axes[idx, 3]
title_idx = 2 * idx

#frequency of slowdown events per member
slowdown_per_member = np.nansum(slowdown,axis=1)
slowdown_per_member_9039 = np.nansum(slowdown[:,:50],axis=1)
slowdown_per_member_4090 = np.nansum(slowdown[:,50:],axis=1)
max_val = slowdown_per_member.max()
bins = np.arange(0, max_val + 3) - 1
counts, bin_edges = np.histogram(slowdown_per_member_9039, bins=bins, density=True)
ax1.bar(bin_edges[:-1], counts, width=1.0, color='steelblue', edgecolor='white', alpha=0.7,label='1990-2039')
counts, bin_edges = np.histogram(slowdown_per_member_4090, bins=bins, density=True)
ax1.bar(bin_edges[:-1], counts, width=1.0, color='red', edgecolor='white', alpha=0.2,label='2040-2099')
ax1.set_xlabel('Number of slowdown events per member')
ax1.set_ylabel('Frequency')
ax1.set_xticks(np.arange(0, max_val + 2, 3))
ax1.set_title('(a)')
ax1.legend()



# Plot the theoretical PDF
slowdown_flat = np.reshape(slowdown, (-1,))
ax2.hist(slowdown_flat,
         bins=[-0.5, 0.5, 1.5],  # to center bins on 0 and 1
         weights=np.ones_like(slowdown_flat) / len(slowdown_flat),  # normalize to fraction
         color='grey')
ax2.set_xticks([0, 1], ['no slowdown', 'slowdown'])
ax2.set_ylabel('fraction of events')
ax2.set_title('(b)')


#----------------------------------------------------------------
#----------------------------------------------------------------
# FS2: PDF of SIE during slowdowns
#----------------------------------------------------------------
#----------------------------------------------------------------
sie_slowdown = siei90[:,:-10]
mask = slowdown

data_0 = sie_slowdown[(mask == 0)]
data_1 = sie_slowdown[(mask == 1)]

# Plot histograms
ax3.hist(data_0, bins=30, alpha=0.6, label='no slowdown', density=True, color='blue')
ax3.hist(data_1, bins=30, alpha=0.6, label='slowdown', density=True, color='red')
ax3.set_xlabel(r'sea ice extent [M km$^2$]')
ax3.set_ylabel('Density')
ax3.legend()
ax3.set_title('(c)')


#----------------------------------------------------------------
#----------------------------------------------------------------
# FS2: PDF of SIE during slowdowns
#----------------------------------------------------------------
#----------------------------------------------------------------
sie_anomaly_slowdown = siei90[:,:-10] - np.nanmean(siei90[:,:-10],axis=0)
mask = slowdown

data_0 = sie_anomaly_slowdown[(mask == 0)]
data_1 = sie_anomaly_slowdown[(mask == 1)]

# Plot histograms
ax4.hist(data_0, bins=30, alpha=0.6, label='no slowdown', density=True, color='blue')
ax4.hist(data_1, bins=30, alpha=0.6, label='slowdown', density=True, color='red')
ax4.set_xlabel(r'sea ice extent anomaly [M km$^2$]')
ax4.set_ylabel('Density')
ax4.legend()
ax4.set_title('(d)')






#----------------------------------------------------------------
#----------------------------------------------------------------
# FS2: PDF of slowdowns (1990 - 2040)
#----------------------------------------------------------------
#----------------------------------------------------------------


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


'''
savepath = rootpath+'sie_nsidc_linear_decadal_slowdowns_1990-2015.nc'
ds = xr.Dataset(
    {
        "slowdown": (("ny","nm"), slowdown),
        "linear_trends_slowdown": (("ny","nm"), linear_trends_slowdown),

    },
    coords={
        "ny": np.arange(slowdown.shape[0]),  # years
        "nm": np.arange(slowdown.shape[1]), 
    },
)

# Save to a NetCDF file
encoding = {var: {"zlib": True, "complevel": 4} for var in ds.data_vars}
ds.to_netcdf(savepath, format="NETCDF4", encoding=encoding)
print(f"NetCDF file saved to {savepath}")
'''

nrows = 1
fig,axes = plt.subplots(nrows=nrows,ncols=4,figsize=(32,8))

if nrows ==1:
    axes = np.array([axes])

# === First dataset ===
idx = 0
ax1 = axes[idx, 0]
ax2 = axes[idx, 1]
ax3 = axes[idx, 2]
ax4 = axes[idx, 3]
title_idx = 2 * idx

#frequency of slowdown events per member
slowdown_per_member = np.nansum(slowdown,axis=1)
slowdown_per_member_9039 = np.nansum(slowdown[:,:50],axis=1)
slowdown_per_member_4090 = np.nansum(slowdown[:,50:],axis=1)
max_val = slowdown_per_member.max()
bins = np.arange(0, max_val + 3) - 1
counts, bin_edges = np.histogram(slowdown_per_member_9039, bins=bins, density=True)
ax1.bar(bin_edges[:-1], counts, width=1.0, color='steelblue', edgecolor='white', alpha=0.7,label='1990-2039')
counts, bin_edges = np.histogram(slowdown_per_member_4090, bins=bins, density=True)
ax1.bar(bin_edges[:-1], counts, width=1.0, color='red', edgecolor='white', alpha=0.2,label='2040-2099')
ax1.set_xlabel('Number of slowdown events per member')
ax1.set_ylabel('Frequency')
ax1.set_xticks(np.arange(0, max_val + 2, 3))
ax1.set_title('(a)')
ax1.legend()



# Plot the theoretical PDF
slowdown_flat = np.reshape(slowdown[:,:50], (-1,))
ax2.hist(slowdown_flat,
         bins=[-0.5, 0.5, 1.5],  # to center bins on 0 and 1
         weights=np.ones_like(slowdown_flat) / len(slowdown_flat),  # normalize to fraction
         color='grey')
ax2.set_xticks([0, 1], ['no slowdown', 'slowdown'])
ax2.set_ylabel('fraction of events')
ax2.set_title('(b)')


#----------------------------------------------------------------
#----------------------------------------------------------------
# FS2: PDF of SIE during slowdowns
#----------------------------------------------------------------
#----------------------------------------------------------------
sie_slowdown = siei90[:,:50]
mask = slowdown[:,:50]

data_0 = sie_slowdown[(mask == 0)]
data_1 = sie_slowdown[(mask == 1)]

# Plot histograms
ax3.hist(data_0, bins=30, alpha=0.6, label='no slowdown', density=True, color='blue')
ax3.hist(data_1, bins=30, alpha=0.6, label='slowdown', density=True, color='red')
ax3.set_xlabel(r'sea ice extent [M km$^2$]')
ax3.set_ylabel('Density')
ax3.legend()
ax3.set_title('(c)')


#----------------------------------------------------------------
#----------------------------------------------------------------
# FS2: PDF of SIE during slowdowns
#----------------------------------------------------------------
#----------------------------------------------------------------
sie_anomaly_slowdown = siei90[:,:50] - np.nanmean(siei90[:,:50],axis=0)
mask = slowdown[:,:50]

data_0 = sie_anomaly_slowdown[(mask == 0)]
data_1 = sie_anomaly_slowdown[(mask == 1)]

# Plot histograms
ax4.hist(data_0, bins=30, alpha=0.6, label='no slowdown', density=True, color='blue')
ax4.hist(data_1, bins=30, alpha=0.6, label='slowdown', density=True, color='red')
ax4.set_xlabel(r'sea ice extent anomaly [M km$^2$]')
ax4.set_ylabel('Density')
ax4.legend()
ax4.set_title('(d)')
#----------------------------------------------------------------
#----------------------------------------------------------------



#----------------------------------------------------------------
#----------------------------------------------------------------
# FIGURE S3:
#----------------------------------------------------------------
#----------------------------------------------------------------

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










