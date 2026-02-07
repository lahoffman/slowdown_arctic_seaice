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
from scipy.interpolate import RegularGridInterpolator
import math 

#other
from datetime import datetime
from datetime import timedelta


#plotting
#------------------
#matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.cm import ScalarMappable
import cartopy.crs as ccrs

#colorbars
#------------------
import cmocean 

#import functions
#------------------
sys.path.append(rootpath+'functions/')
from functions_general import ncdisp
from functions_general import movmean
#------------------------------------------------------
#------------------------------------------------------

#------------------------------------------------------
#------------------------------------------------------
# ERSST-V5 SST
# https://psl.noaa.gov/thredds/catalog/Datasets/noaa.ersst.v5/catalog.html?dataset=Datasets/noaa.ersst.v5/sst.mnmean.nc
#------------------------------------------------------
#------------------------------------------------------

#----------------------------------------
#load
#----------------------------------------
filepath = rootpath+'sst.mnmean.nc'
dataset = nc.Dataset(filepath,'r')  
lat_sst = np.array(dataset.variables['lat'])
lon_sst = np.array(dataset.variables['lon'])
ssti = np.array(dataset.variables['sst'])
time = np.array(dataset.variables['time'])
dates = pd.to_datetime(time, unit='d', origin='1800-01-01')

# 1854-2024
date = dates[:-6]
sst = ssti[:-6,:,:]
sst[sst < -9*10**3] = np.nan

'''
#----------------------------------------
#jja mean
#----------------------------------------
months = date.month
years = date.year
jja_mask = months.isin([6,7,8])
jja_years = years[jja_mask]
unique_years = np.unique(jja_years)
nlat,nlon = sst.shape[1],sst.shape[2]
sst_jja = np.empty((len(unique_years),nlat,nlon))
for i, year in enumerate(unique_years):
    idx = np.where((years==year) & (months.isin([6,7,8])))[0]
    sst_jja[i] = sst[idx].mean(axis=0)


#----------------------------------------
#plot JJA mean
#----------------------------------------

fig,ax = plt.subplots(1, 1, figsize=(16, 16), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

data = sst_jja[-7,:,:]

ax.set_global()
ax.coastlines()
im = ax.pcolormesh(lon_sst, lat_sst, data, cmap=cmocean.cm.thermal,
                    shading='auto', transform=ccrs.PlateCarree())
ax.set_xticks(np.arange(-180,181, 60))   # Longitude labels
ax.set_yticks(np.arange(-90, 91, 30))  # Latitude labels
ax.xaxis.set_tick_params(rotation=45)  # Rotate longitude labels for better visibility
ax.yaxis.set_tick_params(rotation=45)  # Rotate latitude labels

# Add colorbar
cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)
cbar.set_label('')
#------------------------------------------------------
'''

#------------------------------------------------------
#----------------------------------------
# re-grid SST to CESM2-LE grid
#----------------------------------------
#------------------------------------------------------

#------------------------------------------------------
# 0. CESM2-LE LAT, LON 
#------------------------------------------------------
filepath = rootpath+'b.e21.BHISTcmip6.f09_g17.LE2-1001.001.cam.h1.Z200.18500101-18591231.nc'
dataset = nc.Dataset(filepath,'r')
lat_cesm2 = np.array(dataset.variables['lat'])
lon_cesm2 = np.array(dataset.variables['lon'])
#------------------------------------------------------

#------------------------------------------------------
# REGRID
#------------------------------------------------------
# 1. Create 2D meshgrid of SST target points
lon_cesm2_grid, lat_cesm2_grid = np.meshgrid(lon_cesm2, lat_cesm2)

# 2. Flatten and stack target coordinates into shape (N, 2) where N = nlat_sst * nlon_sst
target_points = np.stack([lat_cesm2_grid.ravel(), lon_cesm2_grid.ravel()], axis=-1)  # shape (N, 2)

# 3. Prepare output array
nt = sst.shape[0]
nlat_cesm2 = len(lat_cesm2)
nlon_cesm2 = len(lon_cesm2)
sst_on_cesm2_grid = np.empty((nt, nlat_cesm2, nlon_cesm2))

# 4. Interpolate each time step
for t in range(nt):
    # Create interpolator for the current time slice
    interp_func = RegularGridInterpolator(
        (lat_sst, lon_sst), 
        sst[t], 
        bounds_error=False, 
        fill_value=np.nan  # Optional: could also use extrapolation strategy
    )
    
    # Apply interpolation to target grid
    interpolated = interp_func(target_points)  # shape (nlat_sst * nlon_sst,)
    
    # Reshape and store
    sst_on_cesm2_grid[t] = interpolated.reshape((nlat_cesm2, nlon_cesm2))
   
    
    
 # List of data and metadata
datasets = [
    {"data": sst_on_cesm2_grid[0, :, :], "lat": lat_cesm2, "lon": lon_cesm2, "title": "SST on CESM2 grid"},
    {"data": sst[0, :, :],        "lat": lat_sst, "lon": lon_sst, "title": "SST original grid"},
]
#------------------------------------------------------

#------------------------------------------------------
# PLOT
#------------------------------------------------------
# Create figure and axes
fig, axes = plt.subplots(1, 2, figsize=(20, 10),
                         subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

# Loop through datasets and plot
for ax, ds in zip(axes, datasets):
    ax.set_global()
    ax.coastlines()

    im = ax.pcolormesh(ds["lon"], ds["lat"], ds["data"],
                       cmap=cmocean.cm.thermal,
                       shading='auto',
                       transform=ccrs.PlateCarree())
    
    im.set_clim(0,35)

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
#------------------------------------------------------

#------------------------------------------------------
#------------------------------------------------------

'''
#------------------------------------------------------
#------------------------------------------------------
# SAVE DATA
#------------------------------------------------------
#------------------------------------------------------

savepath = rootpath+'D1_ersstv5_sst_regridded_to_cesm2_1854-2024.nc'
ds = xr.Dataset(
    {
        "sst_obs":(("nte","nx","ny"),sst_on_cesm2_grid),
        "lat_cesm2":(("nx",),lat_cesm2),
        "lon_cesm2":(("ny",),lon_cesm2),
        "date":(("nte,"),date),
    },
    
    coords={
        "nte":np.arange(sst_on_cesm2_grid.shape[0]), #timesteps
        "nx":np.arange(sst_on_cesm2_grid.shape[1]), #lat index
        "ny":np.arange(sst_on_cesm2_grid.shape[2]), #lon index
    },
)

#save to NetCDF file
encoding = {var: {"zlib":True,"coJmplevel":4} for var in ds.data_vars}
ds.to_netcdf(savepath,format='NETCDF4')
print(f'NetCDF file saved to {savepath}')


#------------------------------------------------------
#------------------------------------------------------
'''