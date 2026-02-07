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
from scipy.interpolate import griddata
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
sys.path.append('/home/elic/lhoffman/functions/')
from functions_general import ncdisp
from functions_general import movmean
#------------------------------------------------------
#------------------------------------------------------


#------------------------------------------------------
#------------------------------------------------------
# I. LOAD DATA
#------------------------------------------------------
#------------------------------------------------------

#lat / lon SST
filepath = '/cofast/lhoffman/cesmle/z200/raw/b.e21.BHISTcmip6.f09_g17.LE2-1001.001.cam.h1.Z200.18500101-18591231.nc'
dataset = nc.Dataset(filepath,'r')
lat_sst = np.array(dataset.variables['lat']) #shape (192,)
lon_sst = np.array(dataset.variables['lon']) #shape (288,)

#lat / lon AICE
filepath = '/cofast/lhoffman/cesmle/aice/raw/b.e21.BHISTcmip6.f09_g17.LE2-1001.001.cice.h.aice.185001-185912.nc'
dataset = nc.Dataset(filepath,'r')
lat_aice = np.array(dataset.variables['TLAT']) #shape (384,320)
lon_aice = np.array(dataset.variables['TLON']) #shape (384,320)
bad_mask = (
    np.isnan(lat_aice) | np.isnan(lon_aice) |
    np.isinf(lat_aice) | np.isinf(lon_aice) |
    (lat_aice > 91)    | (lat_aice < -91) |
    (lon_aice > 361)   | (lon_aice < -1)
)


#AICE
filepathlead1 = '/cofast/lhoffman/cesmle/aice/mon/aice_cesmle_first50members_mon_'
filepathlead2 = '/cofast/lhoffman/cesmle/aice/mon/aice_cesmle_last50members_mon_'
filepathtail = '_199001-210012.nc'
mon = ['JUN','JUL','AUG']

# JJA mean
aice_jja = []
for i in range(3):
    #load & concatenate first and last 50 members
    loadpath1 = filepathlead1+mon[i]+filepathtail
    loadpath2 = filepathlead2+mon[i]+filepathtail

    dataset = nc.Dataset(loadpath1,'r')
    aice1 = dataset.variables['aice_mon']
    
    dataset = nc.Dataset(loadpath2,'r')
    aice2 = dataset.variables['aice_mon']

    aice12 = np.concatenate((aice1,aice2),axis=0)
    aicei = aice12[:,:-10,:,:]
    
    aice_jja.append(aicei)
aice = np.nanmean(aice_jja,axis=0)

del aice1, aice2, aicei, aice_jja, aice12


#------------------------------------------------------
#------------------------------------------------------
# II. REGRID AICE to SST GRID
#------------------------------------------------------
#------------------------------------------------------
# 1. Flatten curvilinear grid to coordinate list (N, 2)
source_points = np.stack([lat_aice.ravel(), lon_aice.ravel()], axis=-1)  # shape (N, 2)

# 2. Create target grid mesh
lon_sst_grid, lat_sst_grid = np.meshgrid(lon_sst, lat_sst)
target_points = np.stack([lat_sst_grid.ravel(), lon_sst_grid.ravel()], axis=-1)  # shape (M, 2)

# 3. Prepare output
nlat_sst = len(lat_sst)
nlon_sst = len(lon_sst)

lat_flat = lat_aice.ravel()
lon_flat = lon_aice.ravel()

bad_mask = (
    np.isnan(lat_flat) | np.isnan(lon_flat) |
    np.isinf(lat_flat) | np.isinf(lon_flat) |
    (lat_flat > 91) | (lat_flat < -91) |
    (lon_flat > 361) | (lon_flat < -1)
)
valid_coords = ~bad_mask
lat_valid = lat_flat[valid_coords]
lon_valid = lon_flat[valid_coords]
source_coords = np.stack([lat_valid, lon_valid], axis=-1)

nm = aice.shape[0]
nt = aice.shape[1]

aice_regridded_all = np.empty((nm, nt, nlat_sst, nlon_sst), dtype=np.float32)

for i in range(nm):
    for j in range(nt):
        aiceij = aice[i,j,:,:]
        aice_regridded = np.empty((nlat_sst, nlon_sst))
        
        values = aiceij.ravel()
        values_valid = values[valid_coords]

        interpolated = griddata(points=source_coords,values=values_valid,xi=target_points,method='linear',fill_value=np.nan)

        aice_regridded = interpolated.reshape((nlat_sst, nlon_sst))
        
        aice_regridded_all[i, j, :, :] = aice_regridded
        


'''
#------------------------------------------------------
# plot
#------------------------------------------------------
   
# List of data and metadata
aice_regridded = aice_regridded_all[0,0,:,:]

datasets = [
    {"data": aice_regridded[ :, :], "lat": lat_sst, "lon": lon_sst, "title": "aice on SST grid"},
    {"data": aice[0:, :],        "lat": lat_aice, "lon": lon_aice, "title": "aice original grid"},
]

# Create figure and axes
fig, axes = plt.subplots(1, 2, figsize=(20, 10),
                         subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

# Loop through datasets and plot
for ax, ds in zip(axes, datasets):
    ax.set_global()
    ax.coastlines()

    im = ax.pcolormesh(ds["lon"], ds["lat"], ds["data"],
                       cmap=cmocean.cm.ice,
                       shading='auto',
                       transform=ccrs.PlateCarree())
    
    im.set_clim(0, 1)

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
cbar.set_label('aice')
        

condition = (lat_sst < 0)
aice_regridded[condition] = 0
aice_regridded = np.nan_to_num(aice_regridded, nan=0.0)
aice_regridded[aice_regridded <= 0] = 0
aice_regridded[aice_regridded > 0] = 1


fig,ax = plt.subplots(1, 1, figsize=(16, 16), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

data = aice_regridded

ax.set_global()
ax.coastlines()


im = ax.pcolormesh(lon_sst, lat_sst, data, cmap=cmocean.cm.balance,
                    shading='nearest', transform=ccrs.PlateCarree())
im.set_clim(-1,1)
ax.set_xticks(np.arange(-180,181, 60))   # Longitude labels
ax.set_yticks(np.arange(-90, 91, 30))  # Latitude labels
ax.xaxis.set_tick_params(rotation=45)  # Rotate longitude labels for better visibility
ax.yaxis.set_tick_params(rotation=45)  # Rotate latitude labels

# Add colorbar
cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)
cbar.set_label('')
'''

#------------------------------------------------------
#------------------------------------------------------
# V. SAVE
#------------------------------------------------------
#------------------------------------------------------

savepath = '/cofast/lhoffman/cesmle/cnn_slowdown/manuscript/d1/aice_regridded_cesm2le_sstgrid_jja_1990-2100.nc'
ds = xr.Dataset(
    {
        "aice_regridded_all":(("nm","nt","nx","ny"),aice_regridded_all),
        "lat_sst":(("nx",),lat_sst),
        "lon_sst":(("ny",),lon_sst),
    },
    coords={
        "nx":np.arange(aice_regridded_all.shape[2]), #lat index
        "ny":np.arange(aice_regridded_all.shape[3]), #lon index
        "nm":np.arange(aice_regridded_all.shape[0]), #ensembles
        "nt":np.arange(aice_regridded_all.shape[1]), #time

    },
)

#save to NetCDF file
encoding = {var: {"zlib":True,"coJmplevel":4} for var in ds.data_vars}
ds.to_netcdf(savepath,format='NETCDF4')
print(f'NetCDF file saved to {savepath}')



