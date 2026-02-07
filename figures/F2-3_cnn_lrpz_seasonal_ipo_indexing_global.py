
#------------------------------------------------------
#------------------------------------------------------
# ROOT PATH
rootpath = '/cofast/lhoffman/slowdown/'
#------------------------------------------------------
#------------------------------------------------------


# #set up environment
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
from scipy.ndimage import rotate, shift
from scipy.stats import ttest_1samp
from scipy.ndimage import gaussian_filter
import math 

#other
from datetime import datetime
from datetime import timedelta

#machine learning
#------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.regularizers import l2, l1  # or l1, or l1_l2

import keras.utils
from keras import regularizers

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import label_binarize

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
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle

#colorbars
#------------------
import cmocean 

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

#------------------------------------------------------
#------------------------------------------------------
# 0. LAT, LON
#------------------------------------------------------
#------------------------------------------------------
filepath = rootpath+'b.e21.BHISTcmip6.f09_g17.LE2-1001.001.cam.h1.Z200.18500101-18591231.nc'
dataset = nc.Dataset(filepath,'r')
lat = np.array(dataset.variables['lat'])
lon = np.array(dataset.variables['lon'])
#------------------------------------------------------

#------------------------------------------------------
#------------------------------------------------------
# 0. IPO INDEX
#------------------------------------------------------
#------------------------------------------------------
filepath = rootpath+'D2_cesm2le_ipo_index_1850-2100.nc'
dataset = nc.Dataset(filepath,'r')
ipo_filtered = np.array(dataset.variables['ipo_filt_reshape'])
ipo_unfiltered = np.array(dataset.variables['ipo_unfiltered'])
years = np.arange(1850,2101)
y1990 = np.where(years==1991)[0][0]
y2040 = np.where(years==2041)[0][0]

#ipo_raw = np.nanmean(ipo_filtered[:,y1990:y2040,5:8],axis=2) #filtered
ipo_raw = np.nanmean(ipo_unfiltered[5:8,:,y1990:y2040],axis=0) #unfiltered

#standardize IPO
ipo_standardized = (ipo_raw - np.nanmean(ipo_raw,axis=1,keepdims=True))/np.nanstd(ipo_raw,axis=1,keepdims=True)
ipo_labels = np.zeros_like(ipo_standardized,dtype=int)
ipo_threshold = 0
ipo_labels[ipo_standardized >=  ipo_threshold] =  1   # positive IPO
ipo_labels[ipo_standardized <= -ipo_threshold] = -1   # negative IPO
ipo_jja = ipo_standardized
#------------------------------------------------------

#------------------------------------------------------
# TVT RESHAPE, IPO
#------------------------------------------------------
#reshape for ensemble member grouping with different forcing
nt = ipo_jja.shape[1]
ipo = ipo_jja.reshape(10,10,nt)

#train-test split(80-10-10 for t-v-t)
#training
ipo_tri = ipo[:,2:,:]
ipo_trr = ipo_tri.reshape(80,nt)
ipo_tr = ipo_trr.reshape(80*nt)

del ipo_tri, ipo_trr

#validation
ipo_vali =  ipo[:,1,:]
ipo_var = ipo_vali.reshape(10,nt)
ipo_va = ipo_var.reshape(10*nt)

del ipo_vali, ipo_var

#testing
ipo_tei =  ipo[:,0,:]
ipo_ter = ipo_tei.reshape(10,nt)
ipo_te = ipo_ter.reshape(10*nt)

del ipo_tei, ipo_ter, ipo
#------------------------------------------------------

#------------------------------------------------------
# TVT RESHAPE, IPO LABELS
#------------------------------------------------------
#reshape for ensemble member grouping with different forcing
nt = ipo_labels.shape[1]
ipo_labels = ipo_labels.reshape(10,10,nt)

#train-test split(80-10-10 for t-v-t)
#training
ipo_tri_labels = ipo_labels[:,2:,:]
ipo_trr_labels = ipo_tri_labels.reshape(80,nt)
ipo_tr_labels = ipo_trr_labels.reshape(80*nt)

del ipo_tri_labels, ipo_trr_labels

#validation
ipo_vali_labels =  ipo_labels[:,1,:]
ipo_var_labels = ipo_vali_labels.reshape(10,nt)
ipo_va_labels = ipo_var_labels.reshape(10*nt)

del ipo_vali_labels, ipo_var_labels

#testing
ipo_tei_labels =  ipo_labels[:,0,:]
ipo_ter_labels = ipo_tei_labels.reshape(10,nt)
ipo_te_labels = ipo_ter_labels.reshape(10*nt)

del ipo_tei_labels, ipo_ter_labels, ipo_labels
#------------------------------------------------------

#------------------------------------------------------
#------------------------------------------------------
# I. LOAD TRAINING DATA
#------------------------------------------------------
#------------------------------------------------------

# DATA
#------------------------------------------------------

#lat,lon
filepath = rootpath+'b.e21.BHISTcmip6.f09_g17.LE2-1001.001.cam.h1.Z200.18500101-18591231.nc'
dataset = nc.Dataset(filepath,'r')
lat = np.array(dataset.variables['lat'])
lon = np.array(dataset.variables['lon'])

# slowdown
loadpath = rootpath+'D2_cnn_cesm2le_sieTREND_seasons_TVT_1990-2100.nc'
dataset = nc.Dataset(loadpath,'r')
sie_tr_slowdown = dataset.variables['sie_tr_slowdown']
sie_te_slowdown = dataset.variables['sie_te_slowdown']
sie_va_slowdown = dataset.variables['sie_va_slowdown']

# siextentn
loadpath = rootpath+'D2_cnn_cesm2le_siextent_seasons_TVT_1990-2100.nc'
dataset = nc.Dataset(loadpath,'r')
sie_tr = dataset.variables['sie_tr']
sie_te = dataset.variables['sie_te']
sie_va = dataset.variables['sie_va']

# ** SEASONAL SST
loadpath = rootpath+'D2_cesm2le_sst_seasonal_TVT_1990-2040.nc'
dataset = nc.Dataset(loadpath,'r')
sst_tr_year = dataset.variables['sst_tr']
sst_te_year = dataset.variables['sst_te']
sst_va_year = dataset.variables['sst_va']


#load landmask: zeros for ocean, ones for land [192,288]
loadpath = rootpath+'cnn_cesm2le_landmask.nc'
dataset = nc.Dataset(loadpath,'r')
landmask = dataset.variables['landmask']

#landmask
lm_expand_dims = np.array(landmask)[np.newaxis,np.newaxis,:,:]
lm_expand_dims = np.repeat(lm_expand_dims,sst_te_year.shape[0],axis=0)
landmask_te = np.repeat(lm_expand_dims,sst_te_year.shape[1],axis=1)
landmask_va = np.repeat(lm_expand_dims,sst_va_year.shape[1],axis=1)
landmask_tr = np.repeat(lm_expand_dims,sst_tr_year.shape[1],axis=1)

del lm_expand_dims
# ----------------------------------------------------------------

#------------------------------------------------------
#------------------------------------------------------
# II. DEFINE INPUTS & OUTPUTS
#------------------------------------------------------
#------------------------------------------------------

#----------------------------------------
#TRAINING DATA: CESM2-LE MODELS 
#define training and test variables
#----------------------

output_sie_test = np.array(sie_te_slowdown)
input_landmask_test = 1-landmask_te

output_sie_va = np.array(sie_va_slowdown)
input_landmask_va = 1-landmask_va

output_sie_tr = np.array(sie_tr_slowdown)
input_landmask_tr = 1-landmask_tr


del sie_te_slowdown
del sie_tr_slowdown
del sie_va_slowdown


# --------------------------
# LOOP OVER ABLATION + RUNS
# --------------------------
experiments = range(7)
n_runs = 5
splits_list = ["train", "val", "test"]

for k in range(4,5):   # your selected experiment
    # holders to average ACROSS seeds
    seed_analysis_pos      = [] 
    seed_analysis_neg      = []
    seed_analysis_neutral      = []
    seed_analysis_all  = [] 

    for r_idx in range(n_runs):
        #----------------------------------------
        #CHOOSE INPUTS
        #----------------------------------------
        #set land to -10 in SST (land == 1)
        input_sst_test = np.where(landmask_te == 1, -10, sst_te_year)
        input_sst_tr = np.where(landmask_tr == 1, -10, sst_tr_year)
        input_sst_va = np.where(landmask_va == 1, -10, sst_va_year)

        sst_tr_0 = np.where(landmask_tr == 1, np.nan, sst_tr_year)
        sst_te_0 = np.where(landmask_te == 1, np.nan, sst_te_year)
        sst_va_0 = np.where(landmask_va == 1, np.nan, sst_va_year)
        
        if k == 4: 
            #sst(lm) ALL SEASONS
            input_test = input_sst_test
            input_tr = input_sst_tr
            input_va = input_sst_va
            
            sst_tr_0 = sst_tr_0
            sst_te_0 = sst_te_0
            sst_va_0 = sst_va_0
            
        else: 
            #sst(lm) SON
            input_test = input_sst_test[k,:,:,:][None,:,:,:]
            input_tr = input_sst_tr[k,:,:,:][None,:,:,:]
            input_va = input_sst_va[k,:,:,:][None,:,:,:]
            
            sst_tr_0 = sst_tr_0[k,:,:,:][None,:,:,:]
            sst_te_0 = sst_te_0[k,:,:,:][None,:,:,:]
            sst_va_0 = sst_va_0[k,:,:,:][None,:,:,:]
            
        # --------------------------
        # reshape / transpose
        # --------------------------
        input_TR = np.array(input_tr)
        x_train_z200 = input_TR.transpose(1,2,3,0)  # [N, lat, lon, C]

        y_train = output_sie_tr.T  # [N,]
        loadpath_model = f"/cofast/lhoffman/slowdown/M1_model_cnn_masking_EI.{k}_R{r_idx}_seasonal_globalSTD.h5"
        model = tf.keras.models.load_model(loadpath_model)

        # LRP file (reshape to [N,lat,lon,C])
        loadpath = f'/cofast/lhoffman/slowdown/M2_EI.{k}-{r_idx}_lrpz_seasonal_global.nc'
        dataset = nc.Dataset(loadpath,'r')
        analysis = np.array(dataset.variables['analysis_2d'])  # [40,100,192,288,C]
        ne, nt, nx, ny, nch = analysis.shape
        reshaped_analysis = analysis.reshape(ne*nt,nx,ny,nch)
        del analysis

        # --------------------------
        # PR threshold & predictions
        # --------------------------
        y_true_bin = y_train if y_train.ndim == 1 else np.argmax(y_train, axis=1)
        y_scores   = model.predict(x_train_z200, verbose=0)
        precision, recall, thresholds = precision_recall_curve(y_true_bin, y_scores)
        idx_intersect = np.argmin(np.abs(precision[:-1] - recall[:-1]))
        threshold_intersect = thresholds[idx_intersect]

        y_true = y_train[:, None]
        y_pred = (y_scores >= threshold_intersect).astype(int)

        # --------------------------
        # TP mask & composites (first 4000 as in your code)
        # --------------------------
        y_truei = y_true[:4000]
        y_predi = y_pred[:4000]
        x_train = x_train_z200[:4000,:,:,:]
        ipo_i = ipo_tr_labels[:4000]

        tp_mask = (y_truei == 1) & (y_predi == 1)
        mask_pos = tp_mask[:,0] & (ipo_i == 1)
        mask_neg = tp_mask[:,0] & (ipo_i == -1)
        mask_all = tp_mask[:,0]
        
        analysis_tp_pos     = reshaped_analysis[mask_pos]
        analysis_tp_neg     = reshaped_analysis[mask_neg]
        analysis_tp_all     = reshaped_analysis[mask_all]

        # --- per-channel t-test -> significance masks + masked means ---
        analysis_mask_pos, analysis_mask_neg = [], []
        analysis_mask_neutral, analysis_mask_all = [], []

        axi = analysis_tp_pos.shape[3]  # channels

        for ch in range(axi):
            values_pos     = analysis_tp_pos[:, :, :, ch]
            values_neg     = analysis_tp_neg[:, :, :, ch]
            values_all     = analysis_tp_all[:, :, :, ch]

            _, p_pos     = ttest_1samp(values_pos,     popmean=0, axis=0, nan_policy='omit')
            _, p_neg     = ttest_1samp(values_neg,     popmean=0, axis=0, nan_policy='omit')
            _, p_all     = ttest_1samp(values_all,     popmean=0, axis=0, nan_policy='omit')

            sig_pos     = (p_pos     < 0.05)
            sig_neg     = (p_neg     < 0.05)
            sig_all     = (p_all     < 0.05)

            mean_pos     = np.nanmean(values_pos,     axis=0)
            mean_neg     = np.nanmean(values_neg,     axis=0)
            mean_all     = np.nanmean(values_all,     axis=0)

            # store as masked arrays so cross-seed averaging can ignore non-sig cells
            analysis_mask_pos.append(     np.ma.array(mean_pos,     mask=~sig_pos))
            analysis_mask_neg.append(     np.ma.array(mean_neg,     mask=~sig_neg))
            analysis_mask_all.append(     np.ma.array(mean_all,     mask=~sig_all))

        analysis_masked_all = np.array(analysis_mask_all)              # [C, lat, lon] masked
        analysis_masked_all = np.ma.filled(analysis_masked_all, np.nan)  # mask→NaN for averaging
        analysis_masked_pos = np.array(analysis_mask_pos)              # [C, lat, lon] masked
        analysis_masked_pos = np.ma.filled(analysis_masked_pos, np.nan)  # mask→NaN for averaging
        analysis_masked_neg = np.array(analysis_mask_neg)              # [C, lat, lon] masked
        analysis_masked_neg = np.ma.filled(analysis_masked_neg, np.nan)  # mask→NaN for averaging

        
        seed_analysis_all.append(analysis_masked_all)       # [C,lat,lon] (NaN where non-sig)
        seed_analysis_pos.append(analysis_masked_pos)       # [C,lat,lon] (NaN where non-sig)
        seed_analysis_neg.append(analysis_masked_neg)       # [C,lat,lon] (NaN where non-sig)

        
    # ==============================
    # AVERAGE ACROSS SEEDS (r_idx)
    # ==============================
    seed_analysis_all = np.stack(seed_analysis_all, axis=0)   # [S, C, lat, lon]
    seed_analysis_pos = np.stack(seed_analysis_pos, axis=0)   # [S, C, lat, lon]
    seed_analysis_neg = np.stack(seed_analysis_neg, axis=0)   # [S, C, lat, lon]


    analysis_mean_all_seed = np.nanmean(seed_analysis_all, axis=0)    # [C,lat,lon]
    analysis_mean_pos_seed = np.nanmean(seed_analysis_pos, axis=0)    # [C,lat,lon]
    analysis_mean_neg_seed = np.nanmean(seed_analysis_neg, axis=0)    # [C,lat,lon]


    # ==============================
    # PLOTTING
    # ==============================
    channel_names = ['JJA','MAM','DJF','SON']
    xi = analysis_mean_all_seed.shape[0]  # channels from input composite
    channel_names = channel_names[:xi]


    percentile = 95
    sigma_spatial = 3
    pctl = np.nanpercentile(np.abs(analysis_mean_pos_seed), percentile) + 1e-12

    fig, axes = plt.subplots(3, xi, figsize=(50, 12), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
    plt.subplots_adjust(wspace=0.05, hspace=0.15)

    #------------------------------------------------------
    # plot IPO positive
    #------------------------------------------------------

    for i in range(xi):
        ax = axes[0, i] if xi > 1 else axes[0]
        data = analysis_mean_pos_seed[i, :, :]                       # [lat,lon]
        normalized = np.clip(np.abs(data) / pctl, 0.0, 1.0)          # 0..1
        smoothed = gaussian_filter(normalized, sigma=sigma_spatial)

        ax.set_global()
        ax.coastlines(color='white')
        im = ax.pcolormesh(lon, lat, smoothed, cmap="gnuplot2",
                            shading='auto', transform=ccrs.PlateCarree(),
                            vmin=0.0, vmax=1.0)
        if 'Sea Surface Temperature' in channel_names[i]:
            ax.add_feature(cfeature.LAND, facecolor='gray', edgecolor='none')

        ax.set_yticks(np.arange(-90, 91, 30))
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')

        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label(f"LRP (|·|/{percentile}th) · {channel_names[i]}")
        
    #------------------------------------------------------
    # plot IPO negative
    #------------------------------------------------------

    for i in range(xi):
        ax = axes[1, i] if xi > 1 else axes[1]
        data = analysis_mean_neg_seed[i, :, :]                       # [lat,lon]
        normalized = np.clip(np.abs(data) / pctl, 0.0, 1.0)          # 0..1
        smoothed = gaussian_filter(normalized, sigma=sigma_spatial)

        ax.set_global()
        ax.coastlines(color='white')
        im = ax.pcolormesh(lon, lat, smoothed, cmap="gnuplot2",
                            shading='auto', transform=ccrs.PlateCarree(),
                            vmin=0.0, vmax=1.0)
        if 'Sea Surface Temperature' in channel_names[i]:
            ax.add_feature(cfeature.LAND, facecolor='gray', edgecolor='none')

        ax.set_yticks(np.arange(-90, 91, 30))
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')

        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label(f"LRP (|·|/{percentile}th) · {channel_names[i]}")
        
        
    #------------------------------------------------------
    # plot ALL
    #------------------------------------------------------

    for i in range(xi):
        ax = axes[2, i] if xi > 1 else axes[2]
        data = analysis_mean_all_seed[i, :, :]                       # [lat,lon]
        normalized = np.clip(np.abs(data) / pctl, 0.0, 1.0)          # 0..1
        smoothed = gaussian_filter(normalized, sigma=sigma_spatial)

        ax.set_global()
        ax.coastlines(color='white')
        im = ax.pcolormesh(lon, lat, smoothed, cmap="gnuplot2",
                            shading='auto', transform=ccrs.PlateCarree(),
                            vmin=0.0, vmax=1.0)
        if 'Sea Surface Temperature' in channel_names[i]:
            ax.add_feature(cfeature.LAND, facecolor='gray', edgecolor='none')

        ax.set_yticks(np.arange(-90, 91, 30))
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')

        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label(f"LRP (|·|/{percentile}th) · {channel_names[i]}")


