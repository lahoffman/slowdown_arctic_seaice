

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

#import functions
#------------------
sys.path.append(rootpath+'functions/')
from functions_general import ncdisp
from functions_general import movmean

#------------------------------------------------------
#------------------------------------------------------

def bootstrap_mean_ci(vals, B=1000, alpha=0.05, random_state=None):
        """
        Bootstrap 95% CI (or 100*(1-alpha)%) for the mean of vals.
        Returns (mean, lower, upper). Ignores NaNs.
        """
        vals = np.asarray(vals)
        vals = vals[np.isfinite(vals)]
        n = vals.size
        if n == 0:
            return np.nan, np.nan, np.nan

        rng = np.random.default_rng(random_state)
        means = np.empty(B)
        for b in range(B):
            sample = rng.choice(vals, size=n, replace=True)
            means[b] = sample.mean()

        lower, upper = np.percentile(means, [100*alpha/2, 100*(1 - alpha/2)])
        return vals.mean(), lower, upper
    
def bootstrap_p_value_diff_means(vals1, vals2, B=5000, random_state=None):
    """
    Two-sided p-value for difference in means between vals1 and vals2,
    using a bootstrap/permutation-style null: "both samples come from
    the same parent distribution".

    vals1, vals2: 1D arrays (NaNs allowed; they are ignored)
    B: number of bootstrap samples
    """
    vals1 = np.asarray(vals1)
    vals2 = np.asarray(vals2)

    vals1 = vals1[np.isfinite(vals1)]
    vals2 = vals2[np.isfinite(vals2)]

    n1, n2 = vals1.size, vals2.size
    if n1 == 0 or n2 == 0:
        return np.nan

    # observed difference
    diff_obs = vals1.mean() - vals2.mean()

    # null: both samples from same parent → resample from pooled
    pooled = np.concatenate([vals1, vals2])
    rng = np.random.default_rng(random_state)

    diffs_null = np.empty(B)
    for b in range(B):
        sample = rng.choice(pooled, size=pooled.size, replace=True)
        s1 = sample[:n1]
        s2 = sample[n1:]
        diffs_null[b] = s1.mean() - s2.mean()

    # two-sided p-value: fraction of null diffs at least as extreme
    p = np.mean(np.abs(diffs_null) >= np.abs(diff_obs))
    return p

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

for k in range(4,5):   # selected experiment
    # holders to average ACROSS seeds
    seed_x_tp_all      = []  # each is [lat, lon, C]
    seed_analysis_all  = []  # each is [C, lat, lon] (masked→NaN)
    seed_x_tp_std      = []  # each is [lat, lon, C]
    seed_analysis_std  = []  # each is [C, lat, lon] (masked→NaN)

    for r_idx in range(n_runs):
        # --------------------------
        # CHOOSE INPUTS 
        # --------------------------
        
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
        reshaped_analysis = analysis.reshape(40*100, 192, 288, analysis.shape[4])
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

        tp_mask = (y_truei == 1) & (y_predi == 1)
        mask_all = tp_mask[:,0]

        # input composite (TP): [lat, lon, C]
        x_tp_mean_all = np.nanmean(x_train[mask_all],axis=0)
        x_tp_std_all = np.nanstd(x_train[mask_all],axis=0)
        
        # sie TP
        sie_tp = sie_tr[mask_all]

        # LRP composite with t-test p<0.05 masking → [C, lat, lon] (masked→NaN)
        analysis_tp_all = reshaped_analysis[mask_all]  # [N_tp, 192, 288, C]
        C = analysis_tp_all.shape[3]
        analysis_masked_all = []
        analysis_masked_std = []
        for ch in range(C):
            vals = analysis_tp_all[:,:,:,ch]  # [N_tp, lat, lon]
            _, p_all = ttest_1samp(vals, popmean=0, axis=0, nan_policy='omit')
            sig = (p_all < 0.05)
            mean_vals = np.nanmean(vals, axis=0)
            std_vals = np.nanstd(vals,axis = 0)
            analysis_masked_all.append(np.ma.array(mean_vals, mask=~sig))
            analysis_masked_std.append(np.ma.array(std_vals, mask=~sig))

        analysis_masked_std = np.array(analysis_masked_std)              # [C, lat, lon] masked
        analysis_masked_std = np.ma.filled(analysis_masked_std, np.nan)  # mask→NaN for averaging
        
        analysis_masked_all = np.array(analysis_masked_all)              # [C, lat, lon] masked
        analysis_masked_all = np.ma.filled(analysis_masked_all, np.nan)  # mask→NaN for averaging

        analysis_masked_std = np.array(analysis_masked_std)              # [C, lat, lon] masked
        analysis_masked_std = np.ma.filled(analysis_masked_std, np.nan)  # mask→NaN for averaging

        '''
        # --------------------------------------------------------
        # Figure PDF: (a) sie vs. slowdown (b) sie vs. TP slowdown
        fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(32,8))
        ax1 = axes[0]
        ax2 = axes[1]
        
        sie_anomaly_slowdown = sie_tr
        mask = y_train

        data_0 = sie_anomaly_slowdown[(mask == 0)]
        data_1 = sie_anomaly_slowdown[(mask == 1)]

        # Plot histograms
        ax1.hist(data_0, bins=30, alpha=0.6, label='no slowdown', density=True, color='blue')
        ax1.hist(data_1, bins=30, alpha=0.6, label='slowdown', density=True, color='red')
        ax1.set_xlabel(r'sea ice extent anomaly [M km$^2$]')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.set_title('(a) Slowdown')
        
        sie_anomaly_slowdown = sie_tr
        mask = y_predi[:,0]

        data_0 = sie_anomaly_slowdown[(mask == 0)]
        data_1 = sie_anomaly_slowdown[(mask == 1)]

        # Plot histograms
        ax2.hist(data_0, bins=30, alpha=0.6, label='no slowdown', density=True, color='blue')
        ax2.hist(data_1, bins=30, alpha=0.6, label='slowdown', density=True, color='red')
        ax2.set_xlabel(r'sea ice extent anomaly [M km$^2$]')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.set_title('(b) TP Slowdown')
        # --------------------------------------------------------
        
        # --------------------------------------------------------
        # Figure PDF: (a) SST vs. slowdown (b) SST vs. TP slowdown
        tit_sl = ["(a) JJA","(c) MAM","(e) DJF","(g) SON"]
        tit_tp = ["(b)","(d)","(f)","(h)"]
        fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(32,32))
        for l in range(1):
            ax1 = axes[0]
            ax2 = axes[1]
            
            lat_idx = np.where(lat > 60)[0]
            sst_Arctic = sst_tr_0[l, :, lat_idx, :]
            sst_Arctic_mean = np.nanmean(sst_Arctic, axis=(0, 2))
            mask = y_train

            data_0 = sst_Arctic_mean[(mask == 0)]
            data_1 = sst_Arctic_mean[(mask == 1)]

            # Plot histograms
            ax1.hist(data_0, bins=30, alpha=0.6, label='no slowdown', density=True, color='blue')
            ax1.hist(data_1, bins=30, alpha=0.6, label='slowdown', density=True, color='red')
            if l == 3:
                ax1.set_xlabel(r'SST anomaly, slowdown')
            ax1.set_ylabel('Density')
            ax1.legend()
            ax1.set_title(tit_sl[l])
            
            lat_idx = np.where(lat > 60)[0]
            sst_Arctic = sst_tr_0[l, :, lat_idx, :]
            sst_Arctic_mean = np.nanmean(sst_Arctic, axis=(0, 2))
            mask = y_predi[:,0]

            data_0 = sst_Arctic_mean[(mask == 0)]
            data_1 = sst_Arctic_mean[(mask == 1)]

            # Plot histograms
            ax2.hist(data_0, bins=30, alpha=0.6, label='no slowdown', density=True, color='blue')
            ax2.hist(data_1, bins=30, alpha=0.6, label='slowdown', density=True, color='red')
            if l == 3:
                ax2.set_xlabel(r'SST anomaly, TP slowdown')
            ax2.set_ylabel('Density')
            ax2.legend()
            ax2.set_title(tit_tp[l])
        # --------------------------------------------------------
    
        
        # --------------------------------------------------------
        # Figure SIE vs SST: (a) sie vs. slowdown (b) sie vs. TP slowdown
        lat_idx = np.where(lat > 60)[0]
        sst_Arctic = sst_tr_0[l, :, lat_idx, :]
        sst_Arctic_mean = np.nanmean(sst_Arctic, axis=(0, 2))
        sie_anomaly_slowdown = sie_tr
        
        x = np.asarray(sst_Arctic_mean).ravel()
        y = np.asarray(sie_tr).ravel()  
        
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
        axes.scatter(x, y, alpha=0.6)
        axes.set_xlabel("Arctic SST Anomaly (lat > 60°)")
        axes.set_ylabel("Sea ice extent anomaly")
        plt.show()
        # --------------------------------------------------------
        '''
        
        # collect for seed-mean        
        seed_x_tp_all.append(x_tp_mean_all)                 # [lat,lon,C]
        seed_analysis_all.append(analysis_masked_all) # [C,lat,lon] (NaN where non-sig)
        
        seed_x_tp_std.append(x_tp_std_all)                 # [lat,lon,C]
        seed_analysis_std.append(analysis_masked_std) # [C,lat,lon] (NaN where non-sig)

    # ==============================
    # AVERAGE ACROSS SEEDS (r_idx)
    # ==============================
    seed_x_tp_all = np.stack(seed_x_tp_all, axis=0)           # [S, lat, lon, C]
    seed_analysis_all = np.stack(seed_analysis_all, axis=0)   # [S, C, lat, lon]

    x_tp_mean_all_seed     = np.nanmean(seed_x_tp_all, axis=0)        # [lat,lon,C]
    analysis_mean_all_seed = np.nanmean(seed_analysis_all, axis=0)    # [C,lat,lon]

    seed_x_tp_std = np.stack(seed_x_tp_std, axis=0)           # [S, lat, lon, C]
    seed_analysis_std = np.stack(seed_analysis_std, axis=0)   # [S, C, lat, lon]

    x_tp_std_all_seed     = np.nanmean(seed_x_tp_std, axis=0)        # [lat,lon,C]
    analysis_std_all_seed = np.nanmean(seed_analysis_std, axis=0)    # [C,lat,lon]

    # ==============================
    # PLOTTING
    # ==============================
    if k == 4:
        channel_names = ['JJA','MAM','DJF','SON']

    else:
        channel_names = ['Sea Surface Temperature (SST)','Sea Ice Contentration (SIC)']

    xi = x_tp_mean_all_seed.shape[2]  # channels from input composite
    channel_names = channel_names[:xi]

    import string

    # Example seasonal labels for the first few
    season_labels = ['JJA', 'MAM', 'DJF', 'SON']

    # Generate alphabetical panel letters
    letters = list(string.ascii_lowercase)  # ['a', 'b', 'c', ..., 'z']

    # Combine them: (a) JJA, (b) MAM, (c) DJF, (d) SON, (e), (f), ...
    panel_titles = []
    for i in range(xi*3):
        '''
        if i < len(season_labels):
            panel_titles.append(f"({letters[i]}) {season_labels[i]}")
        else:
        '''
        panel_titles.append(f"({letters[i]})")
        
    # Row 1 (x_tp): raw values (no normalization)
    vmin_list = [-0.5, 0, -0.5, -0.5][:xi]
    vmax_list = [ 0.5,  1,  0.5,  0.5][:xi]

    # Row 2 (LRP): percentile-normalized (0..1) from averaged analysis
    percentile = 90
    sigma_spatial = 3
    pctl = np.nanpercentile(np.abs(analysis_mean_all_seed), percentile) + 1e-12

    # --- Define regions ---
    regions = {
        "Arctic": {"lat": (60, 90), "lon": (0, 360)},
        "Tropical Pacific (ENSO)": {"lat": (-10, 10), "lon": (170, 270)},
        "North Pacific (PDO)": {"lat": (15,55), "lon": (140, 230)},
        "Pacific (IPO)": {"lat": (-55, 55), "lon": (140, 270)},
        "Atlantic (AMO)": {"lat": (0, 60), "lon": (280, 350)},
        "Indian Ocean (IOD)": {"lat": (-35, 15), "lon": (50, 100)},
    }

    region_names = list(regions.keys())
    n_regions = len(region_names)

    # --- Assign a distinct color to each region ---
    cmap = plt.get_cmap("tab10")  # good qualitative palette
    region_colors = {name: cmap(i % 10) for i, name in enumerate(region_names)}

    # --- Setup lat/lon grid for masks ---
    lon2d, lat2d = np.meshgrid(lon, lat)

    mpl.rcParams.update({
    "font.size": 30,           # default text
    "axes.titlesize": 30,      # subplot titles
    "axes.labelsize": 30,      # x/y labels
    "xtick.labelsize": 30,     # tick labels
    "ytick.labelsize": 30,
    "legend.fontsize": 30,
    "figure.titlesize": 30,
    })
    
    # --- FIGURE LAYOUT: 3 rows × xi columns ---
    fig = plt.figure(figsize=(90, 40))
    gs = fig.add_gridspec(
        3, xi,
        height_ratios=[1, 1, 1],
        wspace=0.05, hspace=0.15
    )

    # =========================================================
    # ROW 1: mean field + contour (LRP > 0.4)
    # =========================================================
    for i in range(xi):
        ax = fig.add_subplot(gs[0, i], projection=ccrs.PlateCarree(central_longitude=180))
        data = x_tp_mean_all_seed[:, :, i]
        data = data.astype(float)
        data[data < -8] = np.nan

        ax.set_global()
        ax.coastlines()
        im = ax.pcolormesh(lon, lat, data, cmap=cmocean.cm.balance,
                        shading='auto', transform=ccrs.PlateCarree(),
                        vmin=-0.5, vmax=0.5)
        ax.add_feature(cfeature.LAND, facecolor='gray', edgecolor='none')

        ax.set_yticks(np.arange(-90, 91, 30))
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        ax.set_title(panel_titles[i])

        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label('SST')

        # --- Contours of relevance > 0.4 ---
        rel = analysis_mean_all_seed[i, :, :]
        rel_norm = np.clip(rel / pctl, 0.0, 1.0)
        rel_smooth = gaussian_filter(rel_norm, sigma=sigma_spatial)
        cs = ax.contour(lon, lat, rel_smooth, levels=[0.3],
                        colors='yellow', linewidths=1.5, transform=ccrs.PlateCarree())

    # =========================================================
    # ROW 2: normalized relevance maps + colored boxes
    # =========================================================
    for i in range(xi):
        ax = fig.add_subplot(gs[1, i], projection=ccrs.PlateCarree(central_longitude=180))
        data = analysis_mean_all_seed[i, :, :]
        normalized = np.clip(data / pctl, 0.0, 1.0)
        smoothed = gaussian_filter(normalized, sigma=sigma_spatial)

        #set relevance to nan on land
        data_x = x_tp_mean_all_seed[:, :, i]
        data_x = data_x.astype(float)
        smoothed[data_x < -8] = np.nan
        
        ax.set_global()
        ax.coastlines(color='white')
        im = ax.pcolormesh(lon, lat, smoothed, cmap="gnuplot2",
                        shading='auto', transform=ccrs.PlateCarree(),
                        vmin=0.0, vmax=1.0)
        ax.add_feature(cfeature.LAND, facecolor='gray', edgecolor='none')

        ax.set_yticks(np.arange(-90, 91, 30))
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        ax.set_title(panel_titles[i + xi])

        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label(f"LRP (|·|/{percentile}th)")

        # --- Draw colored boxes for regions ---
        for name, r in regions.items():
            lat_min, lat_max = r["lat"]
            lon_min, lon_max = r["lon"]
            color = region_colors[name]
            rect = Rectangle(
                (lon_min, lat_min),
                lon_max - lon_min,
                lat_max - lat_min,
                fill=False,
                edgecolor=color,
                linewidth=2.5,
                transform=ccrs.PlateCarree()
            )
            ax.add_patch(rect)

    # =========================================================
    # ROW 3: bar plots with matching colors
    # =========================================================
    
    B = 1000      # bootstrap samples
    alpha = 0.05  # 95% CI
    
    for i in range(xi):
        axb = fig.add_subplot(gs[2, i])

        data = analysis_mean_all_seed[i, :, :]
        normalized = np.clip(data / pctl, 0.0, 1.0)
        smoothed = gaussian_filter(normalized, sigma=sigma_spatial)

        #set relevance to nan on land
        data_x = x_tp_mean_all_seed[:, :, i]
        data_x = data_x.astype(float)
        smoothed[data_x < -8] = np.nan
        
        region_means_all = []
        region_ci_low_all = []
        region_ci_high_all = []

        region_means_hi = []
        region_ci_low_hi = []
        region_ci_high_hi = []


        for name in region_names:
            lat_min, lat_max = regions[name]["lat"]
            lon_min, lon_max = regions[name]["lon"]

            mask_region = (
                (lat2d >= lat_min) & (lat2d <= lat_max) &
                (lon2d >= lon_min) & (lon2d <= lon_max)
            )
            vals_all = smoothed[mask_region]
            mean_all, lower_all, upper_all = bootstrap_mean_ci(
            vals_all, B=B, alpha=alpha)
            
            region_means_all.append(mean_all)
            region_ci_low_all.append(mean_all - lower_all)
            region_ci_high_all.append(upper_all - mean_all)


        x = np.arange(len(region_names))
        width = 0.35
        bar_colors = [region_colors[name] for name in region_names]
        
        yerr_all = np.vstack([region_ci_low_all, region_ci_high_all])
        yerr_hi  = np.vstack([region_ci_low_hi,  region_ci_high_hi])
    
        # two bars per region
        axb.bar(
        x - width/2, region_means_all, width=width,
        yerr=yerr_all,
        color=bar_colors, alpha=0.4,
        edgecolor='black', capsize=3,
        label='Region mean' if i == 0 else None
        )

        axb.set_xticks(x)
        axb.set_xticklabels(region_names, rotation=45, ha='right', fontsize=30)
        axb.set_title(panel_titles[i+2*xi])
        if i == 0:
            axb.set_ylabel("Relevance")
        axb.set_ylim(0, 1)
        
        # --- Compute pairwise p-values ---
        n_regions = len(region_names)
        pvals = np.full((n_regions, n_regions), np.nan)

        for a in range(n_regions):
            for b in range(a + 1, n_regions):
                name_a = region_names[a]
                name_b = region_names[b]

                # Masks for the two regions
                lat_min_a, lat_max_a = regions[name_a]["lat"]
                lon_min_a, lon_max_a = regions[name_a]["lon"]
                mask_a = (
                    (lat2d >= lat_min_a) & (lat2d <= lat_max_a) &
                    (lon2d >= lon_min_a) & (lon2d <= lon_max_a)
                )

                lat_min_b, lat_max_b = regions[name_b]["lat"]
                lon_min_b, lon_max_b = regions[name_b]["lon"]
                mask_b = (
                    (lat2d >= lat_min_b) & (lat2d <= lat_max_b) &
                    (lon2d >= lon_min_b) & (lon2d <= lon_max_b)
                )

                vals_a = smoothed[mask_a]
                vals_b = smoothed[mask_b]
                p = bootstrap_p_value_diff_means(vals_a, vals_b, B=3000)
                pvals[a, b] = p
                pvals[b, a] = p

        # --- Format p-values into a short text table ---
        text_lines = []
        for a in range(n_regions):
            for b in range(a + 1, n_regions):
                text_lines.append(f"{region_names[a][:]}–{region_names[b][:]}: {pvals[a,b]:.3f}")
        text_str = "\n".join(text_lines)
        
        # --- Add text box on figure ---
        axb.text(
            0.55, 0.95, text_str, transform=axb.transAxes,
            fontsize=22, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7)
        )

    fig.tight_layout()
    plt.show()
    
    
   
    # =========================================================
    # ENSO REGIONS: bar plots (region means)
    # =========================================================
    # --- Define ENSO regions ---
    regions = {
        "Nino 1+2": {"lat": (-10,0), "lon": (270, 280)},
        "Nino 3": {"lat": (-5, 5), "lon": (210, 270)},
        "Nino 4": {"lat": (-5, 5), "lon": (160, 210)},
    }
    
    region_names = list(regions.keys())
    n_regions = len(region_names)

    # --- Assign a distinct color to each region ---
    cmap = plt.get_cmap("tab10")  # good qualitative palette
    region_colors = {name: cmap(i % 10) for i, name in enumerate(region_names)}
    
    fig_bar, axes_bar = plt.subplots(
        1, xi, figsize=(12 * xi, 10),  # adjust height if needed
        squeeze=False, sharey=True
    )
    axes_bar = axes_bar[0]

    B = 1000      # bootstrap samples
    alpha = 0.05  # 95% CI

    for i in range(xi):
        axb = axes_bar[i]

        data = analysis_mean_all_seed[i, :, :]
        normalized = np.clip(data / pctl, 0.0, 1.0)
        smoothed = gaussian_filter(normalized, sigma=sigma_spatial)

        region_means_all = []
        region_ci_low_all = []
        region_ci_high_all = []

        region_means_hi = []
        region_ci_low_hi = []
        region_ci_high_hi = []


        for name in region_names:
            lat_min, lat_max = regions[name]["lat"]
            lon_min, lon_max = regions[name]["lon"]

            mask_region = (
                (lat2d >= lat_min) & (lat2d <= lat_max) &
                (lon2d >= lon_min) & (lon2d <= lon_max)
            )
            vals_all = smoothed[mask_region]
            mean_all, lower_all, upper_all = bootstrap_mean_ci(
            vals_all, B=B, alpha=alpha)
            
            region_means_all.append(mean_all)
            region_ci_low_all.append(mean_all - lower_all)
            region_ci_high_all.append(upper_all - mean_all)


        x = np.arange(len(region_names))
        width = 0.35
        bar_colors = [region_colors[name] for name in region_names]
        
        yerr_all = np.vstack([region_ci_low_all, region_ci_high_all])
        yerr_hi  = np.vstack([region_ci_low_hi,  region_ci_high_hi])
    
        # two bars per region
        axb.bar(
        x - width/2, region_means_all, width=width,
        yerr=yerr_all,
        color=bar_colors, alpha=0.4,
        edgecolor='black', capsize=3,
        label='Region mean' if i == 0 else None
        )

        axb.set_xticks(x)
        axb.set_xticklabels(region_names, rotation=45, ha='right', fontsize=30)
        axb.set_ylim(0, 1)
        axb.set_title(panel_titles[i], fontsize=30)

        if i == 0:
            axb.set_ylabel("Relevance (normalized)", fontsize=30)
            #axb.legend(loc='upper left', fontsize=30)

        # --- Compute pairwise p-values ---
        n_regions = len(region_names)
        pvals = np.full((n_regions, n_regions), np.nan)

        for a in range(n_regions):
            for b in range(a + 1, n_regions):
                name_a = region_names[a]
                name_b = region_names[b]

                # Masks for the two regions
                lat_min_a, lat_max_a = regions[name_a]["lat"]
                lon_min_a, lon_max_a = regions[name_a]["lon"]
                mask_a = (
                    (lat2d >= lat_min_a) & (lat2d <= lat_max_a) &
                    (lon2d >= lon_min_a) & (lon2d <= lon_max_a)
                )

                lat_min_b, lat_max_b = regions[name_b]["lat"]
                lon_min_b, lon_max_b = regions[name_b]["lon"]
                mask_b = (
                    (lat2d >= lat_min_b) & (lat2d <= lat_max_b) &
                    (lon2d >= lon_min_b) & (lon2d <= lon_max_b)
                )

                vals_a = smoothed[mask_a]
                vals_b = smoothed[mask_b]
                p = bootstrap_p_value_diff_means(vals_a, vals_b, B=3000)
                pvals[a, b] = p
                pvals[b, a] = p

        # --- Format p-values into a short text table ---
        text_lines = []
        for a in range(n_regions):
            for b in range(a + 1, n_regions):
                text_lines.append(f"{region_names[a][:]}–{region_names[b][:]}: {pvals[a,b]:.3f}")
        text_str = "\n".join(text_lines)
        
        # --- Add text box on figure ---
        axb.text(
            0.55, 0.95, text_str, transform=axb.transAxes,
            fontsize=22, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7)
        )

    plt.tight_layout()
    plt.show()

    
    

        
        
        
        
