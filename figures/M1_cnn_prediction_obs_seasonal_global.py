
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
from scipy.stats import chi2_contingency, fisher_exact
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
from sklearn.linear_model import LogisticRegression


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
# 0. LAT, LON
#------------------------------------------------------
filepath = rootpath+'b.e21.BHISTcmip6.f09_g17.LE2-1001.001.cam.h1.Z200.18500101-18591231.nc'
dataset = nc.Dataset(filepath,'r')
lat = np.array(dataset.variables['lat'])
lon = np.array(dataset.variables['lon'])
#------------------------------------------------------


#------------------------------------------------------
# 0. NINO3.4 INDEX (use July to represent JJA)
#------------------------------------------------------
loadpath = rootpath+'X1_ersstv_nino34_index_labels_0.4threshold_1854-2024.nc'
dataset = nc.Dataset(loadpath,'r')
nino_jja = np.array(dataset.variables['nino34_jja'])
nino_labels_jja = np.array(dataset.variables['nino_labels_jja'])
years = np.arange(1854,2025)
y1990 = np.where(years==1991)[0][0]
y2016 = np.where(years==2016)[0][0]
nino34 = nino_jja[y1990:y2016,1]
nino34_labels = nino_labels_jja[y1990:y2016,1]


years = np.arange(1854, 2025)
months = np.arange(1, 13)
time_series = np.array([(y, m) for y in years for m in months])
y1990 = np.where(time_series[:,0] == 1991)[0][0]
nino34F = np.array(dataset.variables['nino34'][y1990:,])
nino34_labelsF = np.array(dataset.variables['labels'][y1990:,])
#------------------------------------------------------


#------------------------------------------------------
# 0. IPO INDEX (use July to represent JJA)
#------------------------------------------------------
loadpath = rootpath+'X1_ersstv_ipo_index_labels_0threshold_1854-2024.nc'
dataset = nc.Dataset(loadpath,'r')
ipo = np.array(dataset.variables['ipo_filtered'])
ipo_labels = np.array(dataset.variables['labels'])
ipo_reshape = ipo.reshape(2052//12,12)
ipo_labels_reshape = ipo_labels.reshape(2052//12,12)

years = np.arange(1854,2025)
y1990 = np.where(years==1991)[0][0]
y2016 = np.where(years==2016)[0][0]

ipo = ipo_reshape[y1990:y2016,6]
ipo_labels = ipo_labels_reshape[y1990:y2016,1]

years = np.arange(1854, 2025)
months = np.arange(1, 13)
time_series = np.array([(y, m) for y in years for m in months])
y1990 = np.where(time_series[:,0] == 1991)[0][0]
ipoF = np.array(dataset.variables['ipo_filtered'])[0,y1990:]
ipo_labelsF = np.array(dataset.variables['labels'])[0,y1990:]
#------------------------------------------------------


#------------------------------------------------------
# 0. SST OBS
#------------------------------------------------------
#linear forced
loadpath = rootpath+'D2_ersstv5_sst_seasonal_TVT_1979-2024.nc'


dataset = nc.Dataset(loadpath,'r')
years = np.arange(1990,2025)
y2016 = np.where(years==2015)[0][0]

sst = dataset.variables['sst']
sst16 = dataset.variables['sst'][:,:y2016,:,:]
#------------------------------------------------------

#------------------------------------------------------
# 0. SLOWDOWN OBS
#------------------------------------------------------
loadpath = rootpath+'sie_nsidc_slowdowns_1990-2015.nc'
dataset = nc.Dataset(loadpath,'r')
slowdown = dataset.variables['sep_slowdown'][1:]
trend_slowdown = dataset.variables['sep_linear_trend']

output_sie_test = np.array(slowdown)
#------------------------------------------------------

#------------------------------------------------------
# 0. LANDMASK
#------------------------------------------------------
#load landmask: zeros for ocean, ones for land [192,288]
loadpath = rootpath+'cnn_cesm2le_landmask.nc'
dataset = nc.Dataset(loadpath,'r')
landmask = dataset.variables['landmask']

#landmask
lm_expand_dims = np.array(landmask)[np.newaxis,np.newaxis,:,:]
lm_expand_dims = np.repeat(lm_expand_dims,sst16.shape[0],axis=0)
landmask_te = np.repeat(lm_expand_dims,sst16.shape[1],axis=1)
landmask_teF = np.repeat(lm_expand_dims,sst.shape[1],axis=1)
#------------------------------------------------------


ipo_cat_seeds = []
nino_cat_seeds = []   # list of int arrays in {-1,0,1} per seed
tp_mask_seeds  = []   # list of bool arrays per seed
y_pred_seeds = []

for k in range(4,5):
    for r_idx in range(5):

        #----------------------------------------
        #CHOOSE INPUTS
        #----------------------------------------
        
        #set land to -10 in SST (land == 1)
        input_sst_test = np.where(landmask_te == 1, -10, sst16)
        input_sst_test = np.where(np.isnan(input_sst_test), -10, input_sst_test)
        
        #future prediction
        input_sst_test_F = np.where(landmask_teF == 1, -10, sst)
        input_sst_test_F = np.where(np.isnan(input_sst_test_F), -10, input_sst_test_F)

        #sst(lm)
        if k == 4: 
            #sst(lm) ALL SEASONS
            input_test = input_sst_test
            input_test_F = input_sst_test_F
            
        else: 
            #sst(lm) 
            input_test = input_sst_test[k,:,:,:][None,:,:,:]
            input_test_F = input_sst_test_F[k,:,:,:][None,:,:,:]

        #----------------------------------------
        # FINAL INPUT, OUTPUT
        #----------------------------------------
        output_sie_TE = np.transpose(np.array(output_sie_test))
        input_z200_TE = np.array(input_test)
        input_z200_TEF = np.array(input_test_F)
        
        # TRANSPOSE
        #----------------------
        x_test = input_z200_TE.transpose(1,2,3,0)
        y_test = output_sie_test.T
        x_test_F = input_z200_TEF.transpose(1,2,3,0)
        #----------------------------------------


        #------------------------------------------------------
        # LOAD MODEL
        #------------------------------------------------------
        loadpath_model = f"/cofast/lhoffman/slowdown/M1_model_cnn_masking_EI.{k}_R{r_idx}_seasonal_globalSTD.h5"
        model = tf.keras.models.load_model(loadpath_model)
        #------------------------------------------------------
        
        #------------------------------------------------------
        # THRESHOLD FROM PRECISION-RECALL CURVE
        #------------------------------------------------------
        y_true = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test
        y_scores = model.predict(x_test)
        y_scores_F = model.predict(x_test_F)
        
        # Compute precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

        # Find where precision and recall are closest
        idx_intersect = np.argmin(np.abs(precision[:-1] - recall[:-1]))
        threshold_intersect = thresholds[idx_intersect]
        #------------------------------------------------------

        #----------------------------------------
        # PREDICT FROM TRAINING DATA
        #----------------------------------------
        y_true = y_test[:,np.newaxis]
        y_pred = (y_scores >= threshold_intersect).astype(int)
        y_pred_F = (y_scores_F >= threshold_intersect).astype(int)
        #----------------------------------------

        #------------------------------------------------------
        # TP: CORRECT SLOW DOWN
        #------------------------------------------------------
        y_truei = y_true
        y_predi = y_pred
        y_pred_Fi = y_pred_F
        
        nino_i = nino34
        nino_i_labels = nino34_labels
        ipo_i = ipo
        ipo_i_labels = ipo_labels
        
        nino_iF = nino34F
        nino_i_labelsF = nino34_labelsF
        ipo_iF = ipoF
        ipo_i_labelsF = ipo_labelsF

        for j in range (1):
            if j == 0:
                #true positive
                tp_mask = (y_truei == 1) & (y_predi == 1)
            elif j == 1:
                #true negative
                tp_mask = (y_truei == 0) & (y_predi == 0)
            elif j == 2:
                #false negative
                tp_mask = (y_truei == 1) & (y_predi == 0)
            else:
                #false positive
                tp_mask = (y_truei == 0) & (y_predi == 1)
                
        nino_mask = nino_i[tp_mask[:,0]]
        nino_mask_labels = nino_i_labels[tp_mask[:,0]]
        
        n_tp_mask = tp_mask[:, 0].astype(bool)               # flatten to [4000,]
        nino_cat_seeds.append(nino_i_labels.astype(int))
        tp_mask_seeds.append(n_tp_mask)
        y_pred_seeds.append(y_pred)
        
        mask = np.asarray(y_true, dtype=bool)
        years = np.arange(1991,2016)
        yearsF = np.arange(1991,2025)
        yearsF_index = np.arange(1991,2025,1/12)
        # Find contiguous spans where y_true == 1
        starts = np.where(np.diff(np.pad(mask.astype(int), (1,1))) == 1)[0]
        ends   = np.where(np.diff(np.pad(mask.astype(int), (1,1))) == -1)[0]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 5))

        # ---- (a) Predicted slowdown line + background shading of actual slowdown ----
        for s, e in zip(starts, ends):
            ax1.axvspan(years[s]-1.4, years[e-1]+.4, color="steelblue", alpha=0.35)

        ax1.plot(yearsF,y_pred_F, linestyle="--", linewidth=2.5, label="Future Predicted slowdown",color='green')
        ax1.plot(years, y_pred, linestyle="--", linewidth=2.5, label="Predicted slowdown",color='maroon')
        ax1.set_ylim(-0.1, 1.3)
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(["No slowdown", "Slowdown"])
        ax1.legend(frameon=False, loc="upper right")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # ---- (b) Niño index time series + same blue shading for actual slowdown ----
        for s, e in zip(starts, ends):
            ax2.axvspan(years[s]-1.4, years[e-1]+0.4, color="steelblue", alpha=0.35)

        ax2.plot(yearsF_index, nino34F, linewidth=2,color='black')
        #ax2.plot(years, nino34, linewidth=2,color='maroon')
        ax2.axhline(0, linewidth=1, color="k", alpha=0.4)
        ax2.set_ylabel("Niño 3.4 Index")
        ax2.set_xlabel("Year")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        
        # ---- (c) IPO index time series + same blue shading for actual slowdown ----
        for s, e in zip(starts, ends):
            ax3.axvspan(years[s]-1.4, years[e-1]+0.4, color="steelblue", alpha=0.35)

        ax3.plot(yearsF_index, ipoF, linewidth=2,color='black')
        #ax3.plot(years, ipo, linewidth=2,color='maroon')
        ax3.axhline(0, linewidth=1, color="k", alpha=0.4)
        ax3.set_ylabel("IPO Index")
        ax3.set_xlabel("Year")
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        


        #------------------------------------------------------
        # PLOT
        #------------------------------------------------------
        sst_anomaly = input_sst_test
        tp_mask_int = tp_mask.astype(int)
        slowdown_mask = y_true

        for i in range(3):
            if i == 0:
                sst_pos = np.nanmean(sst_anomaly[0,ipo_labels ==  1], axis=0)
                sst_neg = np.nanmean(sst_anomaly[0,ipo_labels == -1], axis=0)
                
                datasets = [
                {"data": sst_pos[:, :], "lat": lat, "lon": lon, "title": "SST in IPO+"},
                {"data": sst_neg[:, :],        "lat": lat, "lon": lon, "title": "SST in IPO-"},
            ]
                
            elif i == 1:
                sst_pos = np.nanmean(sst_anomaly[0,slowdown_mask[:,0] ==  1], axis=0)
                sst_neg = np.nanmean(sst_anomaly[0,slowdown_mask[:,0] == 0], axis=0)
                
                datasets = [
                {"data": sst_pos[:, :], "lat": lat, "lon": lon, "title": "SST in SLOWDOWN"},
                {"data": sst_neg[:, :],        "lat": lat, "lon": lon, "title": "SST in NO SLOWDOWN"},
            ]
                
            else:
                sst_pos = np.nanmean(sst_anomaly[0,tp_mask_int[:,0] ==  1], axis=0)
                sst_neg = np.nanmean(sst_anomaly[0,tp_mask_int[:,0] == 0], axis=0) 
                
                datasets = [
                {"data": sst_pos[:, :], "lat": lat, "lon": lon, "title": "SST in TP SLOWDOWN"},
                {"data": sst_neg[:, :],        "lat": lat, "lon": lon, "title": "SST in NO TP SLOWDOWN"},
            ]

            
            # Create figure and axes
            fig, axes = plt.subplots(1, 2, figsize=(20, 10),
                                    subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

            # Loop through datasets and plot
            for ax, ds in zip(axes, datasets):
                ax.set_global()
                ax.coastlines()

                im = ax.pcolormesh(ds["lon"], ds["lat"], ds["data"],
                                cmap=cmocean.cm.balance,
                                shading='auto',
                                transform=ccrs.PlateCarree())
                
                im.set_clim(-1,1)

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