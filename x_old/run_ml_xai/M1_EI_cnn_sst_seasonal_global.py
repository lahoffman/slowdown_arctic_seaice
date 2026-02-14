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
from sklearn.metrics import precision_recall_curve, matthews_corrcoef
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, brier_score_loss, log_loss
from sklearn.preprocessing import label_binarize
from sklearn.utils import resample


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
#------------------------------------------------------
#------------------------------------------------------


#------------------------------------------------------
#------------------------------------------------------
# I. LOAD TRAINING DATA
#------------------------------------------------------
#------------------------------------------------------

# DATA
#------------------------------------------------------
filepath = '/globalscratch/ucl/elic/lhoffman/E_ENSO_IPO/b.e21.BHISTcmip6.f09_g17.LE2-1001.001.cam.h1.Z200.18500101-18591231.nc'
dataset = nc.Dataset(filepath,'r')
lat = np.array(dataset.variables['lat'])
lon = np.array(dataset.variables['lon'])

#made from D2_TVT_z200_sst_u200_jja2sep_slowdown_1990-2040.py
loadpath = '/globalscratch/ucl/elic/lhoffman/H_ENSO_indexing/D2_cnn_cesm2le_sieTREND_seasons_TVT_1990-2100.nc'
dataset = nc.Dataset(loadpath,'r')
sie_tr_slowdown = dataset.variables['sie_tr_slowdown']
sie_te_slowdown = dataset.variables['sie_te_slowdown']
sie_va_slowdown = dataset.variables['sie_va_slowdown']

loadpath = '/globalscratch/ucl/elic/lhoffman/H_ENSO_indexing/D2_cesm2le_sst_seasonal_TVT_1990-2040.nc'
dataset = nc.Dataset(loadpath,'r')
sst_tr_year = dataset.variables['sst_tr']
sst_te_year = dataset.variables['sst_te']
sst_va_year = dataset.variables['sst_va']


#load landmask: zeros for ocean, ones for land [192,288]
loadpath = '/globalscratch/ucl/elic/lhoffman/E_ENSO_IPO/cnn_cesm2le_landmask.nc'
dataset = nc.Dataset(loadpath,'r')
landmask = dataset.variables['landmask']

#landmask
lm_expand_dims = np.array(landmask)[np.newaxis,np.newaxis,:,:]
lm_expand_dims = np.repeat(lm_expand_dims,sst_te_year.shape[0],axis=0)
landmask_te = np.repeat(lm_expand_dims,sst_te_year.shape[1],axis=1)
landmask_va = np.repeat(lm_expand_dims,sst_va_year.shape[1],axis=1)
landmask_tr = np.repeat(lm_expand_dims,sst_tr_year.shape[1],axis=1)

del lm_expand_dims

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

#------------------------------------------------------
#------------------------------------------------------
# II. DEFINE MODEL PARAMETERS
#------------------------------------------------------
#------------------------------------------------------

#define loss functions
#----------------------------------------
#define NRMSE function
def norm_root_mean_squared_error(y_true,y_pred):
    return  (K.sqrt(K.mean(K.square(y_pred - y_true))))/((K.std(y_true)))

#define pearson correlation
def corr(y_true, y_pred):
    return (K.sum((y_true-K.mean(y_true))*(y_pred-K.mean(y_pred))))/((K.sqrt(K.sum(K.square(y_true-K.mean(y_true)))))*(K.sqrt(K.sum(K.square(y_pred-K.mean(y_pred))))))

def focal_loss(gamma=2., alpha=0.25):
    
    def focal_loss_fixed(y_true, y_pred):
        # Ensure that y_true is float32 (same as y_pred)
        y_true = K.cast(y_true, dtype=tf.float32)
        
        # Clip y_pred to prevent log(0) errors (avoid NaNs)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        
        # Compute the cross entropy loss
        cross_entropy = -y_true * K.log(y_pred)
        
        # Compute the focal loss term
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.sum(loss, axis=-1)
    
    return focal_loss_fixed

#define model hyper parameters
#----------------------------------------
# LOSS FUNCTION
LOSS = 'binary_crossentropy'
METRIC = 'accuracy' # Metric for assessing model skill

# MODEL TRAINING
N_UNITS = 10 # number of nodes in layer
NUM_EPOCHS = 50 # Max number of times all of the data will be seen iteratively in training
BATCH_SIZE = 120 # Number of samples per epoch
ACTIVATION_FUNCTION = 'relu' #activation function [others are 'sigmoid','tanh','linear']
LEARNING_RATE = .0001 # Learning rate (think step size)
DROP = 0.2 # dropout rate
OPTIMIZER = 'adam' #gradient descent algorithm
RL2 = 0.00001

# define model build
#----------------------------------------
def build_cnn(nx, ny, nch, rl2=RL2, drop=DROP):    
    input_2d = Input(shape=(nx, ny, nch))
    x = Conv2D(32, (3,3), activation='relu', padding="same", kernel_regularizer=l2(rl2))(input_2d)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(64, (3,3), activation='relu', padding="same", kernel_regularizer=l2(rl2))(x)
    x = MaxPooling2D((2,2))(x)
    x = Flatten()(x)
    x = Dropout(drop)(x)
    output = Dense(1, activation='sigmoid')(x)
    return Model(inputs=input_2d, outputs=output)

# compute metrics
#----------------------------------------
def compute_metrics(y_true,y_scores):
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)

    if thresholds.size == 0:
        thr = 0.5
        prec_at_thr = precision[0]
        rec_at_thr  = recall[0]
    else:
        idx_intersect = np.argmin(np.abs(precision[:-1] - recall[:-1]))
        thr = thresholds[idx_intersect]
        prec_at_thr = precision[idx_intersect]
        rec_at_thr  = recall[idx_intersect] 
    
    y_pred = (y_scores >= thr).astype(int)
    
    # --- point metrics ---
    metrics = {
        "AUPRC": auprc,
        "Brier": brier_score_loss(y_true, y_scores),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Threshold": thr,
        "Prec@Thr": prec_at_thr,
        "Rec@Thr": rec_at_thr,
        "Prevalence": np.mean(y_true),
        "BinaryCrossEntropy": log_loss(y_true, y_scores, labels=[0,1]),
        "MCC": matthews_corrcoef(y_true, y_pred), 
        "AUROC": roc_auc_score(y_true, y_scores)
    }
    return metrics

#------------------------------------------------------
#------------------------------------------------------


# --------------------------
# LOOP OVER ABLATION + RUNS
# --------------------------

experiments = range(1)   # JJA, MAM, DJF, SON, all seasons
n_runs = 50               # 👈 number of seeds per experiment
splits_list = ["train", "val", "test"]

metrics_to_plot = [
    "AUPRC","AUROC",
    "Prec@Thr","Rec@Thr","F1",
    "Accuracy","MCC","Brier","BinaryCrossEntropy"
]

# storage: (M, E, S, R)
M = len(metrics_to_plot)
E = len(experiments)
S = len(splits_list)

vals   = np.full((M, E, S, n_runs), np.nan, dtype=np.float32)
ci_lo  = np.full((M, E, S), np.nan, dtype=np.float32)
ci_hi  = np.full((M, E, S), np.nan, dtype=np.float32)

for e_idx, k in enumerate(experiments):
    for r_idx in range(n_runs):
        # --- reproducibility seed ---
        seed = 42 + r_idx
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        #----------------------------------------
        #CHOOSE INPUTS
        #----------------------------------------
        #set land to -10 in SST (land == 1)
        input_sst_test = np.where(landmask_te == 1, -10, sst_te_year)
        input_sst_tr = np.where(landmask_tr == 1, -10, sst_tr_year)
        input_sst_va = np.where(landmask_va == 1, -10, sst_va_year)

        if k == 4: 
            #sst(lm) ALL SEASONS
            input_test = input_sst_test
            input_tr = input_sst_tr
            input_va = input_sst_va
            
        else: 
            #sst(lm) SON
            input_test = input_sst_test[k,:,:,:][None,:,:,:]
            input_tr = input_sst_tr[k,:,:,:][None,:,:,:]
            input_va = input_sst_va[k,:,:,:][None,:,:,:]
            
        
        #----------------------------------------

        #----------------------------------------
        # FINAL INPUT, OUTPUT
        #----------------------------------------
        #reshape to [N,lat,lon]
        nm = input_tr.shape[0]
        nx = input_tr.shape[1]
        ny = input_tr.shape[2]

        output_sie_TR = np.transpose(np.array(output_sie_tr))
        input_TR = np.array(input_tr)

        output_sie_VA = np.transpose(np.array(output_sie_va))
        input_VA = np.array(input_va)

        output_sie_TE = np.transpose(np.array(output_sie_test))
        input_TE = np.array(input_test)
        #----------------------------------------

        #----------------------------------------
        # TRANSPOSE
        #----------------------
        x_train = input_TR.transpose(1,2,3,0)
        x_val = input_VA.transpose(1,2,3,0)
        x_test = input_TE.transpose(1,2,3,0)

        y_train = output_sie_tr.T
        y_val = output_sie_va.T
        y_test = output_sie_test.T
        #----------------------------------------

        #----------------------------------------
        # CLASS WEIGHTS
        #----------------------------------------

        # Compute class weights
        y_train_binary = y_train
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train_binary), y=y_train_binary)

        # Convert to dictionary (needed for Keras)
        class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

        # Optional: Adjust weight of Class 1 (e.g., multiply by FRACTWEIGHT)
        FRACTWEIGHT = 1.5
        class_weights_dict[1] *= FRACTWEIGHT
        #----------------------------------------
        
        
        #----------------------------------------
        #----------------------------------------            
        #MODEL: BUILD NEURAL NETWORK
        #----------------------------------------
                    
        #shape information
        time_steps = x_train.shape[0]
        nx = x_train.shape[1]
        ny = x_train.shape[2]
        nch = x_train.shape[3]
        
        # model: build, compile
        model = build_cnn(nx, ny, nch, rl2=RL2, drop=DROP)
        model.compile(optimizer=OPTIMIZER,loss=keras.losses.BinaryFocalCrossentropy(alpha=0.75, gamma=2.0),metrics=['accuracy'])
        
        history = model.fit(
            x_train, y_train,
            epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=True,
            validation_data=(x_val,y_val),
            callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
            verbose=0
        )

        model.save(f"/globalscratch/ucl/elic/lhoffman/K_robustness/M1_model_cnn_masking_EI.{k}_R{r_idx}_JJA_globalSTD.h5") 
        # predict probabilities
        y_scores_train = model.predict(x_train).ravel()
        y_scores_val   = model.predict(x_val).ravel()
        y_scores_test  = model.predict(x_test).ravel()

        for s_idx, (split_name, (y_true, y_scores)) in enumerate({
            "train": (y_train, y_scores_train),
            "val":   (y_val,   y_scores_val),
            "test":  (y_test,  y_scores_test),
        }.items()):
            mvals = compute_metrics(y_true, y_scores)
            for m_idx, m in enumerate(metrics_to_plot):
                vals[m_idx, e_idx, s_idx, r_idx] = mvals[m]

    # --- after all runs for this experiment, compute across-run CIs ---
    for m_idx, m in enumerate(metrics_to_plot):
        for s_idx, split in enumerate(splits_list):
            run_vals = vals[m_idx, e_idx, s_idx, :]
            run_vals = run_vals[~np.isnan(run_vals)]
            if run_vals.size > 0:
                ci_lo[m_idx, e_idx, s_idx] = np.percentile(run_vals, 2.5)
                ci_hi[m_idx, e_idx, s_idx] = np.percentile(run_vals, 97.5)

# --------------------------
# PACK INTO DATASET
# --------------------------
ds = xr.Dataset(
    data_vars={
        "metric_value": (("metric","experiment","split","run"), vals),
        "ci_low":       (("metric","experiment","split"), ci_lo),
        "ci_high":      (("metric","experiment","split"), ci_hi),
    },
    coords={
        "metric": metrics_to_plot,
        "experiment": list(experiments),
        "split": splits_list,
        "run": np.arange(n_runs),
    },
    attrs={"desc": "Ablation CNN results: metrics per run with across-run CIs"}
)

savepath = "/globalscratch/ucl/elic/lhoffman/K_robustness/M1_EI.X_performance_TVT_multiRun_JJA_globalSTD.nc"
comp = {"zlib": True, "complevel": 4}
ds.to_netcdf(savepath, format="NETCDF4", encoding={v: comp for v in ds.data_vars})
print("Saved:", savepath)