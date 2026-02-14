#system
#------------------
import sys
import os
import csv
import pickle
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
from scipy.ndimage import gaussian_filter
import h5py
import math 

#other
from datetime import datetime
from datetime import timedelta

#machine learning
#------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import keras.utils
from keras.layers import Dense, Activation
import sklearn
from sklearn.model_selection import train_test_split
from scipy import stats, odr

#from keras import regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# XAI
import innvestigate
tf.compat.v1.disable_eager_execution()


#plotting
#------------------
#matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.cm import ScalarMappable

#colorbars
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

for l in range(9):

    #made from D2_TVT_z200_sst_u200_jja2sep_slowdown_1990-2040.py
    loadpath = f'/globalscratch/ucl/elic/lhoffman/K_robustness/D2_cnn_cesm2le_sieTREND_seasons_robustness_split{l}_TVT_1990-2100.nc'
    dataset = nc.Dataset(loadpath,'r')
    sie_tr_slowdown = dataset.variables['sie_tr_slowdown']

    loadpath = f'/globalscratch/ucl/elic/lhoffman/K_robustness/D2_cesm2le_sst_seasonal_robustness_split{l}_TVT_1990-2040.nc'
    dataset = nc.Dataset(loadpath,'r')
    sst_tr_year = dataset.variables['sst_tr']


    #load landmask: zeros for ocean, ones for land [192,288]
    loadpath = '/globalscratch/ucl/elic/lhoffman/E_ENSO_IPO/cnn_cesm2le_landmask.nc'
    dataset = nc.Dataset(loadpath,'r')
    landmask = dataset.variables['landmask']

    #landmask
    lm_expand_dims = np.array(landmask)[np.newaxis,np.newaxis,:,:]
    lm_expand_dims = np.repeat(lm_expand_dims,sst_tr_year.shape[0],axis=0)
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

    output_sie_tr = np.array(sie_tr_slowdown)
    input_landmask_tr = 1-landmask_tr


    del sie_tr_slowdown


    for e_idx, k in enumerate(range(1)):
        for r_idx in range(1):
            
            #set land to -10 in SST (land == 1)
            input_sst_tr = np.where(landmask_tr == 1, -10, sst_tr_year)
            
            #----------------------------------------
            #CHOOSE INPUTS
            #----------------------------------------
            if k == 4: 
                #sst(lm) ALL SEASONS
                input_tr = input_sst_tr
                
            else: 
                #sst(lm) 
                input_tr = input_sst_tr[k,:,:,:][None,:,:,:]
            #----------------------------------------
            
            #----------------------------------------
            # FINAL INPUT, OUTPUT
            #----------------------------------------
            #reshape to [N,lat,lon]
            nm = input_tr.shape[0]
            nx = input_tr.shape[1]
            ny = input_tr.shape[2]

            output_sie_TR = np.transpose(np.array(output_sie_tr))
            input_z200_TR = np.array(input_tr)

            # transpose
            x_train = input_z200_TR.transpose(1,2,3,0)
            
            #clip input range
            x_train[np.isclose(x_train, -10.0)] = 0.0
            #----------------------------------------
            
            #----------------------------------------
            #load model
            #----------------------------------------
            loadpath_model = f"/globalscratch/ucl/elic/lhoffman/K_robustness/M1_S_model_cnn_masking_EI.{k}_R{r_idx}_JJA_TVTsplit{l}_globalSTD.h5"
            model = tf.keras.models.load_model(loadpath_model)
            #----------------------------------------
            
            #------------------------------------------------------
            # eXplainable AI
            #------------------------------------------------------

            # --- REMOVE DENSE & DROPOUT LAYERS ---
            #--------------------------------------
            # Grab last Dense layer config
            last = model.layers[-1]
            config = last.get_config()
            config["activation"] = None  # remove sigmoid

            # Rebuild last Dense layer without sigmoid
            logits_layer = type(last).from_config(config)

            # Connect it to the previous layer's output
            x = model.layers[-2].output
            y = logits_layer(x)

            # New model without sigmoid
            model_no_dropout_sigmoid = Model(inputs=model.input, outputs=y)

            #transfer weights from old Dense to new Dense
            logits_layer.set_weights(last.get_weights())

            #analyzer
            analyzer = innvestigate.create_analyzer('lrp.z',model_no_dropout_sigmoid)

            # Define chunk size
            chunk_size = 100
            num_chunks = x_train.shape[0] // chunk_size  # Calculate number of chunks

            # Create a list to store analysis results for each chunk

            analysis_2di = []
            for i in range(num_chunks):
            #for i in range(2):
                # Get the start and end indices for the chunk
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size
                
                # Slice the data for the current chunk
                x_chunk = x_train[start_idx:end_idx]
                
                # Calculate SHAP values for a batch of samples (x_input)
                analysis_chunk = analyzer.analyze(x_chunk)
                analysis_2di.append(analysis_chunk)

            analysis_2d = np.array(analysis_2di)

            #save model predictions
            savepath = f'/globalscratch/ucl/elic/lhoffman/K_robustness/M2_S_EI.{k}-{r_idx}_lrpz_JJA_TVTsplit{l}_global.nc'

            #create xarray Dataset
            ds = xr.Dataset(
                    {
                        "analysis_2d": (("nch","nt","nx","ny","nc"), analysis_2d),
                        "lat": (("nx",), lat),
                        "lon": (("ny",), lon),
                        },
                    coords={
                        "nch": np.arange(analysis_2d.shape[0]),
                        "nt": np.arange(analysis_2d.shape[1]),
                        "nx": np.arange(analysis_2d.shape[2]),
                        "ny": np.arange(analysis_2d.shape[3]),
                        "nc": np.arange(analysis_2d.shape[4]),

                        
                    },
            )

            #save to NetCDF file
            encoding = {var: {"zlib": True, "complevel": 4} for var in ds.data_vars}
            ds.to_netcdf(savepath, format="NETCDF4", encoding=encoding)

            print(f"NetCDF file saved to {savepath}")
