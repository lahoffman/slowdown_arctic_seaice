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
filepath = rootpath+'b.e21.BHISTcmip6.f09_g17.LE2-1001.001.cam.h1.Z200.18500101-18591231.nc'
dataset = nc.Dataset(filepath,'r')
lat = np.array(dataset.variables['lat'])
lon = np.array(dataset.variables['lon'])

loadpath = rootpath+'D2_cnn_cesm2le_sieTREND_seasons_TVT_1990-2100.nc'
dataset = nc.Dataset(loadpath,'r')
sie_tr_slowdown = dataset.variables['sie_tr_slowdown']
sie_te_slowdown = dataset.variables['sie_te_slowdown']
sie_va_slowdown = dataset.variables['sie_va_slowdown']

# ** SEASONAL
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
# LOOP OVER RUNS
# --------------------------

ipo_cat_seeds = []   # list of int arrays in {-1,0,1} per seed
tp_mask_seeds  = []   # list of bool arrays per seed

for k in range(4,5):
    for r_idx in range(5):

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
        # FINAL INPUT, OUTPUT
        #----------------------------------------
        #reshape to [N,lat,lon]
        nm = input_tr.shape[0]
        nx = input_tr.shape[1]
        ny = input_tr.shape[2]

        output_sie_TR = np.transpose(np.array(output_sie_tr))
        input_z200_TR = np.array(input_tr)

        output_sie_VA = np.transpose(np.array(output_sie_va))
        input_z200_VA = np.array(input_va)

        output_sie_TE = np.transpose(np.array(output_sie_test))
        input_z200_TE = np.array(input_test)
        
        # TRANSPOSE
        #----------------------
        x_train_z200 = input_z200_TR.transpose(1,2,3,0)
        x_val_z200 = input_z200_VA.transpose(1,2,3,0)
        x_test_z200 = input_z200_TE.transpose(1,2,3,0)

        y_train = output_sie_tr.T
        y_val = output_sie_va.T
        y_test = output_sie_test.T
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
        y_true = np.argmax(y_train, axis=1) if y_train.ndim > 1 else y_train
        y_scores = model.predict(x_train_z200)

        # Compute precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

        # Find where precision and recall are closest
        idx_intersect = np.argmin(np.abs(precision[:-1] - recall[:-1]))
        threshold_intersect = thresholds[idx_intersect]
        #------------------------------------------------------

        #----------------------------------------
        # PREDICT FROM TRAINING DATA
        #----------------------------------------
        y_true = y_train[:,np.newaxis]
        y_pred = (y_scores >= threshold_intersect).astype(int)
        #----------------------------------------

        #------------------------------------------------------
        # TP: CORRECT SLOW DOWN
        #------------------------------------------------------
        y_truei = y_true[:4000,]
        y_predi = y_pred[:4000,]
        ipo_i = ipo_tr[:4000]
        ipo_i_labels = ipo_tr_labels[:4000]

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
                
        ipo_mask = ipo_i[tp_mask[:,0]]
        ipo_mask_labels = ipo_i_labels[tp_mask[:,0]]
        
        n_tp_mask = tp_mask[:, 0].astype(bool)               # flatten to [4000,]
        ipo_cat_seeds.append(ipo_i_labels.astype(int))
        tp_mask_seeds.append(n_tp_mask)
        
        #------------------------------------------------------
        



    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------


    # -----------------------------
    # 1) Core stats (single event)
    # -----------------------------
    def event_dependence_by_phase(phase_labels, event_bool, phases=(-1,1), B=5000):
        """P(event | phase) with 95% CIs + chi2 + Fisher. No seeds."""
        ph = np.asarray(phase_labels).ravel().astype(int)
        evt = np.asarray(event_bool).ravel().astype(bool)
        n   = ph.size

        counts  = np.zeros((len(phases), 2), int)
        p_event = np.full(len(phases), np.nan, float)

        for i, v in enumerate(phases):
            m = (ph == v)
            if np.any(m):
                counts[i, 0] = np.sum(~evt[m])     # no
                counts[i, 1] = np.sum( evt[m])     # yes
                p_event[i]   = counts[i, 1] / counts[i].sum()

        _, p_chi2, _, _ = chi2_contingency(counts)
        pairwise = {}
        for i in range(len(phases)):
            for j in range(i+1, len(phases)):
                _, p = fisher_exact(counts[[i, j], :])
                pairwise[(phases[i], phases[j])] = p

        # time bootstrap
        boot = np.empty((B, len(phases)), float)
        for b in range(B):
            idx = np.random.randint(0, n, size=n)
            for i, v in enumerate(phases):
                m = (ph[idx] == v)
                boot[b, i] = np.mean(evt[idx][m]) if np.any(m) else np.nan

        ci_lo  = np.nanpercentile(boot,  2.5, axis=0)
        ci_hi  = np.nanpercentile(boot, 97.5, axis=0)

        return {"phases": phases, "p": p_event, "ci_lo": ci_lo, "ci_hi": ci_hi,
                "counts": counts, "chi2_p": p_chi2, "fisher": pairwise}

    # ---------------------------------------
    # 2) Core stats (multi-seed TP masks)
    # ---------------------------------------
    def _coerce_to_NxS(E, N):
        E = np.asarray(E)
        if E.ndim == 1: E = E[:, None]
        if E.shape[0] != N and E.shape[1] == N: E = E.T
        assert E.shape[0] == N, f"Expected (N,S) or (S,N); got {E.shape}, N={N}"
        return E

    def event_dependence_by_phase_multiseed(phase_labels, event_masks_seeds, phases=(-1,1), B=5000):
        """Seed-avg P(event|phase) with 95% CIs + chi2 + Fisher. event_masks_seeds is (N,S) bool."""
        ph = np.asarray(phase_labels).ravel().astype(int)
        E  = _coerce_to_NxS(event_masks_seeds, ph.size).astype(bool)
        N, S = E.shape

        per_seed_p = np.full((S, len(phases)), np.nan)
        per_seed_counts = np.zeros((S, len(phases), 2), int)

        for s in range(S):
            for i, v in enumerate(phases):
                m = (ph == v)
                if np.any(m):
                    per_seed_counts[s, i, 0] = np.sum(~E[m, s])
                    per_seed_counts[s, i, 1] = np.sum( E[m, s])
                    c = per_seed_counts[s, i].sum()
                    per_seed_p[s, i] = per_seed_counts[s, i, 1] / c if c > 0 else np.nan

        p_avg  = np.nanmean(per_seed_p, axis=0)
        counts = np.round(np.nanmean(per_seed_counts, axis=0)).astype(int)

        _, p_chi2, _, _ = chi2_contingency(counts)
        pairwise = {}
        for i in range(len(phases)):
            for j in range(i+1, len(phases)):
                _, p = fisher_exact(counts[[i, j], :])
                pairwise[(phases[i], phases[j])] = p

        # nested bootstrap (seeds + time)
        boot = np.empty((B, len(phases)), float)
        for b in range(B):
            seed_ids = np.random.randint(0, S, size=S)
            idx      = np.random.randint(0, N, size=N)
            vals = np.full((S, len(phases)), np.nan)
            for j, sid in enumerate(seed_ids):
                for i, v in enumerate(phases):
                    m = (ph[idx] == v)
                    vals[j, i] = np.mean(E[idx, sid][m]) if np.any(m) else np.nan
            boot[b] = np.nanmean(vals, axis=0)

        ci_lo  = np.nanpercentile(boot,  2.5, axis=0)
        ci_hi  = np.nanpercentile(boot, 97.5, axis=0)

        return {"phases": phases, "p": p_avg, "ci_lo": ci_lo, "ci_hi": ci_hi,
                "counts": counts, "chi2_p": p_chi2, "fisher": pairwise}

    # ---------------------------------------
    # 3) Minimal plots (overlay + stacked)
    # ---------------------------------------
    def plot_overlay_and_stacked(res_slow, res_tp, labels=("-','+")):
        p_s, lo_s, hi_s = res_slow["p"], res_slow["ci_lo"], res_slow["ci_hi"]
        p_t, lo_t, hi_t = res_tp["p"],   res_tp["ci_lo"],  res_tp["ci_hi"]
        x = np.arange(len(labels))

        # A) Overlay
        fig, ax = plt.subplots(figsize=(7,4))
        w = 0.35
        ax.bar(x - w/2, p_s, width=w, yerr=[p_s-lo_s, hi_s-p_s], capsize=5, color="lightgray", label="Slowdowns")
        ax.bar(x + w/2, p_t, width=w, yerr=[p_t-lo_t, hi_t-p_t], capsize=5, color="tab:blue",  label="TP slowdowns")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("P(event | IPO phase)")
        ax.set_title("Conditional probability by IPO phase")
        ax.legend(frameon=False)
        ax.set_ylim(0, max(np.nanmax(hi_s), np.nanmax(hi_t))*1.25)
        plt.tight_layout(); plt.show()

        # B) Stacked (predictable share)
        p_fn = np.clip(p_s - p_t, 0, 1)
        fig, ax = plt.subplots(figsize=(7,4))
        ax.bar(x, p_fn, color="lightgray", label="Unpredicted slowdowns (FN)")
        ax.bar(x, p_t, bottom=p_fn, color="tab:blue", label="Correct slowdowns (TP)")
        ax.errorbar(x, p_s, yerr=[p_s-lo_s, hi_s-p_s], fmt="none", capsize=5, color="k", alpha=0.6)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("P(slowdown | IPO phase)")
        ax.set_title("Predictable share of slowdowns by IPO phase")
        ax.legend(frameon=False)
        ax.set_ylim(0, np.nanmax(hi_s)*1.25)
        plt.tight_layout(); plt.show()

    # ---------------------------------------------------
    # 4) Minimal logistic regression (OR + McFadden R2)
    # ---------------------------------------------------
    def logit_or_r2(enso_labels, y_binary):
        """Odds ratio per +1 IPO step + McFadden R^2 (small but informative)."""
        enso = np.asarray(enso_labels, float).ravel()
        y    = np.asarray(y_binary,  int).ravel()
        X    = enso.reshape(-1,1)

        m = LogisticRegression(fit_intercept=True, solver="lbfgs")
        m.fit(X, y)

        beta1 = float(m.coef_[0][0])
        OR    = float(np.exp(beta1))

        def logL(y, p):
            p = np.clip(p, 1e-12, 1-1e-12)
            return float(np.sum(y*np.log(p) + (1-y)*np.log(1-p)))

        p_full = m.predict_proba(X)[:,1]
        ll_full = logL(y, p_full)
        p0 = float(y.mean())
        ll_null = logL(y, np.full(y.shape, p0, dtype=float))
        R2_mcf = 1 - (ll_full / ll_null)

        # Wald CI/p for slope
        x_c = enso - enso.mean()
        var_b1 = 1.0 / np.sum(p_full*(1-p_full)*x_c**2)
        se_b1  = float(np.sqrt(var_b1))
        z      = beta1 / se_b1
        p_val  = float(2*(1 - norm.cdf(abs(z))))
        OR_lo, OR_hi = float(np.exp(beta1 - 1.96*se_b1)), float(np.exp(beta1 + 1.96*se_b1))

        return {"OR": OR, "OR_CI": (OR_lo, OR_hi), "p": p_val, "R2_mcf": R2_mcf, "model": m}

    # ============================
    # ======= HOW TO USE =========
    # ============================

    # Inputs you already have:
    # nino_i_labels: (N,), values in {-1,0,1}
    # y_truei:       (N,), 0/1 slowdowns
    # tp_mask_seeds: (N,S) booleans, each col: (y_truei==1) & (y_pred_seed==1)

    phases = (-1,1)
    ipo   = np.asarray(ipo_i_labels).ravel()
    slow   = (np.asarray(y_truei).ravel() == 1)
    E      = _coerce_to_NxS(tp_mask_seeds, ipo.size).astype(bool)

    # A) Phase-based dependence
    res_slow = event_dependence_by_phase(ipo, slow, phases, B=5000)
    res_tp   = event_dependence_by_phase_multiseed(ipo, E, phases, B=5000)

    print("=== Slowdowns ===")
    print("P(slow|IPO):", res_slow["p"])
    print("95% CI:", res_slow["ci_lo"], res_slow["ci_hi"])
    print("Chi² p:", res_slow["chi2_p"], "Fisher:", res_slow["fisher"])

    print("\n=== TP slowdowns (seed-avg) ===")
    print("P(TP|IPO):", res_tp["p"])
    print("95% CI:", res_tp["ci_lo"], res_tp["ci_hi"])
    print("Chi² p:", res_tp["chi2_p"], "Fisher:", res_tp["fisher"])

    # B) Minimal figures (overlay + stacked)
    plot_overlay_and_stacked(res_slow, res_tp, labels=('-','+'))

    # C) Logistic regression summary (slowdowns vs TP majority)
    tp_majority = (E.mean(axis=1) >= 0.5).astype(int)
    fit_slow = logit_or_r2(ipo, slow.astype(int))
    fit_tp   = logit_or_r2(ipo, tp_majority)

    print("\n=== Logistic regression (per +1 IPO step) ===")
    print("Slowdowns:  OR = {:.2f} [{:.2f}, {:.2f}], R2_McF = {:.4f}, p = {:.3g}".format(
        fit_slow["OR"], *fit_slow["OR_CI"], fit_slow["R2_mcf"], fit_slow["p"]))
    print("TP slows:   OR = {:.2f} [{:.2f}, {:.2f}], R2_McF = {:.4f}, p = {:.3g}".format(
        fit_tp["OR"], *fit_tp["OR_CI"], fit_tp["R2_mcf"], fit_tp["p"]))

    '''
    # ---------- ---------- ---------- ---------- ---------- ---------- 
    # TP ENRICHMENT
    # ---------- ---------- ---------- ---------- ---------- ---------- 

    # ---------- Minimal bootstrap helpers ----------
    def _coerce_to_NxS(E, N):
        E = np.asarray(E)
        if E.ndim == 1: E = E[:, None]
        if E.shape[0] != N and E.shape[1] == N: E = E.T
        assert E.shape[0] == N, f"Expected (N,S) or (S,N); got {E.shape}, N={N}"
        return E

    def boot_p_event_given_phase_slow(enso_labels, event_bool, phases=(-1,0,1), B=5000):
        enso = np.asarray(enso_labels).ravel().astype(int)
        evt  = np.asarray(event_bool).ravel().astype(bool)
        n = enso.size
        boot = np.empty((B, len(phases)), float)
        for b in range(B):
            idx = np.random.randint(0, n, size=n)
            for i, ph in enumerate(phases):
                m = (enso[idx] == ph)
                boot[b, i] = np.mean(evt[idx][m]) if np.any(m) else np.nan
        return boot

    def boot_p_event_given_phase_tp(enso_labels, tp_mask_seeds, phases=(-1,0,1), B=5000):
        enso = np.asarray(enso_labels).ravel().astype(int)
        E = _coerce_to_NxS(tp_mask_seeds, enso.size).astype(bool)
        n, S = E.shape
        boot = np.empty((B, len(phases)), float)
        for b in range(B):
            idx = np.random.randint(0, n, size=n)     # resample time
            sids = np.random.randint(0, S, size=S)    # resample seeds
            vals = np.full((S, len(phases)), np.nan)
            for j, sid in enumerate(sids):
                for i, ph in enumerate(phases):
                    m = (enso[idx] == ph)
                    vals[j, i] = np.mean(E[idx, sid][m]) if np.any(m) else np.nan
            boot[b] = np.nanmean(vals, axis=0)        # seed-averaged replicate
        return boot

    # ---------- Enrichment computation + per-phase tests ----------
    def enrichment_and_tests(boot_slow, boot_tp, phases=(-1,0,1)):
        # normalize each bootstrap draw by its own mean ("climatology")
        rs = boot_slow / np.nanmean(boot_slow, axis=1, keepdims=True)
        rt = boot_tp   / np.nanmean(boot_tp,   axis=1, keepdims=True)

        # point estimates (use bootstrap means of normalized values)
        rS = np.nanmean(rs, axis=0)
        rT = np.nanmean(rt, axis=0)
        # CIs
        loS, hiS = np.nanpercentile(rs, [2.5, 97.5], axis=0)
        loT, hiT = np.nanpercentile(rt, [2.5, 97.5], axis=0)

        # per-phase difference tests: rT - rS
        P = len(phases)
        diffs = rt - rs
        pvals = np.empty(P)
        d_lo  = np.empty(P); d_hi = np.empty(P); d_bar = np.empty(P)
        for i in range(P):
            d = diffs[:, i]
            d_bar[i] = np.nanmean(d)
            d_lo[i], d_hi[i] = np.nanpercentile(d, [2.5, 97.5])
            # two-sided bootstrap p: fraction on opposite side of 0
            pvals[i] = 2 * min(np.mean(d <= 0), np.mean(d >= 0))

        return {
            "r_slow": rS, "ci_slow": (loS, hiS),
            "r_tp":   rT, "ci_tp":  (loT, hiT),
            "diff_mean": d_bar, "diff_ci": (d_lo, d_hi), "pvals": pvals
        }

    # ---------- Plot ----------
    def plot_enrichment(res, labels=("La Niña","Neutral","El Niño")):
        x = np.arange(len(labels))
        rS, (loS, hiS) = res["r_slow"], res["ci_slow"]
        rT, (loT, hiT) = res["r_tp"],   res["ci_tp"]
        yerrS = np.vstack([rS - loS, hiS - rS])
        yerrT = np.vstack([rT - loT, hiT - rT])

        w = 0.40
        fig, ax = plt.subplots(figsize=(7.2,4.2))
        ax.bar(x - w/2, rS, yerr=yerrS, width=w, capsize=6, color="lightgray", alpha=0.95, label="Slowdowns (data)")
        ax.bar(x + w/2, rT, yerr=yerrT, width=w, capsize=6, color="tab:blue",  alpha=0.90, label="TP slowdowns (CNN)")
        ax.axhline(1.0, ls="--", color="k", alpha=0.45)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("Conditional probability / climatology")
        ax.set_title("ENSO dependence: slowdowns vs CNN true-positives (normalized)")

        # annotate per-phase significance of (TP - Slow) enrichment
        for i in range(len(labels)):
            y_top = max(hiS[i], hiT[i]) * 1.10
            d  = res["diff_mean"][i]
            dlo, dhi = res["diff_ci"][0][i], res["diff_ci"][1][i]
            p  = res["pvals"][i]
            txt = f"Δ={d:.2f} [{dlo:.2f},{dhi:.2f}], p={p:.3f}"
            ax.text(x[i], y_top, txt, ha="center", va="bottom", fontsize=9)

        ax.legend(frameon=False, loc="upper left")
        ax.set_ylim(0, max((rS + yerrS[1]).max(), (rT + yerrT[1]).max()) * 1.25)
        plt.tight_layout(); plt.show()


    # ============================
    # ======= RUN IT =============
    # ============================
    phases = (-1,0,1)
    labels = ("La Niña","Neutral","El Niño")

    enso   = np.asarray(nino_i_labels).ravel()
    slow   = (np.asarray(y_truei).ravel() == 1)
    boot_slow = boot_p_event_given_phase_slow(enso, slow, phases, B=5000)

    boot_tp = boot_p_event_given_phase_tp(enso, tp_mask_seeds, phases, B=5000)

    res_enrich = enrichment_and_tests(boot_slow, boot_tp, phases)
    plot_enrichment(res_enrich, labels=labels)

    # (Optional) also print a concise table of per-phase significance:
    print("\nPer-phase difference in enrichment (TP − Slow):")
    for lab, d, lo, hi, p in zip(
        labels, res_enrich["diff_mean"],
        res_enrich["diff_ci"][0], res_enrich["diff_ci"][1],
        res_enrich["pvals"]
    ):
        print(f"{lab:>8}: Δ={d:.3f}  [{lo:.3f},{hi:.3f}],  p={p:.3f}")

    # ---------- ---------- ---------- ---------- ---------- ---------- 

    '''

    # ---------- ---------- ---------- ---------- ---------- ---------- 
    # JUST TPs
    # ---------- ---------- ---------- ---------- ---------- ---------- 

    def _coerce_to_NxS(E, N):
        E = np.asarray(E)
        if E.ndim == 1: 
            E = E[:, None]
        if E.shape[0] != N and E.shape[1] == N:
            E = E.T
        assert E.shape[0] == N, f"Expected (N,S) or (S,N); got {E.shape}, N={N}"
        return E

    def boot_tp_conditional_probs(enso_labels, tp_mask_seeds, phases=(-1,1), B=5000):
        """Bootstrap P(TP | ENSO) across seeds and time."""
        enso = np.asarray(enso_labels).ravel().astype(int)
        E = _coerce_to_NxS(tp_mask_seeds, enso.size).astype(bool)
        n, S = E.shape

        boot = np.empty((B, len(phases)), float)
        for b in range(B):
            idx  = np.random.randint(0, n, size=n)
            sids = np.random.randint(0, S, size=S)
            vals = np.full((S, len(phases)), np.nan)
            for j, sid in enumerate(sids):
                for i, ph in enumerate(phases):
                    m = (enso[idx] == ph)
                    vals[j, i] = np.mean(E[idx, sid][m]) if np.any(m) else np.nan
            boot[b] = np.nanmean(vals, axis=0)
        return boot


    def plot_tp_conditional(boot_tp, phases=(-1,1), labels=('-','+')):
        """Plot and print P(TP | ENSO) with 95% bootstrap CI."""
        # Mean and CI
        p_tp  = np.nanmean(boot_tp, axis=0)
        ci_lo = np.nanpercentile(boot_tp,  2.5, axis=0)
        ci_hi = np.nanpercentile(boot_tp, 97.5, axis=0)
        yerr  = np.vstack([p_tp - ci_lo, ci_hi - p_tp])

        # ---- Print results ----
        print("\nConditional probability of CNN true-positive slowdowns (P(TP|IPO)):")
        for lbl, mean, lo, hi in zip(labels, p_tp, ci_lo, ci_hi):
            print(f"{lbl:>8}: {mean:.3f}  (95% CI: [{lo:.3f}, {hi:.3f}])")

        # ---- Plot ----
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(6.8,4.0))
        ax.bar(x, p_tp, yerr=yerr, capsize=6, color="tab:blue", alpha=0.9)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("P(TP slowdown | IPO phase)")
        ax.set_title("Conditional probability of CNN true-positive slowdowns")
        ax.set_ylim(0, np.nanmax(ci_hi)*1.25)
        plt.tight_layout()
        plt.show()


    # =======================
    # =======  RUN  =========
    # =======================
    phases = (-1,1)
    labels = ("-","+")

    ipo = np.asarray(ipo_i_labels).ravel()
    boot_tp = boot_tp_conditional_probs(ipo, tp_mask_seeds, phases, B=5000)
    plot_tp_conditional(boot_tp, phases, labels)
    
    
    # -------------------------------------------
    # VARIANCE EXPLAINED
    # -------------------------------------------
    def variance_explained_by_phase(phase_labels, event_bool, phases=(-1,1), B=5000):
        """
        Returns % variance explained by IPO (between-phase / total) for a binary event.
        Also returns a 95% CI via time bootstrap.
        """
        enso = np.asarray(phase_labels).ravel().astype(int)
        y    = np.asarray(event_bool).ravel().astype(int)
        assert enso.size == y.size
        n = y.size

        # overall variance of Y
        ybar = y.mean()
        varY = ybar * (1 - ybar)
        if varY == 0:
            return {"VE": np.nan, "ci": (np.nan, np.nan), "components": None}

        # within-phase stats
        w = []; p = []
        for ph in phases:
            m = (enso == ph)
            if np.any(m):
                w.append(m.mean())
                p.append(y[m].mean())
            else:
                w.append(0.0)
                p.append(np.nan)
        w = np.array(w, float)
        p = np.array(p, float)
        # handle any empty phases gracefully
        mask = (w > 0) & np.isfinite(p)
        w = w[mask]; p = p[mask]

        # between-phase variance of conditional means
        var_between = np.sum(w * (p - ybar)**2)
        VE = var_between / varY

        # bootstrap CI (time resampling)
        boot = np.empty(B)
        for b in range(B):
            idx = np.random.randint(0, n, size=n)
            yy  = y[idx]; ee = enso[idx]
            yb  = yy.mean()
            vY  = yb * (1 - yb)
            if vY == 0:
                boot[b] = np.nan
                continue
            w_b = []; p_b = []
            for ph in phases:
                m = (ee == ph)
                if np.any(m):
                    w_b.append(m.mean()); p_b.append(yy[m].mean())
            w_b = np.array(w_b, float); p_b = np.array(p_b, float)
            vb  = np.sum(w_b * (p_b - yb)**2) / vY
            boot[b] = vb
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])

        return {"VE": float(VE), "ci": (float(lo), float(hi)),
                "components": {"ybar": float(ybar), "w": w, "p": p}}

    # ==== Apply to your data ====
    # ENSO labels and outcomes you already have:
    ipo  = np.asarray(ipo_i_labels).ravel()
    slow  = (np.asarray(y_truei).ravel() == 1).astype(int)

    # (A) Slowdowns: % variance explained by ENSO
    res_VE_slow = variance_explained_by_phase(ipo, slow, phases=(-1,1), B=5000)
    print("Slowdowns: VE_IPO = {:.3%}  (95% CI: [{:.3%}, {:.3%}])".format(
        res_VE_slow["VE"], res_VE_slow["ci"][0], res_VE_slow["ci"][1]))

    # (B) True positives across seeds:
    # collapse seeds to a single binary TP per time (e.g., majority vote)
    E = np.asarray(tp_mask_seeds)
    if E.ndim == 1: E = E[:, None]
    if E.shape[0] != ipo.size and E.shape[1] == ipo.size: E = E.T
    tp_majority = (E.mean(axis=1) >= 0.5).astype(int)

    res_VE_tp = variance_explained_by_phase(ipo, tp_majority, phases=(-1,1), B=5000)
    print("TP slowdowns: VE_IPO = {:.3%}  (95% CI: [{:.3%}, {:.3%}])".format(
        res_VE_tp["VE"], res_VE_tp["ci"][0], res_VE_tp["ci"][1]))

