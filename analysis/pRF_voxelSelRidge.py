#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 15:35:07 2017

@author: marian
"""

import os
import numpy as np
import nibabel as nb
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LinearRegression
from sklearn.model_selection import KFold
from scipy import stats


# %% set paths

# Path to mask
strPathNiiMask = '/media/sf_D_DRIVE/MotionLocaliser/Analysis/P02/Struct/testMask.nii'
# Parent path to functional data
strPathNiiFunc = '/media/sf_D_DRIVE/MotionLocaliser/Analysis/P02/05_SpatSmoothDemean'
# list of nii files in parent directory (all nii files together need to have
# same number of volumes as there are PNGs):
lstNiiFls = ['demean_rafunc01_hpf.nii',
             'demean_rafunc02_hpf.nii',
             'demean_rafunc03_hpf.nii',
             'demean_rafunc04_hpf.nii',
             'demean_rafunc05_hpf.nii',
             'demean_rafunc06_hpf.nii',
             'demean_rafunc07_hpf.nii',
             ]
# length of the runs that were done
vecRunLngth = [172] * len(lstNiiFls)
# which run should be hold out for testing? [python index strating from 0]
varTestRun = 6
# string to models
strPathOut = '/media/sf_D_DRIVE/MotionLocaliser/Analysis/P02/FitResults/6runs_MotionAoMNoXval/MotionNoXvalAoM'

strPathRidgeMdl = strPathOut+'_ExtractedFeatures.npy'


# %% Load existing pRF time course models

print('------Load pRF time course models')

# Load the file:
aryPrfTc = np.load(strPathRidgeMdl)
aryPrfTc = aryPrfTc.reshape((-1, aryPrfTc.shape[-1]))

# Consider only the training runs
lgcPrfTc = np.array(np.split(np.arange(np.sum(vecRunLngth)),
                             np.cumsum(vecRunLngth)[:-1]))
lgcPrfTc = np.hstack(lgcPrfTc[np.arange(len(lstNiiFls)) != varTestRun])
aryPrfTc = aryPrfTc[..., lgcPrfTc]

# %% Load voxel responses
print('---------Loading nii data')
# Load mask (to restrict model finding):
aryMask = nb.load(strPathNiiMask).get_data().astype('bool')

# prepare aryFunc for functional data
aryFunc = np.empty((np.sum(aryMask), 0), dtype='float32')
for idx in np.arange(len(lstNiiFls))[np.arange(len(lstNiiFls)) != varTestRun]:
    print('------------Loading run: ' + str(idx+1))
    # Load 4D nii data:
    niiFunc = nb.load(os.path.join(strPathNiiFunc,
                                   lstNiiFls[idx])).get_data()
    # Load the data into memory:
    aryFunc = np.append(aryFunc, niiFunc[aryMask, :], axis=1)

# remove unneccary array
del(niiFunc)

# %% perform ridge regression

# prepare
aryPrfTc = aryPrfTc.T
aryFunc = aryFunc.T
# zscore the predictors
aryPrfTc = stats.zscore(aryPrfTc, axis=0, ddof=1)
# center the data
aryFunc = np.subtract(aryFunc, np.mean(aryFunc, axis=0)[None, :])

# 20 possible regularization coefficients (log spaced between 10 and 1,000)
alpha = np.logspace(1, 3, 20)

#%% cross validate
kf = KFold(n_splits=6)
kf.get_n_splits(aryPrfTc)

print(kf)  

for train_index, test_index in kf.split(aryPrfTc):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = aryPrfTc[train_index], aryPrfTc[test_index]
   y_train, y_test = aryFunc[train_index], aryFunc[test_index]


ridgeTest = RidgeCV(alphas=(0.1, 1.0, 10.0),
                    fit_intercept=False,
                    normalize=False,
                    scoring=None,
                    cv=kf.split(aryPrfTc),
                    gcv_mode=None,
                    store_cv_values=False)
ridgeTest.fit(aryPrfTc, aryFunc)


ridgeTest.alpha_                                      



# %%
from sklearn.model_selection import cross_val_score
ridgeTest = Ridge(alpha=[300])
scores = cross_val_score(ridgeTest, aryPrfTc, aryFunc, cv=6)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



linearTest = LinearRegression()
ridgeTest = Ridge(alpha=[0.1, 1.0, 10.0])
lassoTest = Lasso(alpha=[0.1, 1.0, 10.0])

ridgeTest.fit(aryPrfTc, aryFunc)
lassoTest.fit(aryPrfTc, aryFunc)

%timeit ridgeTest.fit(aryPrfTc, aryFunc)

%timeit lassoTest.fit(aryPrfTc, aryFunc)

%timeit linearTest.fit(aryPrfTc, aryFunc)
%timeit np.linalg.lstsq(aryPrfTc, aryFunc)