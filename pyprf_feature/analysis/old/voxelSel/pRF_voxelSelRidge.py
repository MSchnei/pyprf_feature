# -*- coding: utf-8 -*-

"""Procedure fit ridge regression to select voxels.""" 

# Part of py_pRF_motion library
# Copyright (C) 2016  Marian Schneider
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import nibabel as nb
import pRF_config as cfg
from pRF_utils import loadNiiData, saveNiiData, calcR2, calcFstats, calcMse
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
from scipy import stats

# %% set paths
cfg.strPathRidgeMdl = cfg.strPathOut+'_ExtractedFeatures.npy'

# %% Load existing pRF time course models

print('------Load pRF time course models')

# Load the file:
aryPrfTc = np.load(cfg.strPathRidgeMdl)
aryPrfTc = aryPrfTc.reshape((-1, aryPrfTc.shape[-1]))

# derive logical for training/test runs
lgcTrnTst = np.ones(np.sum(cfg.vecRunLngth), dtype=bool)
lgcTrnTst[np.cumsum(cfg.vecRunLngth)[cfg.varTestRun-1]:np.cumsum(
         cfg.vecRunLngth)[cfg.varTestRun]] = False

# split in training and test runs
aryPrfTcTrn = aryPrfTc[..., lgcTrnTst].T
aryPrfTcTst = aryPrfTc[..., ~lgcTrnTst].T

# %% Load voxel responses

# Load mask (to restrict model finding):
aryMask = nb.load(cfg.strPathNiiMask).get_data().astype('bool')
# Load data from functional runs
aryFunc = loadNiiData(cfg.lstNiiFls, strPathNiiMask=cfg.strPathNiiMask,
                      strPathNiiFunc=cfg.strPathNiiFunc)

# split in training and test runs
aryFuncTrn = aryFunc[..., lgcTrnTst].T
aryFuncTst = aryFunc[..., ~lgcTrnTst].T
# remove unneccary array
del(aryFunc)

# %% perform ridge regression

# zscore the predictors
aryPrfTcTrn = stats.zscore(aryPrfTcTrn, axis=0, ddof=2)
# center the data
aryFuncTrn = np.subtract(aryFuncTrn, np.mean(aryFuncTrn, axis=0)[None, :])

# 20 possible regularization coefficients (log spaced between 10 and 1,000)
vecAlpha = np.logspace(np.log10(10), np.log10(10000), 20)

# %% use RidgeCV with generalized crossvalidation (GCV), this returns one
# optimal alpha per voxel but GCV might be inappropriate for time series data

## using GCV
#ridgeGCV = RidgeCV(alphas=vecAlpha,
#                   fit_intercept=False,
#                   normalize=False,
#                   scoring=None,
#                   cv=None,
#                   gcv_mode='auto',
#                   store_cv_values=True)
#
#ridgeGCV.fit(aryPrfTcTrn, aryFuncTrn)
#
## extract best alpha per voxel
#aryAvgMse = np.mean(ridgeGCV.cv_values_, axis=0)
#idxBestAlpha = np.argmin(aryAvgMse, axis=1)
#bestAlpha = np.array([vecAlpha[ind] for ind in idxBestAlpha])
#
## refit with best alphas to entire training data
#ridgeRefit = Ridge(alpha=bestAlpha, fit_intercept=False, normalize=False)
#ridgeRefit.fit(aryPrfTcTrn, aryFuncTrn)
#
## calculate Fstats on training data
#vecFval, vecPval = calcFstats(ridgeRefit.predict(aryPrfTcTrn), aryFuncTrn,
#                              aryPrfTcTrn.shape[1])
#
## calculate R2 on training data set
#vecR2Trn = calcR2(ridgeRefit.predict(aryPrfTcTrn), aryFuncTrn)
#
## calculate R2 on test data set
#vecR2Tst = calcR2(ridgeRefit.predict(aryPrfTcTst), aryFuncTst)
#
## save R2 as nii
#strTmp = (cfg.strPathOut + '_GCV_vecR2Trn.nii')
#saveNiiData(vecR2Trn, nb.load(cfg.strPathNiiMask), strTmp, aryMask)
#strTmp = (cfg.strPathOut + '_GCV_vecPvalTrain.nii')
#saveNiiData(vecPval, nb.load(cfg.strPathNiiMask), strTmp, aryMask)

# %% perform kfold crossvalidation, in every fold calculate MSE, return best
# alpha for every voxel
print('------Calculate ridgre regression')

# using Kfold
kf = KFold(n_splits=len(cfg.lstNiiFls)-1)

aryAvgMse = np.empty((aryFuncTrn.shape[1], len(vecAlpha)), dtype='float32')
# walk through different alphas
for idxAlpha, alpha in enumerate(vecAlpha):
    # walk through differnet folds of crossvalidation
    aryMse = np.empty((aryFuncTrn.shape[1], kf.n_splits), dtype='float32')
    for idx, (idxTrn, idxVal) in enumerate(kf.split(aryPrfTcTrn)):
        objRidgeMdl = Ridge(alpha=alpha, fit_intercept=False,
                            normalize=False, solver='auto')

        objRidgeMdl.fit(aryPrfTcTrn[idxTrn], aryFuncTrn[idxTrn])
        # calculate MSE
        aryMse[:, idx] = calcMse(objRidgeMdl.predict(aryPrfTcTrn[idxVal]),
                                 aryFuncTrn[idxVal])
    # average the MSE across all folds for this alpha
    aryAvgMse[:, idxAlpha] = np.mean(aryMse, axis=1)

# extract best alpha per voxel
idxBestAlpha2 = np.argmin(aryAvgMse, axis=1)
bestAlpha = np.array([vecAlpha[ind] for ind in idxBestAlpha2])

# refit with best alphas to entire training data
ridgeRefit = Ridge(alpha=bestAlpha, fit_intercept=False, normalize=False)
ridgeRefit.fit(aryPrfTcTrn, aryFuncTrn)

# calculate Fstats on training data
vecFval, vecPval = calcFstats(ridgeRefit.predict(aryPrfTcTrn), aryFuncTrn,
                              aryPrfTcTrn.shape[1])

# calculate R2 on training data set
vecR2Trn = calcR2(ridgeRefit.predict(aryPrfTcTrn), aryFuncTrn)

# calculate R2 on test data set
vecR2Tst = calcR2(ridgeRefit.predict(aryPrfTcTst), aryFuncTst)

# save R2 as nii
strTmp = (cfg.strPathOut + '_Kfold_vecR2Trn.nii')
saveNiiData(vecR2Trn, nb.load(cfg.strPathNiiMask), strTmp, aryMask)
strTmp = (cfg.strPathOut + '_Kfold_vecPvalTrain.nii')
saveNiiData(vecPval, nb.load(cfg.strPathNiiMask), strTmp, aryMask)
strTmp = (cfg.strPathOut + '_Kfold_vecFvalTrain.nii')
saveNiiData(vecFval, nb.load(cfg.strPathNiiMask), strTmp, aryMask)
