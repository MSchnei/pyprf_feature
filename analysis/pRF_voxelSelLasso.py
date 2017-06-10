# -*- coding: utf-8 -*-

"""Procedure fit lasso regression to select voxels.""" 

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
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import KFold
from scipy import stats

# %% set paths
cfg.strPathLassoMdl = cfg.strPathOut+'_ExtractedFeatures.npy'

# %% Load existing pRF time course models

print('------Load pRF time course models')

# Load the file:
aryPrfTc = np.load(cfg.strPathLassoMdl)
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

# %% perform lasso regression

# zscore the predictors
aryPrfTcTrn = stats.zscore(aryPrfTcTrn, axis=0, ddof=2)
# center the data
aryFuncTrn = np.subtract(aryFuncTrn, np.mean(aryFuncTrn, axis=0)[None, :])

# 20 possible regularization coefficients (log spaced between 10 and 1,000)
vecAlpha = np.logspace(np.log10(0.1), np.log10(100), 20)

# %% perform kfold crossvalidation, in every fold calculate MSE, return best
# alpha for every voxel

# using Kfold
kf = KFold(n_splits=len(cfg.lstNiiFls)-1)

aryAvgMse = np.empty((aryFuncTrn.shape[1], len(vecAlpha)), dtype='float32')
# walk through different alphas
for idxAlpha, alpha in enumerate(vecAlpha):
    # walk through differnet folds of crossvalidation
    aryMse = np.empty((aryFuncTrn.shape[1], kf.n_splits), dtype='float32')
    for idx, (idxTrn, idxVal) in enumerate(kf.split(aryPrfTcTrn)):
        objLassoMdl = Lasso(alpha=alpha, fit_intercept=False,
                            normalize=False)

        objLassoMdl.fit(aryPrfTcTrn[idxTrn], aryFuncTrn[idxTrn])
        # calculate MSE
        aryMse[:, idx] = calcMse(objLassoMdl.predict(aryPrfTcTrn[idxVal]),
                                 aryFuncTrn[idxVal])
    # average the MSE across all folds for this alpha
    aryAvgMse[:, idxAlpha] = np.mean(aryMse, axis=1)

# extract best alpha per voxel
idxBestAlpha2 = np.argmin(aryAvgMse, axis=1)
bestAlpha = np.array([vecAlpha[ind] for ind in idxBestAlpha2])

# refit with best alphas to entire training data
lassoRefit = Lasso(alpha=bestAlpha, fit_intercept=False, normalize=False)
lassoRefit.fit(aryPrfTcTrn, aryFuncTrn)

# calculate Fstats on training data
vecFval, VecPval = calcFstats(lassoRefit.predict(aryPrfTcTrn), aryFuncTrn,
                              aryPrfTcTrn.shape[1])

# calculate R2 on training data set
vecR2Trn = calcR2(lassoRefit.predict(aryPrfTcTrn), aryFuncTrn)

# calculate R2 on test data set
vecR2Tst = calcR2(lassoRefit.predict(aryPrfTcTst), aryFuncTst)

# save R2 as nii
strTmp = (cfg.strPathOut + '_Kfold_vecR2Trn.nii')
saveNiiData(vecR2Trn, nb.load(cfg.strPathNiiMask), strTmp, aryMask)
strTmp = (cfg.strPathOut + '_Kfold_vecPvalTrain.nii')
saveNiiData(VecPval, nb.load(cfg.strPathNiiMask), strTmp, aryMask)
