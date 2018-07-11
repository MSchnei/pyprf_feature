# -*- coding: utf-8 -*-

"""Script to plot axis-of-motion tuning curves and histogram.

This script loads the best beta parameter estimates from training (refit on
entire training data) and test data.

1) It then groups voxels according to their preferred axis of motion in the 
training data and plots, per group, (normalized) beta parameters from test data

2) Additionally, a histogram of the preferred axis of motion in training
(or test) data can be plotted.

3) Voxels can be grouped by their preferred axis of motion in the training
data. Four different histograms can then be plotted for the distribution of
polar angles. 

""" 

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
import matplotlib.pyplot as plt
from scipy.stats import zscore

import pRF_config as cfg

# %% Load best models and data

# load the mask
niiMask = nb.load(cfg.strPathNiiMask)
aryMask = niiMask.get_data().astype('bool')

# get best models
aryRes = np.load(cfg.strPathOut + '_aryPrfRes.npy')
# mask the results array
aryRes = aryRes[aryMask, :]

# get beta weights for best models
aryBstTrainBetas = np.load(cfg.strPathOut + '_aryBstTrainBetas.npy',)
aryBstTstBetas = np.load(cfg.strPathOut + '_aryBstTestBetas.npy',)

# load manual MT mask
pathManMtMsk = '/media/sf_D_DRIVE/MotionLocaliser/Analysis/P02/Struct/P02_ManMTMsk.nii'
niiManMtMsk = nb.load(pathManMtMsk)
aryManMtMsk = niiManMtMsk.get_data().astype('bool')
lgcManMtMsk = aryManMtMsk[aryMask]

# %% mask, threshold and find preferred motion axis 
varThreshR2 = 0.15
# use R2 values as a mask
vecR2 = aryRes[:, 3]
lgcR2 = [vecR2 >= varThreshR2][0]

# combine R2 and ManMtMask logical
lgc = np.logical_and(lgcR2, lgcManMtMsk)

# exclude voxels with low R2, exclude last column (since it is weight 4 static)
aryBetasTrn = aryBstTrainBetas[lgc, :-1]
aryBetasTst = aryBstTstBetas[lgc, :-1]

# get preferred motion direction for test and train
vecPrfMtnTrn = np.argmax(aryBetasTrn, axis=1)
vecPrfMtnTst = np.argmax(aryBetasTst, axis=1)

#aryBetasTrn = aryBetasTrn - np.mean(aryBetasTrn, axis=0)[None, :]
#aryBetasTst = aryBetasTst - np.mean(aryBetasTst, axis=0)[None, :]

aryBetasTrnCor = zscore(aryBetasTrn, axis=0, ddof=1)
aryBetasTstCor = zscore(aryBetasTst, axis=0, ddof=1)

# get preferred motion direction for test and train
vecPrfMtnTrnCor = np.argmax(aryBetasTrnCor, axis=1)
vecPrfMtnTstCor = np.argmax(aryBetasTstCor, axis=1)

# %% plt tuning curve (i.e. beta regression coefficients), sort values
# depending on favourite axis of motion / motion direction in training

# print number of voxels and preferred axis of motion
for ind in range(aryBetasTrnCor.shape[1]):
    print('Nr of voxels in AoM ' + str(ind))
    print len(aryBetasTstCor[vecPrfMtnTrnCor == ind])
    print('Prefrred AoM in AoM ' + str(ind))
    print np.argmax(np.mean(aryBetasTstCor[vecPrfMtnTrnCor == ind], axis=0))

# plot axis of motion tuning curves
t = np.arange(aryBetasTrnCor.shape[1])
fig, axs = plt.subplots(len(np.unique(vecPrfMtnTrnCor)), 1, figsize=(10, 5))
axs = axs.ravel()
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
ylimits = [[-0.20, 0.4], [-0.40, 0.20], [0.0, 0.6], [-0.6, 0.0]]
for ind, mtnDir in enumerate(np.unique(vecPrfMtnTrnCor)):
    mean = np.mean(aryBetasTstCor[vecPrfMtnTrnCor == ind], axis=0)
    number = aryBetasTstCor[vecPrfMtnTrnCor == ind].shape[0]
    errorMinus = np.divide(np.std(aryBetasTstCor[vecPrfMtnTrnCor == ind],
                                  axis=0),
                           np.sqrt(number))
    errorPlus = np.divide(np.std(aryBetasTstCor[vecPrfMtnTrnCor == ind],
                                 axis=0),
                          np.sqrt(number))
    axs[ind].errorbar(t, mean, yerr=[errorMinus, errorPlus], color=colors[ind],
                      linewidth=4)
    axs[ind].set_xlim([-0.2, 3.2])
    axs[ind].set_ylim(ylimits[ind])
    axs[ind].get_xaxis().set_ticks([])

name = "/media/sf_D_DRIVE/MotionLocaliser/Presentation/OHBM2017/Poster/" + \
    "FiguresSource/MtnTuning/AoMCurves_p015"
plt.savefig(name + ".svg")


# %% plot histogram of the relative motion axes that wins
plt.figure()
# training
n, bins, patches = plt.hist(vecPrfMtnTrn, facecolor='green', alpha=0.75)
# test
# n, bins, patches = plt.hist(vecPrfMtnTst, facecolor='green', alpha=0.75)

# change histogram colours
bin_centers = 0.5 * (bins[:-1] + bins[1:])
col = bin_centers - min(bin_centers)
col /= max(col)
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
for ind, ind2 in enumerate(np.arange(0, 10, 3)):
    plt.setp(patches[ind2], 'facecolor', colors[ind])

name = "/media/sf_D_DRIVE/MotionLocaliser/Presentation/OHBM2017/Poster/" + \
    "FiguresSource/MtnTuning/HistogramMtnTuningCrv_p015"
plt.savefig(name + ".svg")


# %% plot histogram of correlations between polar angle and prf mtn direction
# polar angle
vecPolAngle = aryRes[:, 4]
vecPolAngle = vecPolAngle[lgc]

fig, axs = plt.subplots(len(np.unique(vecPrfMtnTrn)), 1, figsize=(14, 5))
axs = axs.ravel()
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

for ind, mtnDir in enumerate(np.unique(vecPrfMtnTrn)):
    lgcInd = [vecPrfMtnTrn == mtnDir][0]
    axs[ind].hist(vecPolAngle[lgcInd], 40, facecolor=colors[ind], alpha=0.75,)
