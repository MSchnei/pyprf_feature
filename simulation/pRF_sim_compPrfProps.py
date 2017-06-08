# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 12:22:55 2017

@author: marian
"""
from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt


# %% Set paths and parameters
# *** Set paths to data
strParPathGrdTruth = '/media/sf_D_DRIVE/MotionLocaliser/Simulation2p0/Apertures/pRF_model_tc'
strGrdTrthPickle = 'dicNrlParams_xval.pickle'

strParPathFitRes = '/media/sf_D_DRIVE/MotionLocaliser/Simulation2p0/FitResults'
lstFitRes = [
'simResp_xval_0_aryPrfRes.npy',
'simResp_xval_1_aryPrfRes.npy',
'simResp_xval_2_aryPrfRes.npy',
]

aryFitCNR = np.array([0.1, 0.5, 1])

lstAprtType = [
               'mskBar',
               'mskSquare',
               'mskCircle',
               'mskCircleBar',
               ]

varCriterion = 2  # 0: x pos, 1: y pos, 2: pRF size, 3: tuning curve model
strTxtSave = [['X position in deg of visual angle', '_byXPos_'],
              ['Y position in deg of visual angle', '_byYPos_'],
              ['pRF size (sigma) in deg of visual angle', '_byPrfSize_'],
              ['Axis of Motion', '_byAoM_'],
              ['Sigma of Motion Preference', '_byMotionSpread_']]

# set path to save figures
varPathFigs = '/media/sf_D_DRIVE/MotionLocaliser/Simulation2p0/FitResults/Figures/Comp_4times3/'

# *** derive variables
strTxtSave = strTxtSave[varCriterion]
varNumAprtTypes = len(lstAprtType)
varNumNoiseLvls = len(lstFitRes)

# %% Load ...
# load ground truth parameters in degree
lstGrdTruth = []

for ind1, idxType in enumerate(lstAprtType):  # walk through aperture types
    # load known ground truth
    varPath = os.path.join(strParPathGrdTruth, idxType, strGrdTrthPickle)
    with open(varPath, 'rb') as handle:
        aryPckl = pickle.load(handle)
    lstGrdTruth.append(aryPckl["aryNrlParamsDeg"])
    # this step is repeated in vein, but paramters are same across aprt types
    varNumXY = aryPckl["varNumXY"]
    varNumPrfSizes = aryPckl["varNumPrfSizes"]
    varNumTngCrv = aryPckl["varNumTngCrv"]
    varNumMdls = varNumXY*varNumPrfSizes*varNumTngCrv

aryGrdTruth = np.array(lstGrdTruth)
aryGrdTruth = np.transpose(aryGrdTruth, (1, 2, 0))

# load parameters results found during fitting
aryFitRes = np.empty((varNumMdls, 3, varNumAprtTypes, varNumNoiseLvls))
for ind1, idxType in enumerate(lstAprtType):  # walk through aperture types
    # load estimated data (exclude the last three parameters)
    for idxNsLvls in np.arange(varNumNoiseLvls):  # walk through noise lvls
        aryFitRes[:, :, ind1, idxNsLvls] = np.load(
            os.path.join(strParPathFitRes, idxType, lstFitRes[idxNsLvls])
            )[:, :3]

# %%
# *** calculate similarity S between model parameters of two pRFs a and b
# get labels
if varCriterion in [0, 1, 2]:
    vecCiterion = np.unique(aryGrdTruth[:, varCriterion, 0])
elif varCriterion == 3:  # take out last (equal prob) option
    vecCiterion = np.unique(aryGrdTruth[:, 3, 0])[:-1]
elif varCriterion == 4:
    vecCiterion = np.unique(aryGrdTruth[:, 3, 0])[:-1]
    lstTemp = []
    for ind in np.arange(varNumMtnSprd):
        lstTemp.append(vecCiterion[ind:][::varNumMtnSprd])
    vecCiterion = lstTemp

varNumCrit = len(vecCiterion)
aryS = np.empty((varNumAprtTypes, varNumCrit, varNumNoiseLvls, 2))

for ind1, idxType in enumerate(lstAprtType):  # walk through aperture types

    for ind2, idxCrit in enumerate(vecCiterion):

        # exclude data equal prob tng crv
        lgc1 = [aryGrdTruth[:, 3, ind1] < 12][0]

        # decide what to sort the data by
        if varCriterion in [0, 1, 2, 3]:
            # sort by x pos, y pos, pRF size or Tuning curve
            lgc2 = [aryGrdTruth[:, varCriterion, ind1] == idxCrit][0]
        elif varCriterion in [4, 5]:
            # sort by x pos, y pos or pRF size
            lgc2 = np.in1d(aryGrdTruth[:, 3, ind1], idxCrit)

        # combine logicals for data citerion selection
        lgc = np.logical_and(lgc1, lgc2)

        for idxNsLvls in np.arange(varNumNoiseLvls):

            # define x-range, y-range and sigma-range
            rangeX = np.subtract(np.max(aryGrdTruth[:, 0, ind1]),
                                 np.min(aryGrdTruth[:, 0, ind1]))
            rangeY = np.subtract(np.max(aryGrdTruth[:, 1, ind1]),
                                 np.min(aryGrdTruth[:, 1, ind1]))
            rangeSD = np.subtract(np.max(aryGrdTruth[:, 2, ind1]),
                                  np.min(aryGrdTruth[:, 2, ind1]))
            aryRange = np.array((rangeX, rangeY, rangeSD))

            # similarity index, will be 1 for perfect fit,
            # will be 0 for maximal difference
            vecS = np.subtract(
                np.ones(aryGrdTruth[lgc, :, ind1].shape[0]), (np.divide(np.sqrt(
                np.sum(np.power(np.divide(
                (aryGrdTruth[lgc,:3, ind1] - aryFitRes[lgc, :, ind1, idxNsLvls]),
                aryRange[np.newaxis, :]),2), axis =1)), np.sqrt(3))))
            # assign numbers to array
            aryS[ind1, ind2, idxNsLvls, 0] = np.mean(vecS)
            aryS[ind1, ind2, idxNsLvls, 1] = np.std(vecS)/np.sqrt(len(vecS))

# %% plot

fontsize = 18
# *** plot the data
for idxNsLvls in np.arange(varNumNoiseLvls):
    plotData = aryS[:, :, idxNsLvls, 0].T
    yerr = aryS[:, :, idxNsLvls, 1].T
    if varCriterion in [0, 1, 2, 3]:
        x = vecCiterion
    elif varCriterion == 4:
        x = np.arange(varNumAoM)
    elif varCriterion == 5:
        x = np.arange(varNumMtnSprd)
    plt.figure(idxNsLvls)

    fig, ax = plt.subplots()

    for idxType in np.arange(varNumAprtTypes):
        ax.errorbar(x, plotData[:, idxType], yerr[:, idxType],
                    label=lstAprtType[idxType], linewidth=3,)
    # add title
    title = 'AoM'
    plt.title(title + ', CNR: ' + str(aryFitCNR[idxNsLvls]), fontsize=fontsize)
    plt.xlabel(strTxtSave[0], fontsize = fontsize)
    plt.xlim((np.min(x), np.max(x)))
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    if varCriterion == 4:
        labels = ['0/180', '45/225', '90/270', '135/315']
        plt.xticks(x, labels)
    elif varCriterion == 5:
        labels = ['20/45', '45/45', '70/45']
        plt.xticks(x, labels)
    plt.ylabel('Similarity Index S', fontsize=fontsize)
    plt.ylim((0.0, 1.0))
    # Now add the legend with some customizations.
    legend = ax.legend(loc='lower left', shadow=True, fontsize = fontsize)

    # save
    filename = varPathFigs + 'PrfProps_' + title + strTxtSave[1] + \
        'NoiseLevel_' + str(idxNsLvls) + '.png'
    plt.savefig(filename)
