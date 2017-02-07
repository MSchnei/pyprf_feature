# -*- coding: utf-8 -*-
"""
This function needs to be fixed to account for either AoM or DoM flexibly.
Currently this will run with AoM, not DoM.
Compare also with pRF_compBetas_optCrvFit.py, which works with DoM
The problem is the CircDiff function
"""
from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import itertools

# %%
# *** Set parameters and paths to data
pathFolder = '/media/sf_D_DRIVE/MotionLocaliser/Simulation2p0/Apertures/pRF_model_tc/'
pathFitResults = '/media/sf_D_DRIVE/MotionLocaliser/Simulation2p0/FitResults'
path2saveFigs = '' 

criterion = 2  # 0: x pos, 1: y pos, 2: pRF size, 3: AoM1, 4: AoM2
text4save = [['X position in deg of visual angle', '_byXPos_'],
             ['Y position in deg of visual angle', '_byYPos_'],
             ['pRF size (sigma) in deg of visual angle', '_byPrfSize_'],
             ['Axis Of Motion', '_byAoM1_'],
             ['Axis Of Motion', '_byAoM2_'],
             ]

pathType = ['mskBar',
            'mskCircleBar',
            ]

varNumTypes = len(pathType)

lstFit = ['simResp_xval_1_aryPrfRes.npy',
          ]
lstSimResp = ['simResp_xval_1.npy',
              ]
lstEstim = ['simResp_xval_1_aryBstBetas.npy',
            ]

pathGrdTrth = 'dicNrlParams_xval.pickle'
varNumNoiseLvls = len(lstFit)
CNR = np.array([ 0.5])

# %%
# *** Define some useful functions (for curve fitting)

def funcGauss1D(x, mu, sig):
    """ Create 1D Gaussian. Source:
    http://mathworld.wolfram.com/GaussianFunction.html
    """
    arrOut = np.exp(-np.power((x - mu)/sig, 2.)/2)
    # normalize
    # arrOut = arrOut/(np.sqrt(2.*np.pi)*sig)
    # normalize (laternative)
    arrOut = arrOut/np.sum(arrOut)
    return arrOut


def makeMotDirTuneCurves(x, mu, sig):
    # find middle of x
    centre = np.round(len(x)/2)
    dif = np.subtract(centre, mu).astype(int)
    # initialise the curve in the centre so it will be symmetric
    arrOut = funcGauss1D(x, mu+dif, sig)
    # roll back so that the peak of the gaussian will be over mu
    arrOut = np.roll(arrOut, -dif)
    return arrOut


def makeAoMTuneCurves(x, mu1, mu2, sig):
    arr1 = makeMotDirTuneCurves(x, mu1, sig)
    arr2 = makeMotDirTuneCurves(x, mu2, sig)
    arrOut = 0.5 * arr1 + 0.5 * arr2
    return arrOut


def circDiff(length, ary1, ary2):
    """calculate the circular difference between two paired arrays.
    This function will return the difference between pairs of numbers; however
    the difference that is output will be minimal in the sense that if we
    assume an array with length = 4: [0, 1, 2, 3], the difference between
    0 and 3 will not be 3, but (i.e. circular difference)"""
    x = np.arange(length)
    mod = length % 2
    if mod == 0:
        temp = np.ones(length)
        temp[length/2:] = -1
    else:
        x = x - np.floor(length/2)
        temp = np.copy(x)
        temp[np.less(x, 0)] = 1
        temp[np.greater(x, 0)] = -1
    x = np.cumsum(temp)

    diagDiffmat = np.empty((length, length))
    for idx in np.arange(length):
        x = np.roll(x, 1)
        diagDiffmat[idx, :] = x
    # return diagDiffmat[ary1][ary2]
    flat = diagDiffmat.flatten()
    ind = ary1*diagDiffmat.shape[0] + ary2
    ind = ind.astype('int')
    return flat[ind]

# %%
# *** load ground truth paramters: x, y, sigma, tuning curve index

grdTrth = []
for idxType in pathType:  # walk through aperture types
    # load known ground truth
    # this step is repeated in vein, since paramters are same across aprt types
    with open(os.path.join(pathFolder, idxType, pathGrdTrth), 'rb') as handle:
        aryPckl = pickle.load(handle)
    grdTrth.append(aryPckl["aryNrlParamsDeg"])
    varNumXY = aryPckl["varNumXY"]
    varNumPrfSizes = aryPckl["varNumPrfSizes"]
    varNumTngCrv = aryPckl["varNumTngCrv"]
    varNumMdls = varNumXY*varNumPrfSizes*varNumTngCrv

# %%
# *** get motion tuning curve indices from ground truth
aryMtnCrvIdx = grdTrth[0][:, 3].astype('int')

# get logical to exclude equal probability (used below)
lgcEclEqProb = np.less(aryMtnCrvIdx, 12)

# make motion tuning curves
# 1) determine motion tuning curve parameters (preferred AoM, spread)
x = np.arange(1, 8+1)
aryAoMMus = np.arange(1, 4+1, dtype='int')
aryAoMMus = zip(aryAoMMus, aryAoMMus+4)
aryAoMSig = np.array([20/45, 45/45, 70/45])
iterables = [aryAoMMus, aryAoMSig]
aryAoM = list(itertools.product(*iterables))
for ind, item in enumerate(aryAoM):
    aryAoM[ind] = list(np.hstack(item))
aryAoM = np.array(aryAoM)
# 2) make motion tuning curves based on parameters
aryTngCrvAoM = np.empty((len(x), len(aryAoM)))
for ind, (mu1, mu2, sig) in enumerate(aryAoM):
    aryTngCrvAoM[:, ind] = makeAoMTuneCurves(x, mu1, mu2, sig)
# add one more array that has equal response to all motion positions
aryTngCrvEqProb = np.tile(np.array([1/8]), 8)
aryTngCrvEqProb = aryTngCrvEqProb.reshape((8, 1))
# add two arrays together to form one common array of all tuning curves
aryTngCrvAoM = np.concatenate((aryTngCrvAoM, aryTngCrvEqProb),
                              axis=1)
aryTngCrvAoM = aryTngCrvAoM.T

# get ground truth motion tuning curve by relating indices with curves
aryGrdTrthMtnCrv = aryTngCrvAoM[aryMtnCrvIdx, :]
# get the best motion direction out of all parameters
vecGrdTrthMtnDir = np.argmax(aryGrdTrthMtnCrv, axis=1)

# %%
# *** load estimated motion tuning preference (beta weights)
aryEstimMtnCrv = np.empty((varNumMdls, 5, varNumNoiseLvls, varNumTypes),
                          dtype='float32')
for ind1, idxType in enumerate(pathType):  # walk through aperture types
    for ind2, idxNoise in enumerate(lstEstim):  # walk through noise levels
        aryEstimMtnCrv[:, :, ind2, ind1] = np.load(os.path.join(pathFitResults,
                                                                idxType,
                                                                idxNoise))
# for now, ignore the beta weights for the constant predictor
aryEstimMtnCrv = aryEstimMtnCrv[:, 0:4, :, :]

# %%
# *** exclude equiprobability tuning curves
aryMtnCrvIdx = aryMtnCrvIdx[lgcEclEqProb]
aryGrdTrthMtnCrv = aryGrdTrthMtnCrv[lgcEclEqProb, :]
vecGrdTrthMtnDir = vecGrdTrthMtnDir[lgcEclEqProb]
# get ground truth motion parameters
aryGrdTrthMtnParams = aryAoM[aryMtnCrvIdx, :]
# subtract 1 from column 1 and 2 (to get rid of the constant that was added)
aryGrdTrthMtnParams[:, 0:2] -= 1
# get new estimation results
aryEstimMtnCrv = aryEstimMtnCrv[lgcEclEqProb, :, :, :]


# %%
# *** compare two estimated maxima with ground truth

# find initial values
max1 = np.argmax(aryEstimMtnCrv[:, 0:4, :, :], axis=1)

# set ground truth criteria for parameters x, y, sigma
prfCriteria = grdTrth[0][lgcEclEqProb, 0:3]
# set ground truth parameters for motion direction and spread
mtnParams = aryGrdTrthMtnParams[:, 0:1]
# subtract 4 for preprocessing
criteria = np.concatenate((prfCriteria, mtnParams), axis=1)

# get ranges to normalize the similarity index later
#rangeMax1 = np.max(mtnParams[:, 0]) - np.min(mtnParams[:, 0])
#rangeMax2 = np.max(mtnParams[:, 1]) - np.min(mtnParams[:, 1])
#aryRange = np.array((rangeMax1, rangeMax2))
aryRange = np.array([2])

vecCiterion = np.unique(criteria[:, criterion])
S = np.empty((varNumNoiseLvls,
              varNumTypes,
              len(vecCiterion),
              2
              ))
for idxNsLvls in np.arange(varNumNoiseLvls):
    for idxType in np.arange(varNumTypes):
        for idxCrit, Crit in enumerate(vecCiterion):

            lgc = [criteria[:, criterion] == Crit][0]

            vecS = np.subtract(
                np.ones(mtnParams[lgc].shape[0]), (np.divide(np.sqrt(
                np.squeeze(np.power(np.divide(
                circDiff(4, np.squeeze(mtnParams[lgc,:]), max1[lgc, idxNsLvls, idxType]),
                aryRange[np.newaxis, :]),2))), np.sqrt(2)))
                )
            S[idxNsLvls, idxType, idxCrit, 0] = np.mean(vecS)
            S[idxNsLvls, idxType, idxCrit, 1] = np.std(vecS)/np.sqrt(len(vecS))

# %%
# *** plot
fontsize = 18

for idxNsLvls in np.arange(varNumNoiseLvls):
    x = vecCiterion
    plotData = S[idxNsLvls, :, :, 0].T
    yerr = S[idxNsLvls, :, :, 1].T
    plt.figure(idxNsLvls)

    fig, ax = plt.subplots()

    for idxType in np.arange(varNumTypes):
        ax.errorbar(x,
                    plotData[:, idxType],
                    yerr[:, idxType],
                    label=pathType[idxType],
                    linewidth=3,
                    )
    # add title
    title = 'CNR: ' + str(CNR[idxNsLvls])
    plt.title(title, fontsize=fontsize)
    plt.xlabel(text4save[criterion][0], fontsize = fontsize)
    plt.xlim((np.min(x), np.max(x)))
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    if criterion in [3, 4]:
        labels = ['0/180', '45/225', '90/270', '135/315']
        plt.xticks(x, labels)
    elif criterion == 5:
        labels = ['20/45', '45/45', '70/45']
        plt.xticks(x, labels)
    plt.ylabel('Similarity Index S', fontsize=fontsize)
    plt.ylim((0.5, 1.0))
    # Now add the legend with some customizations.
    legend = ax.legend(loc='lower right', shadow=True, fontsize=fontsize)

