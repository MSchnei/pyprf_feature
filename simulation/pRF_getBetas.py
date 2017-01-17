# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 17:19:35 2016

This script gets the beta parameter estimates of the full model, since
they were calculated only for partial models during the cross validation

It can be called with the bash script:
- pRF_getBetas_xval.sh
- it expects command line arguments: apertIdx [0-3], noiseIdx [0-2]

This script requires:
- pRF_getBetas_functions

@author: marian
"""
from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
import numpy as np
import os
import pickle
import itertools
from pRF_getBetas_functions import getBetas
import multiprocessing as mp
import sys
from pRF_functions import funcHrf, funcConvPar

# %%
# *** Define parameters

# get some parameters from command line
varNumCmdArgs = len(sys.argv) - 1
print 'Argument List:', str(sys.argv)

if varNumCmdArgs == 2:
    # determine the type of aperture
    apertIdx = int(sys.argv[1])
    # set the level of noise
    noiseIdx = int(sys.argv[2])
else:
    raise ValueError('Not enough command line args provided.' +
                     'Provide aperture type information [0, 1, 2, 3]' +
                     'and noise level index [0, 1, 2]')

# %%
# *** Set parameters and paths to data
varTr = 3.0
varPar = 10
varNumVols = 1232


pathFolder = '/media/sf_D_DRIVE/MotionLocaliser/Apertures/pRF_model_tc/'

pathType = ['mskBar',
            'mskSquare',
            'mskCircleBar',
            'mskCircle']
pathType = pathType[apertIdx]

lstFit = ['aryPrfRes_NoiseLev_xval0.npy',
          'aryPrfRes_NoiseLev_xval1.npy',
          'aryPrfRes_NoiseLev_xval2.npy',
          ]
lstFit = lstFit[noiseIdx]

lstSimResp = ['simResp_xval_0.npy',
              'simResp_xval_1.npy',
              'simResp_xval_2.npy',
              ]
lstSimResp = lstSimResp[noiseIdx]

pathGrdTrth = 'dicNrlParams_xval.pickle'

# %%
# *** Load...

# prepare arrays

# get some info about simModels: varNumXY, varNumPrfSizes, varNumSimMdls
with open(os.path.join(pathFolder, pathType, pathGrdTrth), 'rb') as handle:
    aryPckl = pickle.load(handle)
varNumXY = aryPckl["varNumXY"]
varNumPrfSizes = aryPckl["varNumPrfSizes"]
varNumTngCrv = aryPckl["varNumTngCrv"]
varNumSimMdls = varNumXY*varNumPrfSizes*varNumTngCrv

# load simulated responses [nrSimModels x time points]
simResp = np.load(os.path.join(pathFolder, pathType, lstSimResp))

# load predictor time courses [nrFitModels x time points x 8]
aryDsgn = np.load(os.path.join(pathFolder, pathType, 'aryPrfMdls_xval.npy'))

# get estimated paramters: x, y, sigma [nrSimModels x 3 params]
estimPrf = np.load(os.path.join(pathFolder, pathType, lstFit))[:, :3]

# %%
# *** convolve predictors time courses with HRF

# reshape array design and add constant
aryDsgn = np.transpose(aryDsgn, (0, 1, 3, 2))
aryDsgn = aryDsgn.reshape(varNumXY*varNumPrfSizes*8, varNumVols)

# convolve
vecHrf = funcHrf(varNumVols, varTr)
aryDsgnConv = funcConvPar(aryDsgn, vecHrf, varNumVols)

# put array back into its former shape
aryDsgnConv = aryDsgnConv.reshape(varNumXY*varNumPrfSizes, 8, varNumVols)
aryDsgnConv = np.transpose(aryDsgnConv, (0, 2, 1))

# add constant term
aryDsgnConv = np.concatenate((aryDsgnConv,
                              np.ones((aryDsgnConv.shape[0],
                                       aryDsgnConv.shape[1],
                                       1),
                                      dtype='float32')),
                             axis=2)

# %%
# *** calculate fit model params (x, y, sigma) that were used for pRF finding
# (this step currently requires some ugly code repetition, integrate in future)
varNumX = 25
# Number of y-positions to model:
varNumY = 25
# Number of pRF sizes to model:
varNumPrfSizes = 22

# Extend of visual space from centre of the screen (i.e. from the fixation
# point):
varExtXmin = -12.00
varExtXmax = 12.00
varExtYmin = -12.0
varExtYmax = 12.00

# Maximum and minimum pRF model size (standard deviation of 2D Gaussian):
varPrfStdMin = 1.0
varPrfStdMax = 22.0  # max is given by np.sqrt(2*np.power(2*varExtXmax,2))

# get logical
tplPngSize = (100, 100)
vecX = np.linspace(0,
                   (tplPngSize[0] - 1),
                   varNumX,
                   endpoint=True)
# Vector with the y-indicies of the positions in the visual
# space at which to create pRF models.
vecY = np.linspace(0,
                   (tplPngSize[1] - 1),
                   varNumY,
                   endpoint=True)

# exclude x and y combination that are outside the circle (since stimuli
# will be shown in circular aperture)
iterables = [vecX, vecY]
vecXY = list(itertools.product(*iterables))
vecXY = np.asarray(vecXY)
# pass only the combinations inside the circle aperture
temp = vecXY-(tplPngSize[0]/2)
temp = np.sqrt(np.power(temp[:, 0], 2) + np.power(temp[:, 1], 2))
# calculate logical so we can use it on degree (not pixel) version
lgcXYcombi = np.less(temp, tplPngSize[0]/2)

# Vector with the moddeled x-positions of the pRFs:
vecMdlXpos = np.linspace(varExtXmin,
                         varExtXmax,
                         varNumX,
                         endpoint=True)

# Vector with the moddeled y-positions of the pRFs:
vecMdlYpos = np.linspace(varExtYmin,
                         varExtYmax,
                         varNumY,
                         endpoint=True)

# Vector with the moddeled standard deviations of the pRFs:
vecMdlSd = np.linspace(varPrfStdMin,
                       varPrfStdMax,
                       varNumPrfSizes,
                       endpoint=True)

# exclude x and y combinations that are outside the circle mask
iterables = [vecMdlXpos, vecMdlYpos]
vecMdlXY = list(itertools.product(*iterables))
vecMdlXY = np.asarray(vecMdlXY)
# pass only the combinations inside the circle aperture
vecMdlXY = vecMdlXY[lgcXYcombi]

# get combinations of all model parameters (this time in deg of vis angle)
iterables = [vecMdlXY, vecMdlSd]
aryMdls = list(itertools.product(*iterables))
for ind, item in enumerate(aryMdls):
    aryMdls[ind] = list(np.hstack(item))

# results in mdl params that were fitted [varNumFitMdls x 3 parms (x, y sigma)]
aryFitMdlParams = np.array(aryMdls)

# %%
# *** get beta values for found pRF parameters (varNumSimMdls * 8)

# transform to float32
aryFitMdlParams = aryFitMdlParams.astype('int')
aryDsgnConv = aryDsgnConv.astype('float32')
estimPrf = estimPrf.astype('int')
simResp = simResp.astype('float32')

# split arrays fro parallel processing
lstEstimPrf = np.array_split(estimPrf, varPar)
lstSimResp = np.array_split(simResp, varPar)

# Empty list for results (parameters of best fitting pRF model):
lstBetasRes = [None] * varPar

# Empty list for processes:
lstPrcs = [None] * varPar

# Create a queue to put the results in:
queOut = mp.Queue()

print('---------Find betas')
for idxPrc in range(0, varPar):
    lstPrcs[idxPrc] = mp.Process(target=getBetas,
                                 args=(idxPrc,
                                       aryFitMdlParams,
                                       aryDsgnConv,
                                       lstEstimPrf[idxPrc],
                                       lstSimResp[idxPrc],
                                       queOut)
                                 )

    # Daemon (kills processes when exiting):
    lstPrcs[idxPrc].Daemon = True

# Start processes:
for idxPrc in range(0, varPar):
    lstPrcs[idxPrc].start()

# Collect results from queue:
for idxPrc in range(0, varPar):
    lstBetasRes[idxPrc] = queOut.get(True)

# Join processes:
for idxPrc in range(0, varPar):
    lstPrcs[idxPrc].join()

print('---------Save betas results to list')
the_filename = '/media/sf_D_DRIVE/MotionLocaliser/Apertures/pRF_model_tc/' + \
    pathType + '/lstBetasRes' + '_Noise' + str(noiseIdx) + '.pickle'
with open(the_filename, 'wb') as f:
    pickle.dump(lstBetasRes, f)

print('---------Save to numpy array')
lstRes = [None] * varPar
# Put output into correct order:
for idxRes in range(0, varPar):
    # Index of results (first item in output list):
    varTmpIdx = lstBetasRes[idxRes][0]
    # Put fitting results into list, in correct order:
    lstRes[varTmpIdx] = lstBetasRes[idxRes][1]

aryRes = np.empty((0, 9))
for idxRes in range(0, varPar):
    aryRes = np.concatenate((aryRes, lstRes[idxRes]), axis=0)

name = os.path.join(pathFolder, pathType,
                    'aryEstimMtnCrv' + '_Noise' + str(noiseIdx))
np.save(name, aryRes)
