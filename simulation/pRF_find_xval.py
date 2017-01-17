# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 16:52:56 2016

@author: marian
many elements of this script are taken from
py_28_pRF_finding from Ingo Marquardt
"""
# %%
# *** Import modules
from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
import os
import numpy as np
import scipy as sp
from scipy import stats
import nibabel as nb
import tables
import time
import pickle
import itertools
from shutil import copyfile
import multiprocessing as mp
from pRF_functions import (funcHrf, funcNrlTcMotPred, funcFindPrfMltpPrdXVal,
                           funcConvPar)
import sys

# %%
# *** Define parameters

# get some parameters from command line
varNumCmdArgs = len(sys.argv) - 1
print 'Argument List:', str(sys.argv)

if varNumCmdArgs == 3:
    # determine the type of aperture
    aprtType = str(sys.argv[1])
    # set the level of noise
    noiseIdx = int(sys.argv[2])
    # Create pRF time course models?
    lgcCrteMdl = bool(int(sys.argv[3]))
elif varNumCmdArgs == 2:
    # determine the type of aperture
    aprtType = str(sys.argv[1])
    # set the level of noise
    noiseIdx = int(sys.argv[2])
    # Create pRF time course models?
    lgcCrteMdl = True
else:
    raise ValueError('Not enough command line args provided.' +
                     'Provide aperture type information' +
                     '[mskBar, mskCircle, mskSquare] and noise level index')

# Number of x-positions to model:
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

# %%
# Number of png files (meaning aperture positions) to load:
varNumPNG = 33

# Number of presented runs
varNumRuns = 8

# Number of cross validations
varNumXval = int(varNumRuns/2)

# Volume TR of input data [s]:
varTr = 3.0

# Determine the factors by which the image should be downsampled (since the
# original array is rather big; factor 2 means:downsample from 1200 to 600):
factorX = 12
factorY = 12

# state the number of parallel processes
varPar = 10

# Output path for time course files:
strPathMdl = '/media/sf_D_DRIVE/MotionLocaliser/Apertures/pRF_model_tc/' + \
    aprtType + '/'

# %%
# provide parameters for pRF time course creation

# Base name of pickle files that contain order of stim presentat. in each run
# file should contain 1D array, column contains present. order of aperture pos,
# here file is 2D where 2nd column contains present. order of motion directions
strPathPresOrd = '/media/sf_D_DRIVE/MotionLocaliser/PsychoPyScripts/Conditions/Conditions_run0'

# Base name of png files representing the stimulus aperture as black and white
# the decisive information is in alpha channel (4th dimension)
strPathPNGofApertPos = '/media/sf_D_DRIVE/MotionLocaliser/Apertures/PNGs/' + \
    aprtType + '/' + aprtType + '_'

# Size of png files (pixel*pixel):
tplPngSize = (1200, 1200)


# %%
# ***  Check time
varTme01 = time.time()

# %%
# *** Load presentation order of conditions
print('------Load presentation order of apert pos and mot dir')
aryPresOrd = np.empty((0, 2))
for idx01 in range(0, varNumRuns):
    # reconstruct file name
    filename1 = (strPathPresOrd + str(idx01+1) + '.pickle')
    # load array
    with open(filename1, 'rb') as handle:
        array1 = pickle.load(handle)
    tempCond = array1["Conditions"]
    # add temp array to aryPresOrd
    aryPresOrd = np.concatenate((aryPresOrd, tempCond), axis=0)
aryPresOrd = aryPresOrd.astype(int)

# deduce the number of time points (conditions/volumes) from len aryPresOrd
varNumTP = len(aryPresOrd)
varNumTPinSingleRun = int(varNumTP/varNumRuns)

# %%
# *** Load PNGs with aperture position information
print('------Load PNGs containing aperture position')

# Create list of png files to load:
lstPathsPNGofApertPos = [None] * varNumPNG
for idx01 in range(0, varNumPNG):
    lstPathsPNGofApertPos[idx01] = (strPathPNGofApertPos +
                                    str(idx01) + '.png')

# Load png files. The png data will be saved in a numpy array of the
# following order: aryApertPos[x-pixel, y-pixel, PngNumber]. The
# first three values per pixel (RGB) that sp.misc.imread function returns
# are not relevant here, what matters is the alpha channel (4th dimension).
# So only the 4th dimension is considered and we discard the others.
aryApertPos = np.zeros((tplPngSize[0], tplPngSize[1], varNumPNG))
for idx01 in range(0, varNumPNG):
    aryApertPos[:, :, idx01] = sp.misc.imread(
        lstPathsPNGofApertPos[idx01])[:, :, 3]

# Convert RGB values (0 to 255) to integer ones and zeros:
aryApertPos = (aryApertPos > 0).astype(int)

# up- or downsample the image
aryApertPos = aryApertPos[0::factorX, 0::factorY, :]
tplPngSize = aryApertPos.shape[0:2]

# %%
# *** Combine information of aperture pos and presentation order
print('------Combine information of aperture pos and presentation order')
aryCond = aryApertPos[:, :, aryPresOrd[:, 0]]

# calculate an array that contains pixelwise box car functions for every
# motion direction
vecMtDrctn = np.unique(aryPresOrd[:, 1])[1:]  # exclude zeros
varNumMtDrctn = len(vecMtDrctn)
aryBoxCar = np.empty((aryCond.shape + (varNumMtDrctn,)), dtype='int64')
for ind, num in enumerate(vecMtDrctn):
    aryCondTemp = np.zeros((aryCond.shape), dtype='int64')
    lgcTempMtDrctn = [aryPresOrd[:, 1] == num][0]
    aryCondTemp[:, :, lgcTempMtDrctn] = np.copy(aryCond[:, :, lgcTempMtDrctn])
    aryBoxCar[:, :, :, ind] = aryCondTemp

# %%
# *** Prepare the creation of "neural" time courses models

# Vector with the x-indicies of the positions in the visual
# space at which to create pRF models.
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
lgcXYcombi = np.less(temp, tplPngSize[0]/2)
vecXY = vecXY[lgcXYcombi]
# define number of combinations for x and y positions
varNumXY = len(vecXY)

# Vector with the standard deviations of the pRF models in visual space.
vecPrfSd = np.linspace(varPrfStdMin,
                       varPrfStdMax,
                       varNumPrfSizes,
                       endpoint=True)
# check whether deg to pix ratio is the same in x and y dimensions
if (tplPngSize[0]/(varExtXmax-varExtXmin) ==
        tplPngSize[1]/(varExtYmax-varExtYmin)):
    pix2degX = tplPngSize[0]/(varExtXmax-varExtXmin)
else:
    print('------ERROR. Deg: Pix ratio differs between x and y dim. ' +
          'Check whether calculation of pRF sizes is still correct.')

# We need to account for the pixel:degree ratio by multiplying the pixel
# PrfSd in visual degree with that ratio to yield pixel PrfSd
vecPrfSd = np.multiply(vecPrfSd, pix2degX)

# put all possible combinations for three 2D Gauss parameters (x, y, sigma)
# and motion direction tuning curve models into tuple array
iterables = [vecXY, vecPrfSd]
aryPrfParams = list(itertools.product(*iterables))
# save this ary, it indicates for every model x-pos, y-pos, sigma, tng crv
np.save(strPathMdl+'aryPrfParams_xval',
        aryPrfParams)
# undo the zipping
for ind, item in enumerate(aryPrfParams):
    aryPrfParams[ind] = list(np.hstack(item))
aryPrfParams = np.array(aryPrfParams)

# calculate number of models
varNumMdls = varNumXY*varNumPrfSizes

# %%
if lgcCrteMdl:
    # *** Create "neural" time course models
    print('------Create pRF time course models')

    # Empty list for results (2D gaussian models):
    lstPrfMdls = [None] * varPar
    # Empty list for processes:
    lstPrcs = [None] * varPar
    # Create a queue to put the results in:
    queOut = mp.Queue()

    # Number of neural models:
    print('---------Number of pRF models that will be created: ' +
          str(varNumMdls))

    # put n =varPar number of chunks of the Gauss parameters into list for
    # parallel processing
    lstPrlPrfMdls = np.array_split(aryPrfParams, varPar)

    print('---------Creating parallel processes for pRF models')

    # Create processes:
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc] = mp.Process(target=funcNrlTcMotPred,
                                     args=(idxPrc,
                                           tplPngSize[0],
                                           tplPngSize[1],
                                           lstPrlPrfMdls[idxPrc],
                                           varNumTP,
                                           aryBoxCar,  # aryCond,
                                           strPathMdl,
                                           varNumMdls,
                                           varNumMtDrctn,
                                           varPar,
                                           queOut)
                                     )

        # Daemon (kills processes when exiting):
        lstPrcs[idxPrc].Daemon = True

    # Start processes:
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc].start()

    # Collect results from queue:
    for idxPrc in range(0, varPar):
        lstPrfMdls[idxPrc] = queOut.get(True)

    # Join processes:
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc].join()

    aryPrfMdls = np.empty((0, varNumTP, varNumMtDrctn), dtype='float32')
    # sort list by first entry (idxPrc), prll processes finish in mixed order
    lstPrfMdls = sorted(lstPrfMdls)
    # Put output into correct order:
    for idxRes in range(0, varPar):
        # Put fitting results into list, in correct order:
        aryPrfMdls = np.concatenate((aryPrfMdls, lstPrfMdls[idxRes][1]))
    # reshaping yields and array representing the visual space, of the form
    # aryPrfMdls[combi xy positions, pRF-size, tng curves, nr of time points],
    # which will hold the "neural" model time courses.
    aryPrfMdls = aryPrfMdls.reshape((varNumXY, varNumPrfSizes,
                                     varNumTP, varNumMtDrctn))
    np.save(strPathMdl + 'aryPrfMdls_xval', aryPrfMdls)
    # delete the list to save memory
    del(lstPrfMdls)
else:
    aryPrfMdls = np.load(strPathMdl + 'aryPrfMdls_xval.npy')

# %%
# *** Define training and test pRF models (i.e. design matrices)
print('------Define training and test pRF models')
aryPrfMdls = np.transpose(aryPrfMdls, (0, 1, 3, 2))
aryPrfMdls = aryPrfMdls.astype('float32')
aryPrfMdlsTrn = np.empty(aryPrfMdls.shape[0:3]
                         + (varNumXval,)
                         + (aryPrfMdls.shape[3] - varNumTPinSingleRun*2,),
                         dtype='float32')
aryPrfMdlsTst = np.empty(aryPrfMdls.shape[0:3]
                         + (varNumXval,)
                         + (varNumTPinSingleRun*2,), dtype='float32')

vecXval = np.arange(varNumXval)
lsSplit = np.array(np.split(np.arange(varNumTP), varNumXval))
for ind in vecXval:
    idx1 = np.where(ind != vecXval)[0]
    idx2 = np.where(ind == vecXval)[0]
    aryPrfMdlsTrn[:, :, :, ind, :] = aryPrfMdls[:, :, :,
                                                np.hstack(lsSplit[idx1])]
    aryPrfMdlsTst[:, :, :, ind, :] = aryPrfMdls[:, :, :,
                                                np.hstack(lsSplit[idx2])]
del(aryPrfMdls)  # (453, 22, 8, 1232)

# %%
# *** Convolve neural tc with HRF to get time courses
# before paralelisation was used for this but it runs quickly enough without
print('------Convolve training and test pRF models with HRF')

# get number of time points for training and test data set
varNumTpTrn = aryPrfMdlsTrn.shape[-1]
varNumTpTst = aryPrfMdlsTst.shape[-1]

# reshape the ary containing model predictors for convolution with HRF
aryPrfMdlsTrn = aryPrfMdlsTrn.reshape(varNumMdls * varNumMtDrctn *
                                      varNumXval, varNumTpTrn)
aryPrfMdlsTst = aryPrfMdlsTst.reshape(varNumMdls * varNumMtDrctn *
                                      varNumXval, varNumTpTst)
# convolve
vecHrf = funcHrf(varNumTpTrn, varTr)
aryPrfMdlsTrnConv = funcConvPar(aryPrfMdlsTrn, vecHrf, varNumTpTrn)
vecHrf = funcHrf(varNumTpTst, varTr)
aryPrfMdlsTstConv = funcConvPar(aryPrfMdlsTst, vecHrf, varNumTpTst)
# put array back into its former shape
aryPrfMdlsTrnConv = aryPrfMdlsTrnConv.reshape(varNumMdls, varNumMtDrctn,
                                              varNumXval, varNumTpTrn)
aryPrfMdlsTstConv = aryPrfMdlsTstConv.reshape(varNumMdls, varNumMtDrctn,
                                              varNumXval, varNumTpTst)

del(aryPrfMdlsTrn)
del(aryPrfMdlsTst)

# %%
# *** Find pRF models for voxel time courses
print('------Find pRF models for voxel time courses')

print('---------Preparing functional data for xvalidation')
# Load the simulated voxel time courses:
path2simTc = strPathMdl + 'simResp_xval_' + str(noiseIdx) + '.npy'
aryFunc = np.load(path2simTc)

aryFunc = aryFunc.astype('float32')

aryFuncTrn = np.empty((aryFunc.shape[0], varNumXval,
                       aryFunc.shape[1] - varNumTPinSingleRun*2),
                      dtype='float32')
aryFuncTst = np.empty((aryFunc.shape[0], varNumXval,
                      varNumTPinSingleRun*2),
                      dtype='float32')

vecXval = np.arange(varNumXval)
lsSplit = np.array(np.split(np.arange(varNumTP), varNumXval))
for ind in vecXval:
    idx1 = np.where(ind != vecXval)[0]
    idx2 = np.where(ind == vecXval)[0]
    aryFuncTrn[:, ind, :] = aryFunc[:, np.hstack(lsSplit[idx1])]
    aryFuncTst[:, ind, :] = aryFunc[:, np.hstack(lsSplit[idx2])]

print('---------Preparing parallel pRF model finding')

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
vecMdlXY = vecMdlXY[lgcXYcombi]

# get combinations of all model parameters (this time in deg of vis angle)
iterables = [vecMdlXY, vecMdlSd]
aryMdls = list(itertools.product(*iterables))
for ind, item in enumerate(aryMdls):
    aryMdls[ind] = list(np.hstack(item))
aryMdls = np.array(aryMdls)

# Empty list for results (parameters of best fitting pRF model):
lstPrfRes = [None] * varPar

# Empty list for processes:
lstPrcs = [None] * varPar

# Create a queue to put the results in:
queOut = mp.Queue()

# Number of voxels for which pRF finding will be performed:
varNumVoxInc = aryFunc.shape[0]

print('---------Number of simulated time course on which pRF finding will ' +
      'be performed: ' + str(varNumVoxInc))

# Put functional data into chunks:
lstFuncTrn = np.array_split(aryFuncTrn, varPar)
lstFuncTst = np.array_split(aryFuncTst, varPar)

# We don't need the original array with the functional data anymore:
del(aryFuncTrn)
del(aryFuncTst)

print('---------Creating parallel processes')

for idxPrc in range(0, varPar):
    lstPrcs[idxPrc] = mp.Process(target=funcFindPrfMltpPrdXVal,
                                 args=(idxPrc,
                                       lstFuncTrn[idxPrc],
                                       lstFuncTst[idxPrc],
                                       aryPrfMdlsTrnConv,
                                       aryPrfMdlsTstConv,
                                       aryMdls,
                                       queOut)
                                 )

    # Daemon (kills processes when exiting):
    lstPrcs[idxPrc].Daemon = True

# Start processes:
for idxPrc in range(0, varPar):
    lstPrcs[idxPrc].start()

# Collect results from queue:
for idxPrc in range(0, varPar):
    lstPrfRes[idxPrc] = queOut.get(True)

# Join processes:
for idxPrc in range(0, varPar):
    lstPrcs[idxPrc].join()

print('---------Save pRF results to list')
the_filename = (strPathMdl + 'lstPrfRes_Noise_xval' + str(noiseIdx) +
                '.pickle')
with open(the_filename, 'wb') as f:
    pickle.dump(lstPrfRes, f)

print('---------Prepare pRF finding results for export')

# Create list for vectors with fitting results, in order to put the results
# into the correct order:
lstResXpos = [None] * varPar
lstResYpos = [None] * varPar
lstResSd = [None] * varPar

# Put output into correct order:
for idxRes in range(0, varPar):

    # Index of results (first item in output list):
    varTmpIdx = lstPrfRes[idxRes][0]

    # Put fitting results into list, in correct order:
    lstResXpos[varTmpIdx] = lstPrfRes[idxRes][1]
    lstResYpos[varTmpIdx] = lstPrfRes[idxRes][2]
    lstResSd[varTmpIdx] = lstPrfRes[idxRes][3]

# Concatenate output vectors (into the same order as the voxels that were
# included in the fitting):
aryBstXpos = np.zeros(0)
aryBstYpos = np.zeros(0)
aryBstSd = np.zeros(0)
for idxRes in range(0, varPar):
    aryBstXpos = np.append(aryBstXpos, lstResXpos[idxRes])
    aryBstYpos = np.append(aryBstYpos, lstResYpos[idxRes])
    aryBstSd = np.append(aryBstSd, lstResSd[idxRes])

# Delete unneeded large objects:
del(lstPrfRes)
del(lstResXpos)
del(lstResYpos)
del(lstResSd)

# Array for pRF finding results, of the form
# aryPrfRes[total-number-of-voxels, 0:3], where the 2nd dimension
# contains the parameters of the best-fitting pRF model for the voxel, in
# the order (0) pRF-x-pos, (1) pRF-y-pos, (2) pRF-SD, (3) pRF-R2.
aryPrfRes = np.zeros((varNumVoxInc, 3))

# Put results form pRF finding into array (they originally needed to be
# saved in a list due to parallelisation).
aryPrfRes[:, 0] = aryBstXpos
aryPrfRes[:, 1] = aryBstYpos
aryPrfRes[:, 2] = aryBstSd

# Save nii results:
np.save(strPathMdl+'aryPrfRes_NoiseLev_xval' + str(noiseIdx),
        aryPrfRes)

# %%
# *** Report time

varTme02 = time.time()
varTme03 = varTme02 - varTme01
print('-Elapsed time: ' + str(varTme03) + ' s')
print('-Done.')
