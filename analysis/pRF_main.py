# -*- coding: utf-8 -*-
"""Find best fitting model time courses for population receptive fields."""

# Part of py_pRF_motion library
# Copyright (C) 2016  Marian Schneider, Ingo Marquardt
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

print('---pRF analysis')


# %% Import modules
import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))
import numpy as np
import nibabel as nb
import time
import multiprocessing as mp
from pRF_mdlCrt import (loadPng, loadPrsOrd, crtBoxCarWeights, crtPwBoxCarFn,
                        rsmplInHighRes, funcPrfTc)
from pRF_filtering import funcSmthTmp
from pRF_funcFindPrf import funcFindPrf, funcFindPrfXval
from pRF_calcR2_getBetas import getBetas
from pRF_hrfutils import spmt, dspmt, ddspmt, cnvlTc, cnvlTcOld
import sys

# %% get some parameters from command line
sys.argv = sys.argv[1:]
varNumCmdArgs = len(sys.argv)
print 'Argument List:', str(sys.argv)

if varNumCmdArgs == 0:
    # import the default cfg file
    import pRF_config as cfg
else:
    print "------Imported custom cfg file"
    # determine the type of aperture
    strCfgFile = str(sys.argv[0])
    strCfgFilePath = os.path.dirname(strCfgFile)
    strCfgFileName = os.path.split(strCfgFile)[1]
    strCfgFileName = os.path.splitext(strCfgFileName)[0]

    sys.path.insert(0, strCfgFilePath)
    cfg = __import__(strCfgFileName, globals(), locals(), [])
    del sys.path[0]

# %% Check time
varTme01 = time.time()

# %% Create new pRF time course models, or load existing models

if cfg.lgcCrteMdl:

    # *** Load PNGs
    aryPngData = loadPng(cfg.varNumVol,
                         cfg.tplPngSize,
                         cfg.strPathPng)

    # *** Load presentation order of motion directions
    aryPresOrd = loadPrsOrd(cfg.vecRunLngth,
                            cfg.strPathPresOrd,
                            cfg.vecVslStim)

    # *** if lgcAoM, reduce motion directions from 8 to 4
    if cfg.lgcAoM:
        print('------Reduce motion directions from 8 to 4')
        aryPresOrd[aryPresOrd == 5] = 1
        aryPresOrd[aryPresOrd == 6] = 2
        aryPresOrd[aryPresOrd == 7] = 3
        aryPresOrd[aryPresOrd == 8] = 4
        aryPresOrd[aryPresOrd == 9] = 5

    vecMtDrctn = np.unique(aryPresOrd)[1:]  # exclude zeros
    # determine new numbe rof motion directions
    cfg.varNumMtDrctn = len(vecMtDrctn) * cfg.switchHrfSet

    if cfg.lgcVonMises:
        vecIndV1 = crtBoxCarWeights(cfg.lgcVonMises,
                                    cfg.lgcAoM,
                                    len(vecMtDrctn)-1,
                                    cfg.varKappa)
    else:
        vecIndV1 = crtBoxCarWeights(cfg.lgcVonMises,
                                    cfg.lgcAoM,
                                    len(vecMtDrctn)-1)

    # *** add the indices for the static condition
    vecIndV1 = np.append(vecIndV1, np.zeros((1, vecIndV1.shape[1])), 0)
    vecIndV1 = np.append(vecIndV1, np.zeros((vecIndV1.shape[0], 1)), 1)
    vecIndV1[-1, -1] = 1

    # *** create pixel-wise boxcar functions
    aryBoxCar = crtPwBoxCarFn(cfg.varNumVol,
                              aryPngData,
                              aryPresOrd,
                              vecIndV1,
                              vecMtDrctn)
    del(aryPngData)
    del(aryPresOrd)

    # *** convolve every pixel box car function with hrf function(s)
#    aryBoxCarConv = cnvlPwBoxCarFn(aryBoxCar,
#                                   cfg.varNumVol,
#                                   cfg.varTr,
#                                   cfg.tplPngSize,
#                                   cfg.varNumMtDrctn,
#                                   cfg.switchHrfSet,
#                                   cfg.varPar,
#                                   cfg.lgcOldSchoolHrf,
#                                   )
    print('------Convolve every pixel box car function with hrf function(s)')

    # Create hrf time course function:
    if cfg.switchHrfSet == 3:
        lstHrf = [spmt, dspmt, ddspmt]
    elif cfg.switchHrfSet == 2:
        lstHrf = [spmt, dspmt]
    elif cfg.switchHrfSet == 1:
        lstHrf = [spmt]

    # Reshape png data:
    aryBoxCar = np.reshape(aryBoxCar,
                           ((aryBoxCar.shape[0] * aryBoxCar.shape[1] *
                            aryBoxCar.shape[2]), aryBoxCar.shape[3]))

    # Put input data into chunks:
    lstBoxCar = np.array_split(aryBoxCar, cfg.varPar)
    # We don't need the original array with the input data anymore:
    del(aryBoxCar)

    # Create a queue to put the results in:
    queOut = mp.Queue()

    # Empty list for processes:
    lstPrcs = [None] * cfg.varPar

    # Empty list for results of parallel processes:
    lstConv = [None] * cfg.varPar

    print('---------Creating parallel processes')

    if cfg.lgcOldSchoolHrf:
        for idxPrc in range(0, cfg.varPar):
            lstPrcs[idxPrc] = mp.Process(target=cnvlTcOld,
                                         args=(idxPrc,
                                               lstBoxCar[idxPrc],
                                               cfg.varTr,
                                               cfg.varNumVol,
                                               queOut)
                                         )
    else:
        # Create processes:
        for idxPrc in range(0, cfg.varPar):
            lstPrcs[idxPrc] = mp.Process(target=cnvlTc,
                                         args=(idxPrc,
                                               lstBoxCar[idxPrc],
                                               lstHrf,
                                               cfg.varTr,
                                               cfg.varNumVol,
                                               queOut)
                                         )

        # Daemon (kills processes when exiting):
        lstPrcs[idxPrc].Daemon = True

    # Start processes:
    for idxPrc in range(0, cfg.varPar):
        lstPrcs[idxPrc].start()

    # Collect results from queue:
    for idxPrc in range(0, cfg.varPar):
        lstConv[idxPrc] = queOut.get(True)

    # Join processes:
    for idxPrc in range(0, cfg.varPar):
        lstPrcs[idxPrc].join()

    print('---------Collecting results from parallel processes')

    # Put output into correct order:
    lstConv = sorted(lstConv)

    # Concatenate convolved pixel time courses (into the same order as they
    # were entered into the analysis):
    aryBoxCarConv = np.zeros((0, cfg.switchHrfSet, cfg.varNumVol))
    for idxRes in range(0, cfg.varPar):
        aryBoxCarConv = np.concatenate((aryBoxCarConv, lstConv[idxRes][1]),
                                       axis=0)
    del(lstConv)

    # Reshape results:
    aryBoxCarConv = np.reshape(aryBoxCarConv,
                               [cfg.tplPngSize[0],
                                cfg.tplPngSize[1],
                                cfg.varNumMtDrctn,
                                cfg.varNumVol])
    # aryBoxCarConv will have shape 128, 128, 15, 688

    # *** resample pixel-time courses in high-res visual space
    aryBoxCarConvHigh = rsmplInHighRes(aryBoxCarConv,
                                       cfg.tplPngSize,
                                       cfg.tplVslSpcHighSze,
                                       cfg.varNumMtDrctn,
                                       cfg.varNumVol)

    # aryBoxCarConvHigh will have shape 200, 200, 15, 688

    # *** Create pRF time courses models
    # The pRF time course models are created using the super-sampled model of
    # the pixel time courses.

    print('------Create pRF time course models')

    # Upsampling factor:
    if (cfg.tplVslSpcHighSze[0] / cfg.varNumX) == (cfg.tplVslSpcHighSze[1]
                                                   / cfg.varNumY):
        varFctUp = cfg.tplVslSpcHighSze[0] / cfg.varNumX
    else:
        print('------ERROR. Dimensions of upsampled visual space do not ' +
              'agree with specified number of pRFs to model.')

    # Vector with the x-indicies of the positions in the super-sampled visual
    # space at which to create pRF models.
    vecX = np.linspace(0,
                       (cfg.tplVslSpcHighSze[0] - 1),
                       cfg.varNumX,
                       endpoint=True)

    # Vector with the y-indicies of the positions in the super-sampled visual
    # space at which to create pRF models.
    vecY = np.linspace(0,
                       (cfg.tplVslSpcHighSze[1] - 1),
                       cfg.varNumY,
                       endpoint=True)

    # Vector with the standard deviations of the pRF models. We need to convert
    # the standard deviation values from degree of visual angle to the
    # dimensions of the visual space. We calculate the scaling factor from
    # degrees of visual angle to pixels in the *upsampled* visual space
    # separately for the x- and the y-directions (the two should be the same).
    varDgr2PixUpX = cfg.tplVslSpcHighSze[0] / (cfg.varExtXmax - cfg.varExtXmin)
    varDgr2PixUpY = cfg.tplVslSpcHighSze[1] / (cfg.varExtYmax - cfg.varExtYmin)

    # The factor relating pixels in the upsampled visual space to degrees of
    # visual angle should be roughly the same (allowing for some rounding error
    # if the visual stimulus was not square):
    if 0.5 < np.absolute((varDgr2PixUpX - varDgr2PixUpY)):
        print('------ERROR. The ratio of X and Y dimensions in stimulus ' +
              'space (in degrees of visual angle) and the ratio of X and Y ' +
              'dimensions in the upsampled visual space do not agree')

    # Vector with pRF sizes to be modelled (still in degree of visual angle):
    vecPrfSd = np.linspace(cfg.varPrfStdMin,
                           cfg.varPrfStdMax,
                           cfg.varNumPrfSizes,
                           endpoint=True)

    # We multiply the vector with the pRF sizes to be modelled with the scaling
    # factor (for the x-dimensions - as we have just found out, the scaling
    # factors for the x- and y-direction are identical, except for rounding
    # error). Now the vector with the pRF sizes to be modelled is can directly
    # be used for the creation of Gaussian pRF models in upsampled visual
    # space.
    vecPrfSd = np.multiply(vecPrfSd, varDgr2PixUpX)

    # Number of pRF models to be created (i.e. number of possible combinations
    # of x-position, y-position, and standard deviation):
    varNumMdls = cfg.varNumX * cfg.varNumY * cfg.varNumPrfSizes

    # Array for the x-position, y-position, and standard deviations for which
    # pRF model time courses are going to be created, where the columns
    # correspond to: (0) an index starting from zero, (1) the x-position, (2)
    # the y-position, and (3) the standard deviation. The parameters are in
    # units of the upsampled visual space.
    aryMdlParams = np.zeros((varNumMdls, 4))

    # Counter for parameter array:
    varCntMdlPrms = 0

    # Put all combinations of x-position, y-position, and standard deviations
    # into the array:

    # Loop through x-positions:
    for idxX in range(0, cfg.varNumX):

        # Loop through y-positions:
        for idxY in range(0, cfg.varNumY):

            # Loop through standard deviations (of Gaussian pRF models):
            for idxSd in range(0, cfg.varNumPrfSizes):

                # Place index and parameters in array:
                aryMdlParams[varCntMdlPrms, 0] = varCntMdlPrms
                aryMdlParams[varCntMdlPrms, 1] = vecX[idxX]
                aryMdlParams[varCntMdlPrms, 2] = vecY[idxY]
                aryMdlParams[varCntMdlPrms, 3] = vecPrfSd[idxSd]

                # Increment parameter index:
                varCntMdlPrms = varCntMdlPrms + 1

    # The long array with all the combinations of model parameters is put into
    # separate chunks for parallelisation, using a list of arrays.
    lstMdlParams = np.array_split(aryMdlParams, cfg.varPar)

    # Create a queue to put the results in:
    queOut = mp.Queue()

    # Empty list for results from parallel processes (for pRF model time course
    # results):
    lstPrfTc = [None] * cfg.varPar

    # Empty list for processes:
    lstPrcs = [None] * cfg.varPar

    print('---------Creating parallel processes')

    # Create processes:
    for idxPrc in range(0, cfg.varPar):
        lstPrcs[idxPrc] = mp.Process(target=funcPrfTc,
                                     args=(idxPrc,
                                           lstMdlParams[idxPrc],
                                           cfg.tplVslSpcHighSze,
                                           cfg.varNumVol,
                                           aryBoxCarConvHigh,
                                           queOut)
                                     )
        # Daemon (kills processes when exiting):
        lstPrcs[idxPrc].Daemon = True

    # Start processes:
    for idxPrc in range(0, cfg.varPar):
        lstPrcs[idxPrc].start()

    # Collect results from queue:
    for idxPrc in range(0, cfg.varPar):
        lstPrfTc[idxPrc] = queOut.get(True)

    # Join processes:
    for idxPrc in range(0, cfg.varPar):
        lstPrcs[idxPrc].join()

    # Put output arrays from parallel process into one big array
    lstPrfTc = sorted(lstPrfTc)
    aryPrfTc = np.empty((0, cfg.varNumMtDrctn, cfg.varNumVol))
    for idx in range(0, cfg.varPar):
        print('---------Order list by index: ' + str(lstPrfTc[idx][0]))
        aryPrfTc = np.concatenate((aryPrfTc, lstPrfTc[idx][1]), axis=0)

    # check that all the models were collected correctly
    assert aryPrfTc.shape[0] == varNumMdls

    # Clean up:
    del(aryMdlParams)
    del(lstMdlParams)
    del(lstPrfTc)

    # Array representing the low-resolution visual space, of the form
    # aryPrfTc[x-position, y-position, pRF-size, varNum Vol], which will hold
    # the pRF model time courses.
    aryPrfTc4D = np.zeros([cfg.varNumX,
                           cfg.varNumY,
                           cfg.varNumPrfSizes,
                           cfg.varNumMtDrctn,
                           cfg.varNumVol])

    # We use the same loop structure for organising the pRF model time courses
    # that we used for creating the parameter array. Counter:
    varCntMdlPrms = 0

    # Put all combinations of x-position, y-position, and standard deviations
    # into the array:

    # Loop through x-positions:
    for idxX in range(0, cfg.varNumX):

        # Loop through y-positions:
        for idxY in range(0, cfg.varNumY):

            # Loop through standard deviations (of Gaussian pRF models):
            for idxSd in range(0, cfg.varNumPrfSizes):

                # Put the pRF model time course into its correct position in
                # the 4D array, leaving out the first column (which contains
                # the index):
                aryPrfTc4D[idxX, idxY, idxSd, :, :] = aryPrfTc[
                    varCntMdlPrms, :, :]

                # Increment parameter index:
                varCntMdlPrms = varCntMdlPrms + 1

    # Change array name for consistency, and delete unnecessary copy:
    aryPrfTc = np.copy(aryPrfTc4D)
    del(aryPrfTc4D)
    # *************************************************************************

    # *************************************************************************
    # *** Save pRF time course models

    # Save the 4D array as '*.npy' file:
    np.save(cfg.strPathMdl,
            aryPrfTc)

    # Set test for correct dimensions of '*.npy' file to true:
    lgcDim = True
    # *************************************************************************

else:
    # %% Load existing pRF time course models

    print('------Load pRF time course models')

    # Load the file:
    aryPrfTc = np.load(cfg.strPathMdl)

    # Check whether pRF time course model matrix has the expected dimensions:
    vecPrfTcShp = aryPrfTc.shape

    # Logical test for correct dimensions:
    strErrMsg = ('Dimensions of specified pRF time course models ' +
                 'do not agree with specified model parameters')
    print(strErrMsg)
    assert vecPrfTcShp[0] == cfg.varNumX and \
        vecPrfTcShp[1] == cfg.varNumY and \
        vecPrfTcShp[2] == cfg.varNumPrfSizes and \
        vecPrfTcShp[3] == cfg.varNumMtDrctn and \
        vecPrfTcShp[4] == cfg.varNumVol, strErrMsg

# %% Find pRF models for voxel time courses

print('------Find pRF models for voxel time courses')

print('---------Loading nii data')
# Load mask (to restrict model fining):
niiMask = nb.load(cfg.strPathNiiMask)
# Get nii header of mask:
hdrMsk = niiMask.header
# Get nii 'affine':
affMsk = niiMask.affine
# Load the data into memory:
aryMask = niiMask.get_data().astype('bool')

# prepare aryFunc for functional data
aryFunc = np.empty((np.sum(aryMask), 0), dtype='float32')
for idx in np.arange(len(cfg.lstNiiFls)):
    print('------------Loading run: ' + str(idx+1))
    # Load 4D nii data:
    niiFunc = nb.load(os.path.join(cfg.strPathNiiFunc,
                                   cfg.lstNiiFls[idx]))
    # Load the data into memory:
    aryFuncTemp = niiFunc.get_data()
    aryFunc = np.append(aryFunc, aryFuncTemp[aryMask, :], axis=1)

# remove unneccary array
del(aryFuncTemp)

# Take mean over time of functional nii data:
aryFuncMean = np.mean(aryFunc, axis=1)
# Logical test for voxel inclusion: is the mean of functional time series
# above the cutoff value?
aryLgc = np.greater(aryFuncMean, cfg.varIntCtf)
# update 3D mask accordingly
aryMask[aryMask] = np.copy(aryLgc)

# Array with functional data for which conditions (cutoff value)
# are fullfilled:
aryFunc = aryFunc[aryLgc, :]
# Number of voxels for which pRF finding will be performed:
varNumVoxInc = aryFunc.shape[0]

# Convert preprocessing parameters (for temporal and spatial smoothing) from
# SI units (i.e. [s] and [mm]) into units of data array:
cfg.varSdSmthTmp = np.divide(cfg.varSdSmthTmp, cfg.varTr)

# Perform temporal smoothing on pRF time course models:
if 0.0 < cfg.varSdSmthTmp:
    print('---------Temporal smoothing on pRF time course models')
    print('------------SD tmp smooth is: ' + str(cfg.varSdSmthTmp))
    aryPrfTc = funcSmthTmp(aryPrfTc,
                           cfg.varSdSmthTmp,
                           )

print('---------Preparing parallel pRF model finding')

# Vector with the moddeled x-positions of the pRFs:
vecMdlXpos = np.linspace(cfg.varExtXmin,
                         cfg.varExtXmax,
                         cfg.varNumX,
                         endpoint=True)

# Vector with the moddeled y-positions of the pRFs:
vecMdlYpos = np.linspace(cfg.varExtYmin,
                         cfg.varExtYmax,
                         cfg.varNumY,
                         endpoint=True)

# Vector with the moddeled standard deviations of the pRFs:
vecMdlSd = np.linspace(cfg.varPrfStdMin,
                       cfg.varPrfStdMax,
                       cfg.varNumPrfSizes,
                       endpoint=True)

# Empty list for results (parameters of best fitting pRF model):
lstPrfRes = [None] * cfg.varPar

# Empty list for processes:
lstPrcs = [None] * cfg.varPar

# Counter for parallel processes:
varCntPar = 0

# Counter for output of parallel processes:
varCntOut = 0

# Create a queue to put the results in:
queOut = mp.Queue()

print('---------Number of voxels on which pRF finding will be ' +
      'performed: ' + str(varNumVoxInc))

if cfg.lgcXval:
    # get number of time points
    varNumTP = aryFunc.shape[1]
    # make sure that the predictors are demeaned
    aryPrfTc = np.subtract(aryPrfTc, np.mean(aryPrfTc, axis=4)[
        :, :, :, :, None])
    # make sure that the data is demeaned
    aryFunc = np.subtract(aryFunc, np.mean(aryFunc, axis=1)[:, None])
    # prepare cross validation
    vecXval = np.arange(cfg.varNumXval)
    lsSplit = np.array(np.split(np.arange(varNumTP),
                                np.cumsum(cfg.vecRunLngth)))
    # get rid of last element, which is empty
    lsSplit = lsSplit[:-1]
    # put the pRF models into different training and test folds
    lstPrfMdlsTrn = []
    lstPrfMdlsTst = []
    for ind in vecXval:
        idx1 = np.where(ind != vecXval)[0]
        idx2 = np.where(ind == vecXval)[0]
        lstPrfMdlsTrn.append(aryPrfTc[:, :, :, :,
                                      np.hstack(lsSplit[idx1])])
        lstPrfMdlsTst.append(aryPrfTc[:, :, :, :,
                                      np.hstack(lsSplit[idx2])])

    # put the functional data into different training and test folds
    lstFunc = np.array_split(aryFunc, cfg.varPar)

    lstFuncTrn = [[] for x in xrange(cfg.varPar)]
    lstFuncTst = [[] for x in xrange(cfg.varPar)]
    for ind1 in np.arange(len(lstFunc)):
        # take voxels for ind1 paralelization
        aryFuncTemp = lstFunc[ind1]
        lstFuncTrnSplit = []
        lstFuncTstSplit = []
        for ind2 in vecXval:
            # create ind2-fold for xvalidation
            idx1 = np.where(ind2 != vecXval)[0]
            idx2 = np.where(ind2 == vecXval)[0]
            lstFuncTrnSplit.append(
                aryFuncTemp[:, np.hstack(lsSplit[idx1])])
            lstFuncTstSplit.append(
                aryFuncTemp[:, np.hstack(lsSplit[idx2])])
        lstFuncTrn[ind1] = lstFuncTrnSplit
        lstFuncTst[ind1] = lstFuncTstSplit

    del(lsSplit)
    del(lstFunc)
    del(lstFuncTrnSplit)
    del(lstFuncTstSplit)
    # save aryFunc so we can load it later for R2 determination, but for now
    # we delete it to save memory
    np.save(cfg.strPathOut + '_aryFunc', aryFunc)
    del(aryFunc)

else:
    # Put functional data into chunks:
    lstFunc = np.array_split(aryFunc, cfg.varPar)
    # We don't need the original array with the functional data anymore:
    del(aryFunc)

print('---------Creating parallel processes')

# Create processes:
if cfg.lgcXval:
    for idxPrc in range(0, cfg.varPar):
        lstPrcs[idxPrc] = mp.Process(target=funcFindPrfXval,
                                     args=(idxPrc,
                                           vecMdlXpos,
                                           vecMdlYpos,
                                           vecMdlSd,
                                           lstFuncTrn[idxPrc],
                                           lstFuncTst[idxPrc],
                                           lstPrfMdlsTrn,
                                           lstPrfMdlsTst,
                                           cfg.lgcCython,
                                           cfg.varNumXval,
                                           queOut)
                                     )
        # Daemon (kills processes when exiting):
        lstPrcs[idxPrc].Daemon = True
else:
    for idxPrc in range(0, cfg.varPar):
        lstPrcs[idxPrc] = mp.Process(target=funcFindPrf,
                                     args=(idxPrc,
                                           vecMdlXpos,
                                           vecMdlYpos,
                                           vecMdlSd,
                                           lstFunc[idxPrc],
                                           aryPrfTc,
                                           cfg.lgcCython,
                                           queOut)
                                     )
        # Daemon (kills processes when exiting):
        lstPrcs[idxPrc].Daemon = True

# Start processes:
for idxPrc in range(0, cfg.varPar):
    lstPrcs[idxPrc].start()

# Collect results from queue:
for idxPrc in range(0, cfg.varPar):
    lstPrfRes[idxPrc] = queOut.get(True)

# Join processes:
for idxPrc in range(0, cfg.varPar):
    lstPrcs[idxPrc].join()

print('---------Prepare pRF finding results for export')


# Put fitting results into list, in correct order:
lstPrfRes = sorted(lstPrfRes)

# Concatenate output vectors (into the same order as the voxels that were
# included in the fitting):
aryBstXpos = np.zeros(0)
aryBstYpos = np.zeros(0)
aryBstSd = np.zeros(0)
if not cfg.lgcXval:
    aryBstR2 = np.zeros(0)
    aryBstBetas = np.zeros((0, cfg.varNumMtDrctn+1))
for idxRes in range(0, cfg.varPar):
    aryBstXpos = np.append(aryBstXpos, lstPrfRes[idxRes][1])
    aryBstYpos = np.append(aryBstYpos, lstPrfRes[idxRes][2])
    aryBstSd = np.append(aryBstSd, lstPrfRes[idxRes][3])
    if not cfg.lgcXval:
        aryBstR2 = np.append(aryBstR2, lstPrfRes[idxRes][4])
        aryBstBetas = np.concatenate((aryBstBetas, lstPrfRes[idxRes][5]),
                                     axis=0)

# Delete unneeded large objects:
del(lstPrfRes)

# %%
# if we did model finding with cross validation, we never estimated
# the betas and R square for the full model. Do that now:
if cfg.lgcXval:
    print('------Find best betas and R2 values')
    # prepare list for results
    lstBetaRes = [None] * cfg.varPar
    # Empty list for processes:
    lstPrcs = [None] * cfg.varPar
    # get an array that shows best x, y, sigma for every voxel
    aryBstMdls = np.array([aryBstXpos, aryBstYpos, aryBstSd]).T
    # divide this ary in parts and put parts in list
    lstBstMdls = np.array_split(aryBstMdls, cfg.varPar)
    # put functional data into list
    aryFunc = np.load(cfg.strPathOut + '_aryFunc.npy')
    lstFunc = np.array_split(aryFunc, cfg.varPar)
    # delete aryFunc from memory and from disk
    del(aryFunc)
    os.remove(cfg.strPathOut + '_aryFunc.npy')

    for idxPrc in range(0, cfg.varPar):
        lstPrcs[idxPrc] = mp.Process(target=getBetas,
                                     args=(idxPrc,
                                           vecMdlXpos,
                                           vecMdlYpos,
                                           vecMdlSd,
                                           aryPrfTc,
                                           lstFunc[idxPrc],
                                           lstBstMdls[idxPrc],
                                           queOut)
                                     )
        # Daemon (kills processes when exiting):
        lstPrcs[idxPrc].Daemon = True

    # Start processes:
    for idxPrc in range(0, cfg.varPar):
        lstPrcs[idxPrc].start()

    # Collect results from queue:
    for idxPrc in range(0, cfg.varPar):
        lstBetaRes[idxPrc] = queOut.get(True)

    # Join processes:
    for idxPrc in range(0, cfg.varPar):
        lstPrcs[idxPrc].join()

    # Put output into correct order:
    lstBetaRes = sorted(lstBetaRes)

    # Concatenate output vectors
    aryBstR2 = np.zeros(0)
    aryBstBetas = np.zeros((0, cfg.varNumMtDrctn))
    for idxRes in range(0, cfg.varPar):
        aryBstR2 = np.append(aryBstR2, lstBetaRes[idxRes][1])
        aryBstBetas = np.concatenate((aryBstBetas, lstBetaRes[idxRes][2]),
                                     axis=0)

# %% Prepare for saving results

# Array for pRF finding results, of the form
# aryPrfRes[total-number-of-voxels, 0:3], where the 2nd dimension
# contains the parameters of the best-fitting pRF model for the voxel, in
# the order (0) pRF-x-pos, (1) pRF-y-pos, (2) pRF-SD, (3) pRF-R2.
aryPrfRes = np.zeros((niiMask.shape + (6,)))

# Put results form pRF finding into array (they originally needed to be
# saved in a list due to parallelisation).
aryPrfRes[aryMask, 0] = aryBstXpos
aryPrfRes[aryMask, 1] = aryBstYpos
aryPrfRes[aryMask, 2] = aryBstSd
aryPrfRes[aryMask, 3] = aryBstR2

# Calculate polar angle map:
aryPrfRes[:, :, :, 4] = np.arctan2(aryPrfRes[:, :, :, 1],
                                   aryPrfRes[:, :, :, 0])

# Calculate eccentricity map (r = sqrt( x^2 + y^2 ) ):
aryPrfRes[:, :, :, 5] = np.sqrt(np.add(np.power(aryPrfRes[:, :, :, 0],
                                                2.0),
                                       np.power(aryPrfRes[:, :, :, 1],
                                                2.0)))
# save as npy
np.save(cfg.strPathOut + '_aryPrfRes', aryPrfRes)
np.save(cfg.strPathOut + '_aryBstBetas', aryBstBetas)

# List with name suffices of output images:
lstNiiNames = ['_x_pos',
               '_y_pos',
               '_SD',
               '_R2',
               '_polar_angle',
               '_eccentricity']

print('---------Exporting results')

# Save nii results:
for idxOut in range(0, 6):
    # Create nii object for results:
    niiOut = nb.Nifti1Image(aryPrfRes[:, :, :, idxOut],
                            affMsk,
                            header=hdrMsk
                            )
    # Save nii:
    strTmp = (cfg.strPathOut + lstNiiNames[idxOut] + '.nii')
    nb.save(niiOut, strTmp)


# %%
# *** Report time

varTme02 = time.time()
varTme03 = varTme02 - varTme01
print('-Elapsed time: ' + str(varTme03) + ' s')
print('-Done.')
