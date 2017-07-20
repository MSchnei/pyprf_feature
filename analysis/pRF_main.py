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


# %% Import modules
import os
import numpy as np
import nibabel as nb
import time
import multiprocessing as mp
import sys
from scipy import stats
from pRF_utils import loadNiiData, saveNiiData, calcR2, calcFstats
from pRF_mdlCrt import (loadPng, loadPrsOrd, crtPwBoxCarFn, cnvlPwBoxCarFn,
                        crtPrfNrlTc)
from pRF_filtering import funcSmthTmp
from pRF_funcFindPrf import funcFindPrf, funcFindPrfXval
from pRF_funcFindPrfGpu import funcFindPrfGpu
from pRF_calcR2_getBetas import getBetas

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
    aryPngData = loadPng(cfg.varNumVol, cfg.tplPngSize, cfg.strPathPng)

    # *** Load presentation order of motion directions
    aryPresOrd = loadPrsOrd(cfg.vecRunLngth, cfg.strPathPresOrd,
                            cfg.vecVslStim)[1]

    # *** if lgcAoM, reduce motion directions from 8 to 4
    if cfg.lgcAoM:
        print('------Reduce motion directions from 8 to 4')
        aryPresOrd[aryPresOrd == 5] = 1
        aryPresOrd[aryPresOrd == 6] = 2
        aryPresOrd[aryPresOrd == 7] = 3
        aryPresOrd[aryPresOrd == 8] = 4

    vecMtDrctn = np.unique(aryPresOrd)[1:]  # exclude zeros
    cfg.varNumMtDrctn = len(vecMtDrctn) * cfg.switchHrfSet

    # *** create pixel-wise boxcar functions
    aryBoxCar = crtPwBoxCarFn(cfg.varNumVol, aryPngData, aryPresOrd,
                              vecMtDrctn)
    del(aryPngData)
    del(aryPresOrd)

    # *** Create neural time course models
    aryNrlTc = crtPrfNrlTc(aryBoxCar, cfg.varNumMtDrctn, cfg.varNumVol,
                           cfg.tplPngSize, cfg.varNumX, cfg.varExtXmin,
                           cfg.varExtXmax, cfg.varNumY, cfg.varExtYmin,
                           cfg.varExtYmax, cfg.varNumPrfSizes,
                           cfg.varPrfStdMin, cfg.varPrfStdMax, cfg.varPar)
    # aryNrlTc will have shape (25, 25, 22, 5, 1204)
    name = '/media/sf_D_DRIVE/MotionLocaliser/Analysis/P02/FitResults/Compare/aryNrlTc.npy'
    np.save(name, aryNrlTc)
    # aryNrlTc = np.load(name)

    # *** convolve every neural time course model with hrf function(s)
    aryPrfTc = cnvlPwBoxCarFn(aryNrlTc, cfg.varNumVol, cfg.varTr,
                              cfg.tplPngSize, cfg.varNumMtDrctn,
                              cfg.switchHrfSet, cfg.lgcOldSchoolHrf,
                              cfg.varPar,
                              )
    # aryPrfTc will have shape (25, 25, 22, 5*1, 1204)
    name = '/media/sf_D_DRIVE/MotionLocaliser/Analysis/P02/FitResults/Compare/aryPrfTc.npy'
    np.save(name, aryPrfTc)
    # aryPrfTc = np.load(name)

    # *** Save pRF time course models
    np.save(cfg.strPathMdl, aryPrfTc)

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
    assert vecPrfTcShp[0] == cfg.varNumX and \
        vecPrfTcShp[1] == cfg.varNumY and \
        vecPrfTcShp[2] == cfg.varNumPrfSizes and \
        vecPrfTcShp[3] == cfg.varNumMtDrctn and \
        vecPrfTcShp[4] == cfg.varNumVol, strErrMsg

# %% Find pRF models for voxel time courses

print('------Find pRF models for voxel time courses')

# Load mask (to restrict model finding):
niiMask = nb.load(cfg.strPathNiiMask)
aryMask = niiMask.get_data().astype('bool')
# Load data from functional runs
aryFunc = loadNiiData(cfg.lstNiiFls, strPathNiiMask=cfg.strPathNiiMask,
                      strPathNiiFunc=cfg.strPathNiiFunc)

print('---------Consider only training pRF time courses and func data')
# derive logical for training/test runs
lgcTrnTst = np.ones(np.sum(cfg.vecRunLngth), dtype=bool)
lgcTrnTst[np.cumsum(cfg.vecRunLngth)[cfg.varTestRun-1]:np.cumsum(
          cfg.vecRunLngth)[cfg.varTestRun]] = False

# split in training and test runs
aryPrfTc = aryPrfTc[..., lgcTrnTst]
aryFunc = aryFunc[..., lgcTrnTst]

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
vecMdlXpos = np.linspace(cfg.varExtXmin, cfg.varExtXmax, cfg.varNumX,
                         endpoint=True)

# Vector with the moddeled y-positions of the pRFs:
vecMdlYpos = np.linspace(cfg.varExtYmin, cfg.varExtYmax, cfg.varNumY,
                         endpoint=True)

# Vector with the moddeled standard deviations of the pRFs:
vecMdlSd = np.linspace(cfg.varPrfStdMin, cfg.varPrfStdMax, cfg.varNumPrfSizes,
                       endpoint=True)

# If using the GPU version, we set parallelisation factor to one, because
# parallelisation is done within the GPU function (no separate CPU threads).
if cfg.strVersion == 'gpu':
    cfg.varPar = 1

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
      'performed: ' + str(aryFunc.shape[0]))
# zscore/demean predictors and responses after smoothing
aryPrfTc = stats.zscore(aryPrfTc, axis=4, ddof=2)
aryFunc = np.subtract(aryFunc, np.mean(aryFunc, axis=1)[:, None])
if cfg.lgcXval:

    if cfg.strVersion != 'gpu':

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
        # save aryFunc so we can load it later for R2 determination, but for
        # now we delete it to save memory
        np.save(cfg.strPathOut + '_aryFunc', aryFunc)
        del(aryFunc)

    else:
        raise ValueError('Crossvalidation on GPU currently not implemented.')

else:
    # Put functional data into chunks:
    lstFunc = np.array_split(aryFunc, cfg.varPar)
    # We don't need the original array with the functional data anymore:
    del(aryFunc)





print('---------Creating parallel processes')

# Create processes:
if cfg.lgcXval:

    if cfg.strVersion != 'gpu':

        for idxPrc in range(0, cfg.varPar):
            lstPrcs[idxPrc] = mp.Process(target=funcFindPrfXval,
                                         args=(idxPrc, vecMdlXpos, vecMdlYpos,
                                               vecMdlSd, lstFuncTrn[idxPrc],
                                               lstFuncTst[idxPrc],
                                               lstPrfMdlsTrn, lstPrfMdlsTst,
                                               cfg.lgcCython, cfg.varNumXval,
                                               queOut)
                                         )
            # Daemon (kills processes when exiting):
            lstPrcs[idxPrc].Daemon = True
    else:
        raise ValueError('Crossvalidation on GPU currently not implemented.')

else:

    if cfg.strVersion != 'gpu':

        for idxPrc in range(0, cfg.varPar):
            lstPrcs[idxPrc] = mp.Process(target=funcFindPrf,
                                         args=(idxPrc, vecMdlXpos, vecMdlYpos,
                                               vecMdlSd, lstFunc[idxPrc],
                                               aryPrfTc, cfg.lgcCython, queOut)
                                         )
            # Daemon (kills processes when exiting):
            lstPrcs[idxPrc].Daemon = True
    else:
        print('---------pRF finding on GPU')
        # Create processes:
        for idxPrc in range(0, cfg.varPar):
            lstPrcs[idxPrc] = mp.Process(target=funcFindPrfGpu,
                                         args=(idxPrc,
                                               vecMdlXpos,
                                               vecMdlYpos,
                                               vecMdlSd,
                                               lstFunc[idxPrc],
                                               aryPrfTc,
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
    aryBstBetas = np.zeros((0, cfg.varNumMtDrctn))
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
                                     args=(idxPrc, vecMdlXpos, vecMdlYpos,
                                           vecMdlSd, aryPrfTc, lstFunc[idxPrc],
                                           lstBstMdls[idxPrc], queOut)
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
aryPrfRes[:, :, :, 5] = np.sqrt(np.add(np.power(aryPrfRes[:, :, :, 0], 2.0),
                                       np.power(aryPrfRes[:, :, :, 1], 2.0)))
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
for idxOut in range(0, len(lstNiiNames)):
    # Create nii object for results:
    niiOut = nb.Nifti1Image(aryPrfRes[:, :, :, idxOut],
                            niiMask.affine,
                            header=niiMask.header
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
