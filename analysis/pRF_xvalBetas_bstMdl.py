# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 12:16:46 2017

@author: marian
"""
import sys
import os
import itertools
import numpy as np
import nibabel as nb
import multiprocessing as mp
from sklearn.model_selection import KFold
from pRF_calcR2_getBetas import getBetas
from pRF_utils import loadNiiData
from pRF_filtering import funcSmthTmp

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

# %% preperations
# Vector with the moddeled x-positions of the pRFs:
vecMdlXpos = np.linspace(cfg.varExtXmin, cfg.varExtXmax, cfg.varNumX,
                         endpoint=True)

# Vector with the moddeled y-positions of the pRFs:
vecMdlYpos = np.linspace(cfg.varExtYmin, cfg.varExtYmax, cfg.varNumY,
                         endpoint=True)

# Vector with the moddeled standard deviations of the pRFs:
vecMdlSd = np.linspace(cfg.varPrfStdMin, cfg.varPrfStdMax, cfg.varNumPrfSizes,
                       endpoint=True)

# %%
print('------Load best models and data')
# load the mask
niiMask = nb.load(cfg.strPathNiiMask)
aryMask = niiMask.get_data().astype('bool')

# get best models
aryRes = np.load(cfg.strPathOut + '_aryPrfRes.npy')
# mask the results array
aryRes = aryRes[aryMask, :]
# get an array that shows best x, y, sigma for every voxel
aryBstMdls = aryRes[:, 0:3]

iterables = [range(len(vecMdlXpos)), range(len(vecMdlYpos)), range(len(vecMdlSd))]
lstAllMdlInd = list(itertools.product(*iterables))
# TODO: this part should be replaced in the future when the best indices are
# saved directly
iterables = [vecMdlXpos, vecMdlYpos, vecMdlSd]
lstAllMdls = list(itertools.product(*iterables))
# for every row in aryBstMdls, get best model index
aryBstIndices = np.empty(len(aryBstMdls))
aryCounter = np.zeros(len(aryBstMdls))
for ind, mdl in enumerate(lstAllMdls):
    aryBstIndices[np.all(aryBstMdls == mdl, axis=1)] = ind
    aryCounter[np.all(aryBstMdls == mdl, axis=1)] += 1

# check that every voxel was visited only once
strErrMsg = ('It looks like at least voxel was revisted more than once. ' +
             'Check whether the R2 was calculated correctly')
assert np.sum(aryCounter) == len(aryCounter), strErrMsg

# get data for training runs
print('------Load training data')
aryFunc = loadNiiData(cfg.lstNiiFls, strPathNiiMask=cfg.strPathNiiMask,
                      strPathNiiFunc=cfg.strPathNiiFunc)

print('------Load prediction time courses')
# load predictors
aryPrfTc = np.load(cfg.strPathMdl)

print('---------Consider only traing pRF time courses')
# derive logical for training/test runs
lgcTrnTst = np.ones(np.sum(cfg.vecRunLngth), dtype=bool)
lgcTrnTst[np.cumsum(cfg.vecRunLngth)[cfg.varTestRun-1]:np.cumsum(
         cfg.vecRunLngth)[cfg.varTestRun]] = False
# consider only training runs
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

# make sure that the predictors are demeaned
aryPrfTc = np.subtract(aryPrfTc, np.mean(aryPrfTc, axis=4)[
    :, :, :, :, None])
# make sure that the data is demeaned
aryFunc = np.subtract(aryFunc, np.mean(aryFunc, axis=1)[:, None])

# %%
print('------Find best betas and R2 values')
# prepare list for results
lstRes = [None] * cfg.varPar
# Empty list for processes:
lstPrcs = [None] * cfg.varPar
# divide this ary in parts and put parts in list
lstBstIndices = np.array_split(aryBstIndices, cfg.varPar)
# prepare iterator for crossvalidation
kf = KFold(n_splits=len(cfg.lstNiiFls)-1)
kf.split(aryPrfTc)
# put functional data into list
lstFunc = np.array_split(aryFunc, cfg.varPar)
# delete arrays from memory
del(aryRes)
del(aryBstMdls)
del(aryFunc)
# Create a queue to put the results in:
queOut = mp.Queue()

for idxPrc in range(0, cfg.varPar):
    lstPrcs[idxPrc] = mp.Process(target=getBetas,
                                 args=(idxPrc, aryPrfTc,
                                       lstAllMdlInd,
                                       lstFunc[idxPrc],
                                       lstBstIndices[idxPrc],
                                       kf,
                                       queOut)
                                 )
    # Daemon (kills processes when exiting):
    lstPrcs[idxPrc].Daemon = True

# Start processes:
for idxPrc in range(0, cfg.varPar):
    lstPrcs[idxPrc].start()

# Collect results from queue:
for idxPrc in range(0, cfg.varPar):
    lstRes[idxPrc] = queOut.get(True)

# Join processes:
for idxPrc in range(0, cfg.varPar):
    lstPrcs[idxPrc].join()

# Put output into correct order:
lstRes = sorted(lstRes)

# Concatenate output vectors
aryBstBetasTrn = np.zeros((0, cfg.varNumMtDrctn, kf.get_n_splits()))
aryBstBetasTst = np.zeros((0, cfg.varNumMtDrctn, kf.get_n_splits()))
aryBstTvalsTrn = np.zeros((0, 4, kf.get_n_splits()))
aryBstTvalsTst = np.zeros((0, 4, kf.get_n_splits()))
for idxRes in range(0, cfg.varPar):
    aryBstBetasTrn = np.concatenate((aryBstBetasTrn, lstRes[idxRes][1]),
                                    axis=0)
    aryBstBetasTst = np.concatenate((aryBstBetasTst, lstRes[idxRes][2]),
                                    axis=0)
    aryBstTvalsTrn = np.concatenate((aryBstTvalsTrn, lstRes[idxRes][3]),
                                    axis=0)
    aryBstTvalsTst = np.concatenate((aryBstTvalsTst, lstRes[idxRes][4]),
                                    axis=0)


# %% save results
np.save(cfg.strPathOut + '_aryXvalBetasTrn.npy', aryBstBetasTrn)
np.save(cfg.strPathOut + '_aryXvalBetasTst.npy', aryBstBetasTst)
np.save(cfg.strPathOut + '_aryXvalTvalsTrn.npy', aryBstTvalsTrn)
np.save(cfg.strPathOut + '_aryXvalTvalsTst.npy', aryBstTvalsTst)
