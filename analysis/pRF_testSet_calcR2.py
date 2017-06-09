# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 12:16:46 2017

@author: marian
"""
import sys
import os
import numpy as np
import nibabel as nb
import multiprocessing as mp
from pRF_calcR2_getBetas import getBetas
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

# get beta weights for best models
aryBstTrainBetas = np.load(cfg.strPathOut + '_aryBstTrainBetas.npy',)

# get data for test run
niiFunc = nb.load(os.path.join(cfg.strPathNiiFunc,
                               cfg.lstNiiFls[cfg.varTestRun]))
# Load the data into memory:
aryFunc = niiFunc.get_data()
# mask the data
aryFunc = aryFunc[aryMask, :]

print('------Load prediction time courses')
# load
aryPrfTc = np.load(cfg.strPathMdl)
print('---------Consider only test pRF time courses')
# Consider only the test runs
lgcPrfTc = np.array(np.split(np.arange(np.sum(cfg.vecRunLngth)),
                             np.cumsum(cfg.vecRunLngth)[:-1]))
lgcPrfTc = np.hstack(
    lgcPrfTc[np.arange(len(cfg.lstNiiFls)) == cfg.varTestRun])
aryPrfTc = aryPrfTc[..., lgcPrfTc]

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
lstBetaRes = [None] * cfg.varPar
# Empty list for processes:
lstPrcs = [None] * cfg.varPar
# get an array that shows best x, y, sigma for every voxel
aryBstMdls = np.array([aryRes[:, 0], aryRes[:, 1], aryRes[:, 2]]).T
# divide this ary in parts and put parts in list
lstBstMdls = np.array_split(aryBstMdls, cfg.varPar)
# put best training betas in list
lstBstTrainBetas = np.array_split(aryBstTrainBetas, cfg.varPar)
# put functional data into list
lstFunc = np.array_split(aryFunc, cfg.varPar)
# delete arrays from memory
del(aryRes)
del(aryBstMdls)
del(aryBstTrainBetas)
del(aryFunc)
# Create a queue to put the results in:
queOut = mp.Queue()

for idxPrc in range(0, cfg.varPar):
    lstPrcs[idxPrc] = mp.Process(target=getBetas,
                                 args=(idxPrc, vecMdlXpos, vecMdlYpos,
                                       vecMdlSd, aryPrfTc, lstFunc[idxPrc],
                                       lstBstMdls[idxPrc],
                                       lstBstTrainBetas[idxPrc],
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

# %% save results
np.save(cfg.strPathOut + '_aryBstTestR2.npy', aryBstR2)
np.save(cfg.strPathOut + '_aryBstTestBetas.npy', aryBstBetas)

# Save R2 as nii::
aryOut = np.zeros((niiMask.shape ))
aryOut[aryMask] = aryBstR2
niiOut = nb.Nifti1Image(aryOut,
                        niiMask.affine,
                        header=niiMask.header
                        )
strTmp = (cfg.strPathOut + '_aryBstTestR2.nii')
nb.save(niiOut, strTmp)

