# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 17:02:01 2016

@author: marian
"""

from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
import os
import sys
import itertools
import numpy as np
import pRF_config as cfg
from pRF_mdlCrt import loadPng, loadPrsOrd, cnvlPwBoxCarFn

# add parent path
strParentPath = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, strParentPath)

# how many times should the display be split along the x and y axis?
nsplits = 8

# %% Create ary that shows which aperture was presented for every time point

# *** Load PNGs
aryCond = loadPng(cfg.varNumVol, cfg.tplPngSize, cfg.strPathPng)

# *** Load presentation order of motion directions
aryPresOrd = loadPrsOrd(cfg.vecRunLngth, cfg.strPathPresOrd, cfg.vecVslStim)

# *** if lgcAoM, reduce motion directions from 8 to 4
if cfg.lgcAoM:
    print('------Reduce motion directions from 8 to 4')
    aryPresOrd[aryPresOrd == 5] = 1
    aryPresOrd[aryPresOrd == 6] = 2
    aryPresOrd[aryPresOrd == 7] = 3
    aryPresOrd[aryPresOrd == 8] = 4

# %%  Create array that parcellates stimulus display into different parts
# to work as a mask for psychopy, the matrix size must be square

xpix, ypix = cfg.tplPngSize

# define parcels along x and y-axis
xsplitIdx = np.split(np.arange(xpix), nsplits)
ysplitIdx = np.split(np.arange(ypix), nsplits)
# create empy numpy arrays
mskQuadrant = np.empty([xpix, ypix, nsplits**2], dtype='bool')
# combine conditions
iterables = [np.arange(nsplits).astype(int),
             np.arange(nsplits).astype(int)]
Conditions = list(itertools.product(*iterables))
for i, (idx1, idx2) in enumerate(Conditions):
    msk = np.zeros((xpix, ypix))
    msk[xsplitIdx[idx1].astype(int)[..., None],
        ysplitIdx[idx2].astype(int)[None, ...]] = True
    mskQuadrant[:, :, i] = np.copy(msk)

# %% Create array that indicates for every parcellation of visual space and
# condition (i.e. motion direction) by how much it was stimulated at which
# time point

aryBoxCar = np.empty((mskQuadrant.shape[-1], cfg.varNumMtDrctn,
                      aryCond.shape[-1]), dtype='float32')
# walk through parcels
for indParcel in range(mskQuadrant.shape[-1]):
    # walk through condition
    for indCond, cond in enumerate(np.unique(aryPresOrd)[1:]):
        # get msk for this parcellation
        msk = mskQuadrant[..., indParcel]
        # calculate overlap msk and presented stimuli
        aryTcAllCond = np.mean(aryCond[msk, :], axis=0)
        # set all time points that do not belong to current condition to zero
        aryTcAllCond[aryPresOrd != cond] = 0
        # put predictor time course into array
        aryBoxCar[indParcel, indCond, :] = np.copy(aryTcAllCond)
# remove entries that contain zeros in all conditions (ax1) and all tps (ax2)
aryBoxCar = aryBoxCar[~np.all(aryBoxCar == 0, axis=(1,2)), :, :]

#%% Convolve
aryBoxCarConv = cnvlPwBoxCarFn(aryBoxCar,
                               cfg.varNumVol,
                               cfg.varTr,
                               cfg.tplPngSize,
                               cfg.varNumMtDrctn,
                               cfg.switchHrfSet,
                               cfg.varPar,
                               cfg.lgcOldSchoolHrf,
                               )

#%% Save
np.save(cfg.strPathOut+'_ExtractedFeatures', aryBoxCarConv)
