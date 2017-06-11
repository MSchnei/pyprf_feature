#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 13:48:50 2017

@author: Marian
"""

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
from sklearn.preprocessing import scale
from pRF_calcR2_getBetas import getBetas
from pRF_filtering import funcSmthTmp

# this needs to go later
import pRF_config_test as cfg



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

# %%
varThreshR2 = 0.2
# use R2 values as a mask
vecR2 = aryRes[:, 3]
lgcR2 = [vecR2 >= varThreshR2][0]

# exclude voxels with low R2, exclude last column (since it is weight 4 static)
aryBetas = aryBstTrainBetas[lgcR2, :-1]

# demean the betas
aryBetasDemean = np.subtract(aryBetas, np.mean(aryBetas,axis=0)[None, :])
            
aryBetasArgMax = np.argmax(aryBetas, axis=1)


aryBetasScaled = np.divide(aryBetas, np.max(aryBetas,axis=1)[:, None])

aryBetasScaled = scale(aryBetas, axis=1, with_mean=False, with_std=True,
                       copy=True)

test1 = aryBetasDemean[:30, :].T
test2 = aryBetasScaled[:30, :].T