#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Wed Jul 11 18:51:28 2018

@author: marian
"""

import os
import numpy as np

# %% set parameters

# set path to folder with the spatial masks used during stimulation
strPthPrnt = '/media/sf_D_DRIVE/MotionLocaliser/UsedPsychoPyScripts/P02/Masks'
# set output path name
strPthOut = '/home/marian/Documents/Testing/pyprf_testing/expInfo/sptInfo'
# set name of with spatial masks
strFileName = 'mskCircleBar.npy'

# set factors for downsampling
factorX = 8
factorY = 8

# %% load and downsample

arySptExpInf = np.load(os.path.join(strPthPrnt, strFileName))
arySptExpInf = arySptExpInf[::factorX, ::factorY, :]

# save
strAryPth = os.path.join(strPthOut, 'arySptExpInf')
np.save(strAryPth, arySptExpInf)
