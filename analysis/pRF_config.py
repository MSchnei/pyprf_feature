# -*- coding: utf-8 -*-
"""Define pRF finding parameters here."""

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


# Number of x-positions to model:
varNumX = 25
# Number of y-positions to model:
varNumY = 25
# Number of pRF sizes to model:
varNumPrfSizes = 22

# Extend of visual space from centre of the screen (i.e. from the fixation
# point) [degrees of visual angle]:
varExtXmin = -12.00
varExtXmax = 12.00
varExtYmin = -12.00
varExtYmax = 12.00

# Maximum and minimum pRF model size (standard deviation of 2D Gaussian)
# [degrees of visual angle]:
varPrfStdMin = 1.0
varPrfStdMax = 22.0

# Volume TR of input data [s]:
varTr = 3.0
# Voxel resolution of the fMRI data [mm]:
varVoxRes = 0.8

# Extent of temporal smoothing that has been applied to fMRI data
# [standard deviation of the Gaussian kernel, in seconds]:
# the same smotthing will be applied to pRF time course models
# [set 0 if not desired]
varSdSmthTmp = 0

# Number of processes to run in parallel:
varPar = 10

# Size of high-resolution visual space model in which the pRF models are
# created (x- and y-dimension). The x and y dimensions specified here need to
# be the same integer multiple of the number of x- and y-positions to model, as
# specified above. In other words, if the the resolution in x-direction of the
# visual space model is ten times that of varNumX, the resolution in
# y-direction also has to be ten times varNumY. The order is: first x, then y.
tplVslSpcHighSze = (200, 200)

# Parent path to functional data
strPathNiiFunc = '/media/sf_D_DRIVE/MotionLocaliser/Analysis/P02/05_SpatSmoothDemean'
# list of nii files in parent directory (all nii files together need to have
# same number of volumes as there are PNGs):
lstNiiFls = ['demean_rafunc01_hpf.nii',
             'demean_rafunc02_hpf.nii',
             'demean_rafunc03_hpf.nii',
             'demean_rafunc04_hpf.nii',
             'demean_rafunc05_hpf.nii',
             'demean_rafunc06_hpf.nii',
             'demean_rafunc07_hpf.nii',
             ]
# which run should be hold out for testing? [python index strating from 0]
varTestRun = 6

# Path of mask (to restrict pRF model finding):
strPathNiiMask = '/media/sf_D_DRIVE/MotionLocaliser/Analysis/P02/Struct/mask.nii'

# Output basename:
strPathOut = '/media/sf_D_DRIVE/MotionLocaliser/Analysis/P02/FitResults/Compare/MotionNoXvalAoM'

# Which version to use for pRF finding. 'numpy' or 'cython' for pRF finding on
# CPU, 'gpu' for using GPU.
strVersion = 'gpu'

# Create pRF time course models?
lgcCrteMdl = False

# reduce presented motion direction from 8 to 4?
lgcAoM = True

# length of the runs that were done
vecRunLngth = [172] * len(lstNiiFls)

# Number of fMRI volumes and png files to load:
varNumVol = sum(vecRunLngth)

# cross validate?
lgcXval = False

# set which set of hrf functions should be used
lgcOldSchoolHrf = True

if lgcOldSchoolHrf:  # use legacy hrf function
    strBasis = '_oldSch'
    # use only canonical hrf function
    switchHrfSet = 1
else:  # use hrf basis
    # decide of how many functions the basis set should consist:
    # 1: canonical hrf function
    # 2: canonical hrf function and 1st tmp derivative
    # 3: canonical hrf function, 1st tmp and spatial derivative
    switchHrfSet = 3
    strBasis = '_bsSet' + str(switchHrfSet)

if lgcXval:
    varNumXval = len(lstNiiFls)  # set nr of xvalidations, equal to nr of runs

# For time course model creation, the following parameters have to
# be provided:

# visual stimuli that were used for this run (if everything is well 1,2,3, asf)
vecVslStim = [1, 2, 3, 4, 5, 6, 7]

# Basename of the filenames that have the presentation orders saved
strPathPresOrd = '/media/sf_D_DRIVE/MotionLocaliser/UsedPsychoPyScripts/P02/Conditions/Conditions_run0'

# Size of png files (pixel*pixel):
tplPngSize = (128, 128)

# Basename of the 'binary stimulus files'. The files need to be in png
# format and number in the order of their presentation during the
# experiment.
strPathPng = '/media/sf_D_DRIVE/MotionLocaliser/Analysis/P02/PNGs/Ima_'

# If we use existing pRF time course models, the path to the respective
# file has to be provided (including file extension, i.e. '*.npy'):
strPathMdl = '/media/sf_D_DRIVE/MotionLocaliser/Analysis/P02/FitResults/Compare/pRF_model_tc' + strBasis + '.npy'

if lgcAoM:
    # number of motion directions
    varNumMtDrctn = 5 * switchHrfSet
else:
    # number of motion directions
    varNumMtDrctn = 9 * switchHrfSet
