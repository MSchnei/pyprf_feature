# -*- coding: utf-8 -*-
"""Define pRF finding parameters here"""

# Part of py_pRF_motion library
# Copyright (C) 2016  Ingo Marquardt
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
varNumX = 21
# Number of y-positions to model:
varNumY = 21
# Number of pRF sizes to model:
varNumPrfSizes = 35

# Extend of visual space from centre of the screen (i.e. from the fixation
# point) [degrees of visual angle]:
varExtXmin = -5.00
varExtXmax = 5.00
varExtYmin = -5.00
varExtYmax = 5.00

# Maximum and minimum pRF model size (standard deviation of 2D Gaussian)
# [degrees of visual angle]:
varPrfStdMin = 0.1
varPrfStdMax = 7.0

# Volume TR of input data [s]:
varTr = 2.832
# Voxel resolution of the fMRI data [mm]:
varVoxRes = 0.8

# Extent of temporal smoothing that has been applied to fMRI data
# [standard deviation of the Gaussian kernel, in seconds]:
# the same smotthing will be applied to pRF time course models
# [set 0 if not desired]
varSdSmthTmp = 2.832

# Number of fMRI volumes and png files to load:
varNumVol = 688

# Intensity cutoff value for fMRI time series. Voxels with a mean intensity
# lower than the value specified here are not included in the pRF model finding
# (this speeds up the calculation, and, more importatnly, avoids division by
# zero):
varIntCtf = -100.0

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
strPathNiiFunc = '/media/sf_D_DRIVE/PacMan/Analysis/P3/UndistortedMotCor/03_SmoothSpat1SmoothTmp1Demean'
# list of nii files in parent directory (all nii files together need to have
# same number of volumes as there are PNGs):
lstNiiFls = ['zs1_1func_07_hpf.nii',
             'zs1_1rprf_08_hpf.nii',
             'zs1_1rprf_09_hpf.nii',
             'zs1_1rprf_10_hpf.nii',
             ]

# Path of mask (to restrict pRF model finding):
strPathNiiMask = '/media/sf_D_DRIVE/PacMan/Analysis/P3/UndistortedMotCor/Struct/Mask.nii'

# Output basename:
strPathOut = '/media/sf_D_DRIVE/PacMan/Analysis/P3/UndistortedMotCor/FitResults/MotionNoXval'

# Use cython (i.e. compiled code) for faster performance? (Requires cython to
# be installed.)
lgcCython = False

# Create pRF time course models?
lgcCrteMdl = False

# reduce presented motion direction from 8 to 4?
lgcAoM = True

# length of the runs that were done
vecRunLngth = [172, 172, 172, 172]

# cross validate?
lgcXval = False

if lgcXval:
    varNumXval = len(lstNiiFls)  # set nr of xvalidations, equal to nr of runs

if lgcCrteMdl:
    # If we create new pRF time course models, the following parameters have to
    # be provided:

    # visual stimuli that were used for this run (if everything is well 1,2,3 )
    vecVslStim = [1, 2, 3, 4]

    # Basename of the filenames that have the presentation orders saved
    strPathPresOrd = '/media/sf_D_DRIVE/PacMan/PsychoPyScripts/Pacman_Scripts/PacMan_Pilot3_20161220/ModBasedMotLoc/Conditions/Conditions_run0'

    # Size of png files (pixel*pixel):
    tplPngSize = (128, 128)

    # Basename of the 'binary stimulus files'. The files need to be in png
    # format and number in the order of their presentation during the
    # experiment.
    strPathPng = '/media/sf_D_DRIVE/PacMan/Analysis/P3/UndistortedMotCor/PrfPngs/Ima_'

    # Output path for pRF time course models file (without file extension):
    strPathMdl = '/media/sf_D_DRIVE/PacMan/Analysis/P3/UndistortedMotCor/FitResults/pRF_model_mtn_tc'

else:
    # provide number of motion directions
    varNumMtDrctn = 5
    # If we use existing pRF time course models, the path to the respective
    # file has to be provided (including file extension, i.e. '*.npy'):
    strPathMdl = '/media/sf_D_DRIVE/PacMan/Analysis/P3/UndistortedMotCor/FitResults/pRF_model_mtn_tc.npy'
