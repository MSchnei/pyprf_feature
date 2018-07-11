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

# Number of fMRI volumes and png files to load:
varNumVol = 1032

# Intensity cutoff value for fMRI time series. Voxels with a mean intensity
# lower than the value specified here are not included in the pRF model finding
# (this speeds up the calculation, and, more importatnly, avoids division by
# zero):
varIntCtf = -100.0

# Number of processes to run in parallel:
varPar = 8

aperture = 'mskCircleBar'

# Parent path to functional data
strPathNiiFunc = '/media/sf_D_DRIVE/MotionLocaliser/Simulation2p0/Apertures/pRF_model_tc/' + aperture + '/simResp_xval_2.npy'

# Output basename:
strPathOut = '/media/sf_D_DRIVE/MotionLocaliser/Simulation2p0/FitResults/' + aperture + '/simResp_xval_2'

# Use cython (i.e. compiled code) for faster performance? (Requires cython to
# be installed.)
lgcCython = False

# Create pRF time course models?
lgcCrteMdl = False

# reduce presented motion direction from 8 to 4?
lgcAoM = True

# length of the runs that were done
vecRunLngth = [172, 172, 172, 172, 172, 172]

# cross validate?
lgcXval = True

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
    varNumXval = 6  # set nr of xvalidations, equal to nr of runs

if lgcCrteMdl:
    # If we create new pRF time course models, the following parameters have to
    # be provided:

    # visual stimuli that were used for this run (if everything is well 1,2,3 )
    vecVslStim = [1, 2, 3, 4, 5, 6]

    # Basename of the filenames that have the presentation orders saved
    strPathPresOrd = '/media/sf_D_DRIVE/MotionLocaliser/Simulation2p0/Conditions/Conditions_run0'

    # Size of png files (pixel*pixel):
    tplPngSize = (128, 128)

    # Basename of the 'binary stimulus files'. The files need to be in png
    # format and number in the order of their presentation during the
    # experiment.
    strPathPng = '/media/sf_D_DRIVE/MotionLocaliser/Simulation2p0/Apertures/PNGs/' + aperture + '/Ima_'

    # Output path for pRF time course models file (without file extension):
    strPathMdl = '/media/sf_D_DRIVE/MotionLocaliser/Simulation2p0/FitResults/pRF_model_mtn_tc' + aperture + strBasis

else:
    # provide number of motion directions
    varNumMtDrctn = 5 * switchHrfSet
    # If we use existing pRF time course models, the path to the respective
    # file has to be provided (including file extension, i.e. '*.npy'):
    strPathMdl = '/media/sf_D_DRIVE/MotionLocaliser/Simulation2p0/FitResults/pRF_model_mtn_tc' + aperture + strBasis + '.npy'
