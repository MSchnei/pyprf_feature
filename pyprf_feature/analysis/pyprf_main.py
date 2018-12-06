# -*- coding: utf-8 -*-
"""Find best fitting model time courses for population receptive fields."""

# Part of pyprf_feature library
# Copyright (C) 2018  Marian Schneider, Ingo Marquardt
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.

import time
import numpy as np
import multiprocessing as mp

from pyprf_feature.analysis.load_config import load_config
from pyprf_feature.analysis.utils_general import (cls_set_config, export_nii,
                                                  joinRes)
from pyprf_feature.analysis.model_creation_main import model_creation
from pyprf_feature.analysis.model_creation_utils import crt_mdl_prms
from pyprf_feature.analysis.prepare import prep_models, prep_func

###### DEBUGGING ###############
#strCsvCnfg = "/home/marian/Documents/Testing/pyprf_feature_devel/control/S02_config_motDepPrf_flck_smooth_inw.csv"
#lgcTest = False
#varRat=None
################################

def pyprf(strCsvCnfg, lgcTest=False, varRat=None, strPathHrf=None):
    """
    Main function for pRF mapping.

    Parameters
    ----------
    strCsvCnfg : str
        Absolute file path of config file.
    lgcTest : Boolean
        Whether this is a test (pytest). If yes, absolute path of pyprf libary
        will be prepended to config file paths.
    varRat : float, default None
        Ratio of size suppressive surround to size of center pRF
    strPathHrf : str or None:
        Path to npy file with custom hrf parameters. If None, default
        parameters will be used.

    """
    # *************************************************************************
    # *** Check time
    print('---pRF analysis')
    varTme01 = time.time()
    # *************************************************************************

    # *************************************************************************
    # *** Preparations

    # Load config parameters from csv file into dictionary:
    dicCnfg = load_config(strCsvCnfg, lgcTest=lgcTest)

    # Load config parameters from dictionary into namespace:
    cfg = cls_set_config(dicCnfg)

    # Conditional imports:
    if cfg.strVersion == 'gpu':
        from pyprf_feature.analysis.find_prf_gpu import find_prf_gpu
    if ((cfg.strVersion == 'cython') or (cfg.strVersion == 'numpy')):
        from pyprf_feature.analysis.find_prf_cpu import find_prf_cpu

    # Convert preprocessing parameters (for temporal smoothing)
    # from SI units (i.e. [s]) into units of data array (volumes):
    cfg.varSdSmthTmp = np.divide(cfg.varSdSmthTmp, cfg.varTr)
    # *************************************************************************

    # *************************************************************************
    # *** Create or load pRF time course models

    # Create model time courses. Also return logical for inclusion of model
    # parameters which will be needed later when we create model parameters
    # in degree.
    aryPrfTc = model_creation(dicCnfg, varRat=varRat, strPathHrf=strPathHrf)

    # Deduce the number of features from the pRF time course models array
    cfg.varNumFtr = aryPrfTc.shape[1]

    # *************************************************************************

    # *************************************************************************
    # *** Preprocessing

    # The model time courses will be preprocessed such that they are smoothed
    # (temporally) with same factor as the data and that they will be z-scored:
    aryPrfTc = prep_models(aryPrfTc, varSdSmthTmp=cfg.varSdSmthTmp)

    # The functional data will be masked and demeaned:
    aryLgcMsk, aryLgcVar, hdrMsk, aryAff, aryFunc, tplNiiShp = prep_func(
        cfg.strPathNiiMask, cfg.lstPathNiiFunc, varAvgThr=-100)

    # set the precision of the header to np.float32 so that the prf results
    # will be saved in this precision later
    hdrMsk.set_data_dtype(np.float32)
    # *************************************************************************

    # *************************************************************************
    # *** Checks

    # Make sure that if gpu fitting is used, the number of cross-validations is
    # set to 1, not higher
    if cfg.strVersion == 'gpu':
        strErrMsg = 'Stopping program. ' + \
            'Cross-validation on GPU is currently not supported. ' + \
            'Set varNumXval equal to 1 in csv file in order to continue. '
        assert cfg.varNumXval == 1, strErrMsg

    # For the GPU version, we need to set down the parallelisation to 1 now,
    # because no separate CPU threads are to be created. We may still use CPU
    # parallelisation for preprocessing, which is why the parallelisation
    # factor is only reduced now, not earlier.
    if cfg.strVersion == 'gpu':
        cfg.varPar = 1

    # Make sure that if cython is used, the number of features is 1 or 2,
    # not higher
    if cfg.strVersion == 'cython':
        strErrMsg = 'Stopping program. ' + \
            'Cython is not supported for more features than 1. ' + \
            'Set strVersion equal \'numpy\'.'
        assert cfg.varNumFtr in [1, 2], strErrMsg

    # Check whether we need to crossvalidate
    if np.greater(cfg.varNumXval, 1):
        cfg.lgcXval = True
    elif np.equal(cfg.varNumXval, 1):
        cfg.lgcXval = False
    strErrMsg = 'Stopping program. ' + \
        'Set numXval (number of crossvalidation folds) to 1 or higher'
    assert np.greater_equal(cfg.varNumXval, 1), strErrMsg

    # *************************************************************************
    # *** Find pRF models for voxel time courses

    print('------Find pRF models for voxel time courses')

    # Number of voxels for which pRF finding will be performed:
    cfg.varNumVoxInc = aryFunc.shape[0]

    print('---------Number of voxels on which pRF finding will be performed: '
          + str(cfg.varNumVoxInc))
    print('---------Number of features pRF finding will be performed with: '
          + str(cfg.varNumFtr))

    print('---------Preparing parallel pRF model finding')

    # Get array with all possible model parameter combination:
    # [x positions, y positions, sigmas]
    aryMdlParams = crt_mdl_prms((int(cfg.varVslSpcSzeX),
                                 int(cfg.varVslSpcSzeY)), cfg.varNum1,
                                cfg.varExtXmin, cfg.varExtXmax, cfg.varNum2,
                                cfg.varExtYmin, cfg.varExtYmax,
                                cfg.varNumPrfSizes, cfg.varPrfStdMin,
                                cfg.varPrfStdMax, kwUnt='deg',
                                kwCrd=cfg.strKwCrd)

    # Empty list for results (parameters of best fitting pRF model):
    lstPrfRes = [None] * cfg.varPar

    # Empty list for processes:
    lstPrcs = [None] * cfg.varPar

    # Create a queue to put the results in:
    queOut = mp.Queue()

    # Create list with chunks of functional data for the parallel processes:
    lstFunc = np.array_split(aryFunc, cfg.varPar)
    # We don't need the original array with the functional data anymore:
    del(aryFunc)

    # Prepare dictionary to pass as kwargs to find_prf_cpu
    dctKw = {'lgcRstr': None,
             'lgcPrint': True}

    # CPU version (using numpy or cython for pRF finding):
    if ((cfg.strVersion == 'numpy') or (cfg.strVersion == 'cython')):

        print('---------pRF finding on CPU')

        print('---------Creating parallel processes')

        # Create processes:
        for idxPrc in range(0, cfg.varPar):
            lstPrcs[idxPrc] = mp.Process(target=find_prf_cpu,
                                         args=(idxPrc,
                                               lstFunc[idxPrc],
                                               aryPrfTc,
                                               aryMdlParams,
                                               cfg.strVersion,
                                               cfg.lgcXval,
                                               cfg.varNumXval,
                                               queOut),
                                         kwargs=dctKw,
                                         )
            # Daemon (kills processes when exiting):
            lstPrcs[idxPrc].Daemon = True

    # GPU version (using tensorflow for pRF finding):
    elif cfg.strVersion == 'gpu':

        print('---------pRF finding on GPU')

        # Create processes:
        for idxPrc in range(0, cfg.varPar):
            lstPrcs[idxPrc] = mp.Process(target=find_prf_gpu,
                                         args=(idxPrc,
                                               aryMdlParams,
                                               lstFunc[idxPrc],
                                               aryPrfTc,
                                               queOut),
                                         kwargs=dctKw,
                                         )
            # Daemon (kills processes when exiting):
            lstPrcs[idxPrc].Daemon = True

    # Start processes:
    for idxPrc in range(0, cfg.varPar):
        lstPrcs[idxPrc].start()

    # Delete reference to list with function data (the data continues to exists
    # in child process):
    del(lstFunc)

    # Collect results from queue:
    for idxPrc in range(0, cfg.varPar):
        lstPrfRes[idxPrc] = queOut.get(True)

    # Join processes:
    for idxPrc in range(0, cfg.varPar):
        lstPrcs[idxPrc].join()

    # *************************************************************************

    # *************************************************************************
    # *** Prepare pRF finding results for export
    print('---------Prepare pRF finding results for export')

    # Put output into correct order:
    lstPrfRes = sorted(lstPrfRes)

    # collect results from parallelization
    aryBstXpos = joinRes(lstPrfRes, cfg.varPar, 1, inFormat='1D')
    aryBstYpos = joinRes(lstPrfRes, cfg.varPar, 2, inFormat='1D')
    aryBstSd = joinRes(lstPrfRes, cfg.varPar, 3, inFormat='1D')
    aryBstR2 = joinRes(lstPrfRes, cfg.varPar, 4, inFormat='1D')
    aryBstBts = joinRes(lstPrfRes, cfg.varPar, 5, inFormat='2D')
    if np.greater(cfg.varNumXval, 1):
        aryBstR2Single = joinRes(lstPrfRes, cfg.varPar, 6, inFormat='2D')

    # Delete unneeded large objects:
    del(lstPrfRes)

    # *************************************************************************

    # *************************************************************************
    # Calculate polar angle map:
    aryPlrAng = np.arctan2(aryBstYpos, aryBstXpos)
    # Calculate eccentricity map (r = sqrt( x^2 + y^2 ) ):
    aryEcc = np.sqrt(np.add(np.square(aryBstXpos),
                            np.square(aryBstYpos)))
    # *************************************************************************

    # *************************************************************************
    # Export each map of best parameters as a 3D nii file

    print('---------Exporting results')

    # Append 'hrf' to cfg.strPathOut, if fitting was done with custom hrf
    if strPathHrf is not None:
        cfg.strPathOut = cfg.strPathOut + '_hrf'

    # Xoncatenate all the best voxel maps
    aryBstMaps = np.stack([aryBstXpos, aryBstYpos, aryBstSd, aryBstR2,
                           aryPlrAng, aryEcc], axis=1)

    # List with name suffices of output images:
    lstNiiNames = ['_x_pos',
                   '_y_pos',
                   '_SD',
                   '_R2',
                   '_polar_angle',
                   '_eccentricity']

    # Append ratio to nii file name, if fitting was done with sup surround
    if varRat is not None:
        lstNiiNames = [strNii + '_' + str(varRat) for strNii in lstNiiNames]

    # Create full path names from nii file names and output path
    lstNiiNames = [cfg.strPathOut + strNii + '.nii.gz' for strNii in
                   lstNiiNames]

    # export map results as seperate 3D nii files
    export_nii(aryBstMaps, lstNiiNames, aryLgcMsk, aryLgcVar, tplNiiShp,
               aryAff, hdrMsk, outFormat='3D')

    # *************************************************************************

    # *************************************************************************
    # Save beta parameter estimates for every feature:

    # List with name suffices of output images:
    lstNiiNames = ['_Betas']

    # Append ratio to nii file name, if fitting was done with sup surround
    if varRat is not None:
        lstNiiNames = [strNii + '_' + str(varRat) for strNii in lstNiiNames]

    # Create full path names from nii file names and output path
    lstNiiNames = [cfg.strPathOut + strNii + '.nii.gz' for strNii in
                   lstNiiNames]

    # export beta parameter as a single 4D nii file
    export_nii(aryBstBts, lstNiiNames, aryLgcMsk, aryLgcVar, tplNiiShp,
               aryAff, hdrMsk, outFormat='4D')

    # *************************************************************************

    # *************************************************************************
    # Save R2 maps from crossvalidation (saved for every run) as nii:

    if np.greater(cfg.varNumXval, 1):

        # truncate extremely negative R2 values
        aryBstR2Single[np.where(np.less_equal(aryBstR2Single, -1.0))] = -1.0

        # List with name suffices of output images:
        lstNiiNames = ['_R2_single']

        # Append ratio to nii file name, if fitting was done with sup surround
        if varRat is not None:
            lstNiiNames = [strNii + '_' + str(varRat) for strNii in
                           lstNiiNames]

        # Create full path names from nii file names and output path
        lstNiiNames = [cfg.strPathOut + strNii + '.nii.gz' for strNii in
                       lstNiiNames]

        # export R2 maps as a single 4D nii file
        export_nii(aryBstR2Single, lstNiiNames, aryLgcMsk, aryLgcVar,
                   tplNiiShp, aryAff, hdrMsk, outFormat='4D')

    # *************************************************************************

    # *************************************************************************
    # *** Report time

    varTme02 = time.time()
    varTme03 = varTme02 - varTme01
    print('---Elapsed time: ' + str(varTme03) + ' s')
    print('---Done.')
    # *************************************************************************
