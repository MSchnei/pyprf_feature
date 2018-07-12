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
import nibabel as nb
import multiprocessing as mp

from pyprf_feature.analysis.load_config import load_config
from pyprf_feature.analysis.utils_general import cls_set_config
from pyprf_feature.analysis.model_creation_main import model_creation
from pyprf_feature.analysis.model_creation_utils import crt_mdl_prms
from pyprf_feature.analysis.prepare import prep_models, prep_func

###### DEBUGGING ###############
#strCsvCnfg = "/home/marian/Documents/Git/pyprf_feature/pyprf_feature/analysis/config_custom.csv"
#lgcTest = False
################################

def pyprf(strCsvCnfg, lgcTest=False):  #noqa
    """
    Main function for pRF mapping.

    Parameters
    ----------
    strCsvCnfg : str
        Absolute file path of config file.
    lgcTest : Boolean
        Whether this is a test (pytest). If yes, absolute path of pyprf libary
        will be prepended to config file paths.
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

    aryPrfTc = model_creation(dicCnfg)
    # *************************************************************************

    # *************************************************************************
    # *** Preprocessing

    # The model time courses will be preprocessed such that they are smoothed
    # (temporally) with same factor as the data and that they will be z-scored:
    aryPrfTc = prep_models(aryPrfTc, varSdSmthTmp=cfg.varSdSmthTmp)

    # The functional data will be masked and demeaned:
    aryLgcMsk, aryLgcVar, hdrMsk, aryAff, aryFunc, tplNiiShp = prep_func(
        cfg.strPathNiiMask, cfg.lstPathNiiFunc, varAvgThr=-100.)
    # *************************************************************************

    # *************************************************************************
    # *** Find pRF models for voxel time courses

    print('------Find pRF models for voxel time courses')

    # Number of voxels for which pRF finding will be performed:
    varNumVoxInc = aryFunc.shape[0]

    # Number of feautures
    varNumFtr = aryPrfTc.shape[1]

    print('---------Number of voxels on which pRF finding will be performed: '
          + str(varNumVoxInc))

    print('---------Preparing parallel pRF model finding')

    # For the GPU version, we need to set down the parallelisation to 1 now,
    # because no separate CPU threads are to be created. We may still use CPU
    # parallelisation for preprocessing, which is why the parallelisation
    # factor is only reduced now, not earlier.
    if cfg.strVersion == 'gpu':
        cfg.varPar = 1

    # Make sure that if gpu fitting is used, the number of cross-validations is
    # set to 1, not higher
    if cfg.strVersion == 'gpu':
        strErrMsg = 'Stopping program. ' + \
            'Cross-validation on GPU is currently not supported. ' + \
            'Set varNumXval equal to 1 in csv file in order to continue. '
        assert cfg.varNumXval == 1, strErrMsg

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

    # check whether we need to crossvalidate
    if np.greater(cfg.varNumXval, 1):
        cfg.lgcXval = True
    elif np.equal(cfg.varNumXval, 1):
        cfg.lgcXval = False
    else:
        print("Please set number of crossvalidation folds (numXval) to 1 \
              (meaning no cross validation) or greater than 1 (meaning number \
              of cross validation folds)")

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
                                               queOut)
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
                                               queOut)
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

    # Concatenate output vectors:
    aryBstXpos = np.zeros(0)
    aryBstYpos = np.zeros(0)
    aryBstSd = np.zeros(0)
    aryBstR2 = np.zeros(0)
    aryBstBts = np.zeros((0, varNumFtr))

    for idxRes in range(0, cfg.varPar):
        aryBstXpos = np.append(aryBstXpos, lstPrfRes[idxRes][1])
        aryBstYpos = np.append(aryBstYpos, lstPrfRes[idxRes][2])
        aryBstSd = np.append(aryBstSd, lstPrfRes[idxRes][3])
        aryBstR2 = np.append(aryBstR2, lstPrfRes[idxRes][4])
        aryBstBts = np.concatenate((aryBstBts, lstPrfRes[idxRes][5]), axis=0)

    # Concatenate output arrays for R2 maps (saved for every run):
    if np.greater(cfg.varNumXval, 1):
        lstBstR2Single = []
        for idxRes in range(0, cfg.varPar):
            lstBstR2Single.append(lstPrfRes[idxRes][5])
        aryBstR2Single = np.concatenate(lstBstR2Single, axis=0)
        del(lstBstR2Single)

    # Delete unneeded large objects:
    del(lstPrfRes)

    # Put results form pRF finding into array (they originally needed to be
    # saved in a list due to parallelisation). Voxels were selected for pRF
    # model finding in two stages: First, a mask was applied. Second, voxels
    # with low variance were removed. Voxels are put back into the original
    # format accordingly.

    # Number of voxels that were included in the mask:
    varNumVoxMsk = np.sum(aryLgcMsk)

    # Array for pRF finding results, of the form aryPrfRes[voxel-count, 0:3],
    # where the 2nd dimension contains the parameters of the best-fitting pRF
    # model for the voxel, in the order (0) pRF-x-pos, (1) pRF-y-pos, (2)
    # pRF-SD, (3) pRF-R2. At this step, only the voxels included in the mask
    # are represented.
    aryPrfRes01 = np.zeros((varNumVoxMsk, 6), dtype=np.float32)

    # Place voxels based on low-variance exlusion:
    aryPrfRes01[aryLgcVar, 0] = aryBstXpos
    aryPrfRes01[aryLgcVar, 1] = aryBstYpos
    aryPrfRes01[aryLgcVar, 2] = aryBstSd
    aryPrfRes01[aryLgcVar, 3] = aryBstR2
    # Calculate polar angle map:
    aryPrfRes01[aryLgcVar, 4] = np.arctan2(aryBstYpos,
                                           aryBstXpos)
    # Calculate eccentricity map (r = sqrt( x^2 + y^2 ) ):
    aryPrfRes01[aryLgcVar, 5] = np.sqrt(np.add(np.square(aryBstXpos),
                                               np.square(aryBstYpos)))

    # Total number of voxels:
    varNumVoxTlt = (tplNiiShp[0] * tplNiiShp[1] * tplNiiShp[2])

    # Place voxels based on mask-exclusion:
    aryPrfRes02 = np.zeros((varNumVoxTlt, aryPrfRes01.shape[-1]),
                           dtype=np.float32)

    for indDim in range(aryPrfRes01.shape[-1]):
        aryPrfRes02[aryLgcMsk, indDim] = aryPrfRes01[:, indDim]

    # Reshape pRF finding results into original image dimensions:
    aryPrfRes = np.reshape(aryPrfRes02,
                           [tplNiiShp[0],
                            tplNiiShp[1],
                            tplNiiShp[2],
                            6])

    del(aryPrfRes01)
    del(aryPrfRes02)

    # List with name suffices of output images:
    lstNiiNames = ['_x_pos',
                   '_y_pos',
                   '_SD',
                   '_R2',
                   '_polar_angle',
                   '_eccentricity']

    print('---------Exporting results')

    # Save nii results:
    for idxOut in range(0, 6):
        # Create nii object for results:
        niiOut = nb.Nifti1Image(aryPrfRes[..., idxOut],
                                aryAff,
                                header=hdrMsk
                                )
        # Save nii:
        strTmp = (cfg.strPathOut + lstNiiNames[idxOut] + '.nii.gz')
        nb.save(niiOut, strTmp)

    # *************************************************************************

    # *************************************************************************
    # Save beta parameter estimates for every feauture:

    # Place voxels based on low-variance exlusion:
    aryPrfRes01 = np.zeros((varNumVoxMsk, varNumFtr),
                           dtype=np.float32)

    for indDim in range(varNumFtr):
        aryPrfRes01[aryLgcVar, indDim] = aryBstBts[:, indDim]

    # Place voxels based on mask-exclusion:
    aryPrfRes02 = np.zeros((varNumVoxTlt, varNumFtr),
                           dtype=np.float32)
    for indDim in range(varNumFtr):
        aryPrfRes02[aryLgcMsk, indDim] = aryPrfRes01[:, indDim]

    # Reshape pRF finding results into original image dimensions:
    aryBstBtsOut = np.reshape(aryPrfRes02,
                              [tplNiiShp[0],
                               tplNiiShp[1],
                               tplNiiShp[2],
                               varNumFtr])

    del(aryPrfRes01)
    del(aryPrfRes02)

    # adjust header
    hdrMsk.set_data_shape(aryBstBtsOut.shape)

    # Create nii object for results:
    niiOut = nb.Nifti1Image(aryBstBtsOut,
                            aryAff,
                            header=hdrMsk
                            )
    # Save nii:
    strTmp = (cfg.strPathOut + '_Betas' + '.nii.gz')
    nb.save(niiOut, strTmp)

    # *************************************************************************

    # *************************************************************************
    # Save R2 maps from crossvalidation (saved for every run) as nii:
    if np.greater(cfg.varNumXval, 1):
        # truncate extremely negative R2 values
        aryBstR2Single[np.where(np.less_equal(aryBstR2Single, -1.0))] = -1.0

        # Place voxels based on low-variance exlusion:
        aryPrfRes01 = np.zeros((varNumVoxMsk, cfg.varNumXval),
                               dtype=np.float32)

        for indDim in range(cfg.varNumXval):
            aryPrfRes01[aryLgcVar, indDim] = aryBstR2Single[:, indDim]

        # Place voxels based on mask-exclusion:
        aryPrfRes02 = np.zeros((varNumVoxTlt, cfg.varNumXval),
                               dtype=np.float32)
        for indDim in range(cfg.varNumXval):
            aryPrfRes02[aryLgcMsk, indDim] = aryPrfRes01[:, indDim]

        # Reshape pRF finding results into original image dimensions:
        aryR2snglRes = np.reshape(aryPrfRes02,
                                  [tplNiiShp[0],
                                   tplNiiShp[1],
                                   tplNiiShp[2],
                                   cfg.varNumXval])

        del(aryPrfRes01)
        del(aryPrfRes02)

        # adjust header
        hdrMsk.set_data_shape(aryR2snglRes.shape)

        # Create nii object for results:
        niiOut = nb.Nifti1Image(aryR2snglRes,
                                aryAff,
                                header=hdrMsk
                                )
        # Save nii:
        strTmp = (cfg.strPathOut + '_R2_single' + '.nii.gz')
        nb.save(niiOut, strTmp)

    # *************************************************************************

    # *************************************************************************
    # *** Report time

    varTme02 = time.time()
    varTme03 = varTme02 - varTme01
    print('---Elapsed time: ' + str(varTme03) + ' s')
    print('---Done.')
    # *************************************************************************
