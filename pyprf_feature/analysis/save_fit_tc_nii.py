# -*- coding: utf-8 -*-
"""Saving empirical and fitted time courses to nii file format"""

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

import os
import numpy as np
import nibabel as nb
from pyprf_feature.analysis.load_config import load_config
from pyprf_feature.analysis.utils_general import (cls_set_config, export_nii,
                                                  load_res_prm)
from pyprf_feature.analysis.prepare import prep_func, prep_models
from pyprf_feature.analysis.model_creation_utils import (crt_mdl_prms,
                                                         fnd_unq_rws)

###### DEBUGGING ###############
#strCsvCnfg = "/media/sf_D_DRIVE/MotionQuartet/Analysis/P3/Prf/Fitting/pRF_results/Testing/P3_config_NoM_sptSmooth_tmpSmooth.csv"
#lgcTest = False
#lstRat = None  # [1.5, 1.8, 2.1]
#lgcMdlRsp = True
#strPathHrf = None
################################


def save_tc_to_nii(strCsvCnfg, lgcTest=False, lstRat=None, lgcMdlRsp=False,
                   strPathHrf=None, lgcSaveRam=False):
    """
    Save empirical and fitted time courses to nii file format.

    Parameters
    ----------
    strCsvCnfg : str
        Absolute file path of config file used for pRF fitting.
    lgcTest : boolean
        Whether this is a test (pytest). If yes, absolute path of pyprf libary
        will be prepended to config file paths.
    lstRat : None or list
        Ratio of size of center to size of suppressive surround.
    lgcMdlRsp : boolean
        Should the aperture responses for the winner model also be saved?
    strPathHrf : str or None:
        Path to npy file with custom hrf parameters. If None, defaults
        parameters were used.
    lgcSaveRam : boolean
        Whether to also save a nii file that uses little RAM.

    Notes
    -----
    This function does not return any arguments but, instead, saves nii files
    to disk.

    """

    # %% Load configuration settings that were used for fitting

    # Load config parameters from csv file into dictionary:
    dicCnfg = load_config(strCsvCnfg, lgcTest=lgcTest)

    # Load config parameters from dictionary into namespace:
    cfg = cls_set_config(dicCnfg)

    # if fitting was done with custom hrf, make sure to retrieve results with
    # '_hrf' appendix
    if strPathHrf is not None:
        cfg.strPathOut = cfg.strPathOut + '_hrf'

    # If suppressive surround flag is on, make sure to retrieve results with
    # '_supsur' appendix
    if lstRat is not None:
        cfg.strPathOut = cfg.strPathOut + '_supsur'
        cfg.strPathMdl = cfg.strPathMdl + '_supsur'
        # Append 0.0 as the first entry, which is the key for fitting without
        # surround (only centre)
        lstRat.insert(0, 0.0)

    # %% Load previous pRF fitting results

    # Derive paths to the x, y, sigma winner parameters from pyprf_feature
    lstWnrPrm = [cfg.strPathOut + '_x_pos.nii.gz',
                 cfg.strPathOut + '_y_pos.nii.gz',
                 cfg.strPathOut + '_SD.nii.gz']

    # Check if fitting has been performed, i.e. whether parameter files exist
    # Throw error message if they do not exist.
    errorMsg = 'Files that should have resulted from fitting do not exist. \
                \nPlease perform pRF fitting first, calling  e.g.: \
                \npyprf_feature -config /path/to/my_config_file.csv'
    assert os.path.isfile(lstWnrPrm[0]), errorMsg
    assert os.path.isfile(lstWnrPrm[1]), errorMsg
    assert os.path.isfile(lstWnrPrm[2]), errorMsg

    # Load the x, y, sigma winner parameters from pyprf_feature
    aryIntGssPrm = load_res_prm(lstWnrPrm,
                                lstFlsMsk=[cfg.strPathNiiMask])[0][0]

    # Load beta parameters estimates, aka weights for time courses
    lstPathBeta = [cfg.strPathOut + '_Betas.nii.gz']
    aryBetas = load_res_prm(lstPathBeta, lstFlsMsk=[cfg.strPathNiiMask])[0][0]
    assert os.path.isfile(lstPathBeta[0]), errorMsg

    # Load ratio image, if fitting was obtained with suppressive surround
    if lstRat is not None:
        lstPathRatio = [cfg.strPathOut + '_Ratios.nii.gz']
        aryRatio = load_res_prm(lstPathRatio,
                                lstFlsMsk=[cfg.strPathNiiMask])[0][0]
        assert os.path.isfile(lstPathRatio[0]), errorMsg

    # Some voxels were excluded because they did not have sufficient mean
    # and/or variance - exclude their initial parameters, too
    # Get inclusion mask and nii header
    aryLgcMsk, aryLgcVar, hdrMsk, aryAff, aryFunc, tplNiiShp = prep_func(
        cfg.strPathNiiMask, cfg.lstPathNiiFunc, varAvgThr=-100)
    # Apply inclusion mask
    aryIntGssPrm = aryIntGssPrm[aryLgcVar, :]
    aryBetas = aryBetas[aryLgcVar, :]
    if lstRat is not None:
        aryRatio = aryRatio[aryLgcVar, :]

    # Get array with model parameters that were fitted on a grid
    # [x positions, y positions, sigmas]
    aryMdlParams = crt_mdl_prms((int(cfg.varVslSpcSzeX),
                                int(cfg.varVslSpcSzeY)), cfg.varNum1,
                                cfg.varExtXmin, cfg.varExtXmax, cfg.varNum2,
                                cfg.varExtYmin, cfg.varExtYmax,
                                cfg.varNumPrfSizes, cfg.varPrfStdMin,
                                cfg.varPrfStdMax, kwUnt='deg',
                                kwCrd=cfg.strKwCrd)

    # Get corresponding pRF model time courses
    aryPrfTc = np.load(cfg.strPathMdl + '.npy')

    # The model time courses will be preprocessed such that they are smoothed
    # (temporally) with same factor as the data and that they will be z-scored:
    aryPrfTc = prep_models(aryPrfTc, varSdSmthTmp=cfg.varSdSmthTmp)

    if lgcMdlRsp:
        aryMdlRsp = np.load(cfg.strPathMdl + '_mdlRsp.npy')

    # %% Derive fitted time course models for all voxels

    # Initialize array that will collect the fitted time courses
    aryFitTc = np.zeros((aryFunc.shape), dtype=np.float32)
    # If desired, initiliaze array that will collect model responses underlying
    # the fitted time course
    if lgcMdlRsp:
        if lstRat is not None:
            aryFitMdlRsp = np.zeros((aryIntGssPrm.shape[0], aryMdlRsp.shape[1],
                                     aryMdlRsp.shape[3]),
                                    dtype=np.float32)
        else:
            aryFitMdlRsp = np.zeros((aryIntGssPrm.shape[0],
                                     aryMdlRsp.shape[1]), dtype=np.float32)

    # create vector that allows to check whether every voxel is visited
    # exactly once
    vecVxlTst = np.zeros(aryIntGssPrm.shape[0])

    # Find unique rows of fitted model parameters
    aryUnqRows, aryUnqInd = fnd_unq_rws(aryIntGssPrm, return_index=False,
                                        return_inverse=True)

    # Loop over all best-fitting model parameter combinations found
    print('---Assign models to voxels')
    for indRow, vecPrm in enumerate(aryUnqRows):
        # Get logical for voxels for which this prm combi was the best
        lgcVxl = [aryUnqInd == indRow][0]
        if np.all(np.invert(lgcVxl)):
            print('---No voxel found')
        # Mark those voxels that were visited
        vecVxlTst[lgcVxl] += 1

        # Get logical index for the model number
        # This can only be 1 index, so we directly get 1st entry of array
        lgcMdl = np.where(np.isclose(aryMdlParams, vecPrm,
                                     atol=0.01).all(axis=1))[0][0]
        # Tell user if no model was found
        if lgcMdl is None:
            print('---No model found')

        # Get model time courses
        aryMdlTc = aryPrfTc[lgcMdl, ...]
        # Get beta parameter estimates
        aryWeights = aryBetas[lgcVxl, :]

        # If fitting was done with surround suppression, find ratios for voxels
        # and the indices of these ratios in lstRat
        if lstRat is not None:
            aryVxlRatio = aryRatio[lgcVxl, :]
            indRat = [ind for ind, rat1 in enumerate(lstRat) for rat2 in
                      aryVxlRatio[:, 0] if np.isclose(rat1, rat2)]
            indVxl = range(len(indRat))

        # Combine model time courses and weights to yield fitted time course
        if lstRat is not None:
            aryFitTcTmp = np.tensordot(aryWeights, aryMdlTc, axes=([1], [0]))
            aryFitTc[lgcVxl, :] = aryFitTcTmp[indVxl, indRat, :]
        else:
            aryFitTc[lgcVxl, :] = np.dot(aryWeights, aryMdlTc)

        # If desired by user, also save the model responses per voxels
        if lgcMdlRsp:
            # If desired also save the model responses that won
            if lstRat is not None:
                aryFitMdlRsp[lgcVxl, :] = aryMdlRsp[lgcMdl, :, indRat, :]
            else:
                aryFitMdlRsp[lgcVxl, :] = aryMdlRsp[lgcMdl, :]

    # check that every voxel was visited exactly once
    errMsg = 'At least one voxel visited more than once for tc recreation'
    assert len(vecVxlTst) == np.sum(vecVxlTst), errMsg

    # %% Export preprocessed voxel time courses as nii

    # List with name suffices of output images:
    lstNiiNames = ['_EmpTc']

    # Create full path names from nii file names and output path
    lstNiiNames = [cfg.strPathOut + strNii + '.nii.gz' for strNii in
                   lstNiiNames]

    # export aryFunc as a single 4D nii file
    print('---Save empirical time courses')
    export_nii(aryFunc, lstNiiNames, aryLgcMsk, aryLgcVar, tplNiiShp,
               aryAff, hdrMsk, outFormat='4D')
    print('------Done.')

    # If desired by user, also save RAM-saving version of nii
    if lgcSaveRam:
        strPthRamOut = cfg.strPathOut + '_EmpTc_saveRAM' + '.nii.gz'
        imgNii = nb.Nifti1Image(np.expand_dims(np.expand_dims(aryFunc, axis=1),
                                               axis=1),
                                affine=np.eye(4))
        nb.save(imgNii, strPthRamOut)

    # %% Export fitted time courses and, if desired, model responses as nii

    # List with name suffices of output images:
    lstNiiNames = ['_FitTc']

    # Create full path names from nii file names and output path
    lstNiiNames = [cfg.strPathOut + strNii + '.nii.gz' for strNii in
                   lstNiiNames]

    # export aryFitTc as a single 4D nii file
    print('---Save fitted time courses')
    export_nii(aryFitTc, lstNiiNames, aryLgcMsk, aryLgcVar, tplNiiShp,
               aryAff, hdrMsk, outFormat='4D')
    print('------Done.')

    if lgcMdlRsp:

        # Create full path name
        strNpyName = cfg.strPathOut + '_FitMdlRsp' + '.npy'

        # Save aryFitMdlRsp as npy file
        print('---Save fitted model responses')
        np.save(strNpyName, aryFitMdlRsp)
        print('------Done.')

    # If desired by user, also save RAM-saving version of nii
    if lgcSaveRam:
        strPthRamOut = cfg.strPathOut + '_FitTc_saveRAM' + '.nii.gz'
        imgNii = nb.Nifti1Image(np.expand_dims(np.expand_dims(aryFitTc,
                                                              axis=1),
                                               axis=1),
                                affine=np.eye(4))
        nb.save(imgNii, strPthRamOut)
