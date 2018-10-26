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
from pyprf_feature.analysis.load_config import load_config
from pyprf_feature.analysis.utils_general import (cls_set_config, export_nii,
                                                  load_res_prm)
from pyprf_feature.analysis.prepare import prep_func, prep_models
from pyprf_feature.analysis.model_creation_utils import (crt_mdl_prms,
                                                         fnd_unq_rws)

###### DEBUGGING ###############
#strCsvCnfg = "/media/sf_D_DRIVE/MotDepPrf/Analysis/S02/03_motLoc/pRF_results/S02_config_MotLoc_feature.csv"
#lgcTest = False
#lstRat=None
#lgcMdlRsp=True
################################


def save_tc_to_nii(strCsvCnfg, lgcTest=False, lstRat=None, lgcMdlRsp=False,
                   strPathHrf=None):
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

    # Some voxels were excluded because they did not have sufficient mean
    # and/or variance - exclude their initial parameters, too
    # Get inclusion mask and nii header
    aryLgcMsk, aryLgcVar, hdrMsk, aryAff, aryFunc, tplNiiShp = prep_func(
        cfg.strPathNiiMask, cfg.lstPathNiiFunc, varAvgThr=-100)
    # Apply inclusion mask
    aryIntGssPrm = aryIntGssPrm[aryLgcVar, :]
    aryBetas = aryBetas[aryLgcVar, :]

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
    aryFitTc = np.zeros((aryIntGssPrm.shape[0],
                         aryPrfTc.shape[-1]), dtype=np.float32)
    # If desired, initiliaze array that will collect model responses underlying
    # the fittedt time course
    if lgcMdlRsp:
        aryFitMdlRsp = np.zeros((aryIntGssPrm.shape[0],
                                 aryMdlRsp.shape[-1]), dtype=np.float32)

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
        # Get logical index for the model number
        # This can only be 1 index, so we directly get 1st entry of array
        lgcMdl = np.where(np.isclose(aryMdlParams, vecPrm,
                                     atol=1e-04).all(axis=1))[0][0]
        if lgcMdl is None:
            print('---No model found')
        # Mark those voxels that were visited
        vecVxlTst[lgcVxl] += 1

        # Get model time courses
        aryMdlTc = aryPrfTc[lgcMdl, :, :]
        # Get beta parameter estimates
        aryWeights = aryBetas[lgcVxl, :]
        # Combine model time courses and weights to yield fitted time course
        aryFitTc[lgcVxl, :] = np.dot(aryWeights, aryMdlTc)
        if lgcMdlRsp:
            # If desired also save the model responses that won
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
        # List with name suffices of output images:
        lstNiiNames = ['_FitMdlRsp']

        # Create full path names from nii file names and output path
        lstNiiNames = [cfg.strPathOut + strNii + '.nii.gz' for strNii in
                       lstNiiNames]

        # export aryFitMdlRsp as a single 4D nii file
        print('---Save fitted model responses')
        export_nii(aryFitMdlRsp, lstNiiNames, aryLgcMsk, aryLgcVar, tplNiiShp,
                   aryAff, hdrMsk, outFormat='4D')
        print('------Done.')
