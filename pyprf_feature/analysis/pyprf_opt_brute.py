# -*- coding: utf-8 -*-
"""Optimize given pRF paramaters using brute-force grid search."""

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
from pyprf_feature.analysis.utils_general import (cls_set_config, load_res_prm,
                                                  export_nii, joinRes,
                                                  map_pol_to_crt,
                                                  find_near_pol_ang)
from pyprf_feature.analysis.model_creation_opt import model_creation_opt
from pyprf_feature.analysis.model_creation_utils import rmp_deg_pixel_xys
from pyprf_feature.analysis.prepare import prep_models, prep_func


###### DEBUGGING ###############
#strCsvCnfg = "/home/marian/Documents/Testing/pyprf_feature_devel/S02_config_motDepPrf_flck_smooth.csv"
#class Object(object):
#    pass
#objNspc = Object()
#objNspc.strPthPrior = "/media/sf_D_DRIVE/MotDepPrf/Analysis/S02/03_motLoc/pRF_results/pRF_results_tmpSmth"
#objNspc.varNumOpt1 = 90
#objNspc.varNumOpt2 = 64
#objNspc.varNumOpt3 = 1
#objNspc.lgcRstrCentre = False
#lgcTest = False
################################

def pyprf_opt_brute(strCsvCnfg, objNspc, lgcTest=False):  #noqa
    """
    Function for optimizing given pRF paramaters using brute-force grid search.

    Parameters
    ----------
    strCsvCnfg : str
        Absolute file path of config file.
    objNspc : object
        Name space from command line arguments.
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
    # *** Preprocessing

    # The functional data will be masked and demeaned:
    aryLgcMsk, aryLgcVar, hdrMsk, aryAff, aryFunc, tplNiiShp = prep_func(
        cfg.strPathNiiMask, cfg.lstPathNiiFunc)

    # set the precision of the header to np.float32 so that the prf results
    # will be saved in this precision later
    hdrMsk.set_data_dtype(np.float32)

    print('---Number of voxels included in analysis: ' +
          str(np.sum(aryLgcVar)))

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

    # check whether we need to crossvalidate
    if np.greater(cfg.varNumXval, 1):
        cfg.lgcXval = True
    elif np.equal(cfg.varNumXval, 1):
        cfg.lgcXval = False
    strErrMsg = 'Stopping program. ' + \
        'Set numXval (number of crossvalidation folds) to 1 or higher'
    assert np.greater_equal(cfg.varNumXval, 1), strErrMsg
    # *************************************************************************

    # *************************************************************************
    # Load previous pRF fitting results
    print('---String to prior results provided by user:')
    print(objNspc.strPthPrior)

    # Load the x, y, sigma winner parameters from pyprf_feature
    lstWnrPrm = [objNspc.strPthPrior + '_x_pos.nii',
                 objNspc.strPthPrior + '_y_pos.nii',
                 objNspc.strPthPrior + '_SD.nii',
                 objNspc.strPthPrior + '_eccentricity.nii']
    lstPrmInt, objHdr, aryAff = load_res_prm(lstWnrPrm,
                                             lstFlsMsk=[cfg.strPathNiiMask])
    # Convert list to array
    assert len(lstPrmInt) == 1
    aryIntGssPrm = lstPrmInt[0]
    del(lstPrmInt)

    # Some voxels were excluded because they did not have sufficient mean
    # and/or variance - exclude their nitial parameters, too
    aryIntGssPrm = aryIntGssPrm[aryLgcVar, :]

    # *************************************************************************

    # *************************************************************************
    # *** Sort voxels by polar angle/previous parameters

    # Calculate the polar angles that were found in independent localiser
    aryPlrAng = np.arctan2(aryIntGssPrm[:, 1], aryIntGssPrm[:, 0])

    # Calculate the unique polar angles that are expected from grid search
    aryUnqPlrAng = np.linspace(0.0, 2*np.pi, objNspc.varNumOpt2,
                               endpoint=False)

    # Expected polar angle values are range from 0 to 2*pi, while
    # the calculated angle values will range from -pi to pi
    # Thus, bring empirical values from range -pi, pi to range 0, 2pi
    aryPlrAng = (aryPlrAng + 2 * np.pi) % (2 * np.pi)

    # For every empirically found polar angle get the index of the nearest
    # theoretically expected polar angle, this is to offset small imprecisions
    aryUnqPlrAngInd, aryDstPlrAng = find_near_pol_ang(aryPlrAng, aryUnqPlrAng)

    # Make sure that the maximum distance from a found polar angle to a grid
    # point is smaller than the distance between two neighbor grid points
    assert np.max(aryDstPlrAng) < np.divide(2*np.pi, objNspc.varNumOpt2)

    # Update unique polar angles such that it contains only the ones which
    # were found in data
    aryUnqPlrAng = aryUnqPlrAng[np.unique(aryUnqPlrAngInd)]
    # Update indices
    aryUnqPlrAngInd, aryDstPlrAng = find_near_pol_ang(aryPlrAng, aryUnqPlrAng)

    # Get logical arrays that index voxels with particular polar angle
    lstLgcUnqPlrAng = []
    for indPlrAng in range(len(aryUnqPlrAng)):
        lstLgcUnqPlrAng.append([aryUnqPlrAngInd == indPlrAng][0])

    print('---Number of radial position options provided by user: ' +
          str(objNspc.varNumOpt1))
    print('---Number of angular position options provided by user: ' +
          str(objNspc.varNumOpt2))
    print('---Number of unique polar angles found in prior estimates: ' +
          str(len(aryUnqPlrAng)))
    print('---Maximum displacement in radial direction that is allowed: ' +
          str(objNspc.varNumOpt3))
    print('---Fitted modelled are restricted to stimulated area: ' +
          str(objNspc.lgcRstrCentre))

    # *************************************************************************
    # *** Perform prf fitting

    # Create array for collecting winner parameters
    aryBstXpos = np.zeros((aryPlrAng.shape[0]))
    aryBstYpos = np.zeros((aryPlrAng.shape[0]))
    aryBstSd = np.zeros((aryPlrAng.shape[0]))
    aryBstR2 = np.zeros((aryPlrAng.shape[0]))
    aryBstBts = np.zeros((aryPlrAng.shape[0], 1))
    if np.greater(cfg.varNumXval, 1):
        aryBstR2Single = np.zeros((aryPlrAng.shape[0],
                                   len(cfg.lstPathNiiFunc)))

    # loop over all found instances of polar angle/previous parameters
    for indPlrAng in range(len(aryUnqPlrAng)):

        print('------Polar angle number ' + str(indPlrAng+1) + ' out of ' +
              str(len(aryUnqPlrAng)))

        # get the polar angle for the current voxel batch
        varPlrAng = np.array(aryUnqPlrAng[indPlrAng])

        # get logical array to index voxels with this particular polar angle
        lgcUnqPlrAng = lstLgcUnqPlrAng[indPlrAng]

        # get prior eccentricities for current voxel batch
        vecPrrEcc = aryIntGssPrm[lgcUnqPlrAng, 3]

        print('---------Number of voxels of this polar angle: ' +
              str(np.sum(lgcUnqPlrAng)))

        # *********************************************************************

        # *********************************************************************
        # *** Create time course models for this particular polar angle

        # Vector with the radial position:
        vecRad = np.linspace(0.0, cfg.varExtXmax, objNspc.varNumOpt1,
                             endpoint=True)

        # Get all possible combinations on the grid, using matrix indexing ij
        # of output
        aryRad, aryTht = np.meshgrid(vecRad, varPlrAng, indexing='ij')

        # Flatten arrays to be able to combine them with meshgrid
        vecRad = aryRad.flatten()
        vecTht = aryTht.flatten()

        # Convert from polar to cartesian
        vecX, vecY = map_pol_to_crt(vecTht, vecRad)

        # Vector with standard deviations pRF models (in degree of vis angle):
        vecPrfSd = np.linspace(cfg.varPrfStdMin, cfg.varPrfStdMax,
                               cfg.varNumPrfSizes, endpoint=True)

        # Create model parameters
        varNumMdls = len(vecX) * len(vecPrfSd)
        aryMdlParams = np.zeros((varNumMdls, 3), dtype=np.float32)

        varCntMdlPrms = 0

        # Loop through x-positions:
        for idxXY in range(0, len(vecX)):

            # Loop through standard deviations (of Gaussian pRF models):
            for idxSd in range(0, len(vecPrfSd)):

                # Place index and parameters in array:
                aryMdlParams[varCntMdlPrms, 0] = vecX[idxXY]
                aryMdlParams[varCntMdlPrms, 1] = vecY[idxXY]
                aryMdlParams[varCntMdlPrms, 2] = vecPrfSd[idxSd]

                # Increment parameter index:
                varCntMdlPrms += 1

        # Convert winner parameters from degrees of visual angle to pixel
        vecIntX, vecIntY, vecIntSd = rmp_deg_pixel_xys(aryMdlParams[:, 0],
                                                       aryMdlParams[:, 1],
                                                       aryMdlParams[:, 2],
                                                       cfg.tplVslSpcSze,
                                                       cfg.varExtXmin,
                                                       cfg.varExtXmax,
                                                       cfg.varExtYmin,
                                                       cfg.varExtYmax)

        aryMdlParamsPxl = np.column_stack((vecIntX, vecIntY, vecIntSd))

        if objNspc.lgcRstrCentre:
            # Calculate the areas that were stimulated during the experiment
            arySptExpInf = np.load(cfg.strSptExpInf)
            arySptExpInf = np.rot90(arySptExpInf, k=3)
            aryStimArea = np.sum(arySptExpInf, axis=-1).astype(np.bool)

            # Get logical to exclude models with pRF centre outside stim area
            lgcMdlInc = aryStimArea[aryMdlParamsPxl[:, 0].astype(np.int32),
                                    aryMdlParamsPxl[:, 1].astype(np.int32)]
            # Exclude models with prf center outside stimulated area
            aryMdlParams = aryMdlParams[lgcMdlInc, :]
            aryMdlParamsPxl = aryMdlParamsPxl[lgcMdlInc, :]

        # Create model time courses
        aryPrfTc = model_creation_opt(dicCnfg, aryMdlParamsPxl)

        # The model time courses will be preprocessed such that they are
        # smoothed (temporally) with same factor as the data and that they will
        # be z-scored:
        aryPrfTc = prep_models(aryPrfTc, varSdSmthTmp=cfg.varSdSmthTmp,
                               lgcPrint=False)

        # *********************************************************************
        # *** Create logical to restrict model fitting in radial direction

        # Calculate eccentricity of currently tested model parameters
        vecMdlEcc = np.sqrt(np.add(np.square(aryMdlParams[:, 0]),
                                   np.square(aryMdlParams[:, 1])))
        # Compare model eccentricity against prior eccentricity
        vecPrrEccGrd, vecMdlEccGrd = np.meshgrid(vecPrrEcc, vecMdlEcc,
                                                 indexing='ij')
        # Consider allowed eccentricity shift as specified by user
        lgcRstr = np.logical_and(np.less_equal(vecMdlEccGrd,
                                               np.add(vecPrrEccGrd,
                                                      objNspc.varNumOpt3)),
                                 np.greater(vecMdlEccGrd,
                                            np.subtract(vecPrrEccGrd,
                                                        objNspc.varNumOpt3)))

        # *********************************************************************
        # *** Check for every voxel there is at least one model being tried

        # Is there at least 1 model for each voxel?
        lgcMdlPerVxl = np.greater(np.sum(lgcRstr, axis=1), 0)
        print('---------Number of voxels fitted: ' + str(np.sum(lgcMdlPerVxl)))

        # Those voxels for which no model would be tried, for example because
        # the pRF parameters estimated in the prior were outside the stimulated
        # area, are escluded from model fitting by setting their logical False
        lgcUnqPlrAng[lgcUnqPlrAng] = lgcMdlPerVxl

        # We need to update the index table for restricting model fitting
        lgcRstr = lgcRstr[lgcMdlPerVxl, :]

        # *********************************************************************
        # *** Find best model for voxels with this particular polar angle

        # Only perform the fitting if there are voxels with models to optimize
        if np.any(lgcUnqPlrAng):
            # Empty list for results (parameters of best fitting pRF model):
            lstPrfRes = [None] * cfg.varPar

            # Empty list for processes:
            lstPrcs = [None] * cfg.varPar

            # Create a queue to put the results in:
            queOut = mp.Queue()

            # Put logical for model restriction in list
            lstRst = np.array_split(lgcRstr, cfg.varPar)
            del(lgcRstr)

            # Create list with chunks of func data for parallel processes:
            lstFunc = np.array_split(aryFunc[lgcUnqPlrAng, :], cfg.varPar)

            # CPU version (using numpy or cython for pRF finding):
            if ((cfg.strVersion == 'numpy') or (cfg.strVersion == 'cython')):

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
                                                 kwargs={'lgcRstr':
                                                         lstRst[idxPrc],
                                                         'lgcPrint': False},
                                                 )
                    # Daemon (kills processes when exiting):
                    lstPrcs[idxPrc].Daemon = True

            # GPU version (using tensorflow for pRF finding):
            elif cfg.strVersion == 'gpu':

                # Create processes:
                for idxPrc in range(0, cfg.varPar):
                    lstPrcs[idxPrc] = mp.Process(target=find_prf_gpu,
                                                 args=(idxPrc,
                                                       aryMdlParams,
                                                       lstFunc[idxPrc],
                                                       aryPrfTc,
                                                       queOut),
                                                 kwargs={'lgcRstr':
                                                         lstRst[idxPrc],
                                                         'lgcPrint': False},
                                                 )
                    # Daemon (kills processes when exiting):
                    lstPrcs[idxPrc].Daemon = True

            # Start processes:
            for idxPrc in range(0, cfg.varPar):
                lstPrcs[idxPrc].start()

            # Delete reference to list with function data (the data continues
            # to exists in child process):
            del(lstFunc)

            # Collect results from queue:
            for idxPrc in range(0, cfg.varPar):
                lstPrfRes[idxPrc] = queOut.get(True)

            # Join processes:
            for idxPrc in range(0, cfg.varPar):
                lstPrcs[idxPrc].join()

            # *****************************************************************

            # *****************************************************************
            # *** Prepare pRF finding results for export

            # Put output into correct order:
            lstPrfRes = sorted(lstPrfRes)

            # collect results from parallelization
            aryBstTmpXpos = joinRes(lstPrfRes, cfg.varPar, 1, inFormat='1D')
            aryBstTmpYpos = joinRes(lstPrfRes, cfg.varPar, 2, inFormat='1D')
            aryBstTmpSd = joinRes(lstPrfRes, cfg.varPar, 3, inFormat='1D')
            aryBstTmpR2 = joinRes(lstPrfRes, cfg.varPar, 4, inFormat='1D')
            aryBstTmpBts = joinRes(lstPrfRes, cfg.varPar, 5, inFormat='2D')
            if np.greater(cfg.varNumXval, 1):
                aryTmpBstR2Single = joinRes(lstPrfRes, cfg.varPar, 6,
                                            inFormat='2D')
            # Delete unneeded large objects:
            del(lstPrfRes)

            # *****************************************************************

            # *****************************************************************
            # Put findings for voxels with specific polar angle into ary with
            # result for all voxels
            aryBstXpos[lgcUnqPlrAng] = aryBstTmpXpos
            aryBstYpos[lgcUnqPlrAng] = aryBstTmpYpos
            aryBstSd[lgcUnqPlrAng] = aryBstTmpSd
            aryBstR2[lgcUnqPlrAng] = aryBstTmpR2
            aryBstBts[lgcUnqPlrAng, :] = aryBstTmpBts
            if np.greater(cfg.varNumXval, 1):
                aryBstR2Single[lgcUnqPlrAng, :] = aryTmpBstR2Single

            # *****************************************************************

    # *************************************************************************
    # Calculate polar angle map:
    aryPlrAng = np.arctan2(aryBstYpos, aryBstXpos)
    # Calculate eccentricity map (r = sqrt( x^2 + y^2 ) ):
    aryEcc = np.sqrt(np.add(np.square(aryBstXpos),
                            np.square(aryBstYpos)))

    # It is possible that after optimization the pRF has moved to location 0, 0
    # In this cases, the polar angle parameter is arbitrary and will be
    # assigned either 0 or pi. To preserve smoothness of the map, assign the
    # initial polar angle value from independent localiser
    lgcMvdOrgn = np.logical_and(aryBstXpos == 0.0, aryBstYpos == 0.0)
    lgcMvdOrgn = np.logical_and(lgcMvdOrgn, aryBstSd > 0)
    aryIntPlrAng = np.arctan2(aryIntGssPrm[:, 1], aryIntGssPrm[:, 0])
    aryPlrAng[lgcMvdOrgn] = np.copy(aryIntPlrAng[lgcMvdOrgn])

    # *************************************************************************

    # *************************************************************************
    # Export each map of best parameters as a 3D nii file

    print('---------Exporting results')

    # Xoncatenate all the best voxel maps
    aryBstMaps = np.stack([aryBstXpos, aryBstYpos, aryBstSd, aryBstR2,
                           aryPlrAng, aryEcc], axis=1)

    # List with name suffices of output images:
    lstNiiNames = ['_x_pos_opt',
                   '_y_pos_opt',
                   '_SD_opt',
                   '_R2_opt',
                   '_polar_angle_opt',
                   '_eccentricity_opt']

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
    lstNiiNames = ['_Betas_opt']

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
        lstNiiNames = ['_R2_single_opt']

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
