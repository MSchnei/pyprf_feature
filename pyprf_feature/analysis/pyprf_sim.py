# -*- coding: utf-8 -*-
"""Simulate pRF response given pRF parameters and stimulus apertures"""

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
from pyprf_feature.analysis.model_creation_utils import (rmp_deg_pixel_xys,
                                                         crt_mdl_rsp,
                                                         crt_prf_tc)
from pyprf_feature.analysis.prepare import prep_func

### DEBUGGING ###
#strPrior = '/media/sf_D_DRIVE/MotDepPrf/Presentation/figures/Figure_perception/vificov_pngs/fig_perception_sim_prf.csv'
#strStmApr = '/media/sf_D_DRIVE/MotDepPrf/Presentation/figures/Figure_perception/vificov_pngs/aprt_stim.npy'


def pyprf_sim(strPrior, strStmApr, lgcNoise=False, lgcRtnNrl=True,
              lstRat=None, lgcTest=False):
    """
    Simulate pRF response given pRF parameters and stimulus apertures.

    Parameters
    ----------
    strPrior : str
        Absolute file path of config file used for pRF fitting.
    strStmApr : str
        Absolute file path to stimulus aperture used in in-silico experiment.
    lgcNoise : boolean
        Should noise be added to the simulated pRF time course. By default, no
        noise is added.
    lgcRtnNrl : boolean
        Should neural time course, unconvolved with hrf, be returned as well?
    lstRat : None or list
        Ratio of size of center to size of suppressive surround.
    lgcTest : boolean
        Whether this is a test (pytest). If yes, absolute path of pyprf libary
        will be prepended to config file paths.

    Notes
    -----
    [1] This function does not return any arguments but, instead, saves nii
        filex to disk.
    [2] strStmApr should be a path to a npy file that contains a 3D numpy
        array. This arrays consists of binary images in boolean array from that
        represent the stimulus aperture. Images are stacked along last axis.

    """

    # %% Load configuration settings that were used for fitting

    # Load config parameters from csv file into dictionary:
    dicCnfg = load_config(strPrior, lgcTest=lgcTest)

    # Load config parameters from dictionary into namespace:
    cfg = cls_set_config(dicCnfg)

    # If suppressive surround flag is on, make sure to retrieve results from
    # that fitting
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

    # Also load suppresive surround params if suppressive surround flag was on
    if lstRat is not None:
        # Load beta parameters estimates, aka weights, this is later needed to
        # scale responses of the center wrt to the surround
        lstPathBeta = [cfg.strPathOut + '_Betas.nii.gz']
        aryBetas = load_res_prm(lstPathBeta,
                                lstFlsMsk=[cfg.strPathNiiMask])[0][0]
        # Load ratio of prf sizes
        lstPathRat = [cfg.strPathOut + '_Ratios.nii.gz']
        aryRat = load_res_prm(lstPathRat, lstFlsMsk=[cfg.strPathNiiMask])[0][0]

    # Some voxels were excluded because they did not have sufficient mean
    # and/or variance - exclude their initial parameters, too
    # Get inclusion mask and nii header
    aryLgcMsk, aryLgcVar, hdrMsk, aryAff, _, tplNiiShp = prep_func(
        cfg.strPathNiiMask, cfg.lstPathNiiFunc, varAvgThr=100.)
    # Apply inclusion mask
    aryIntGssPrm = aryIntGssPrm[aryLgcVar, :]
    if lstRat is not None:
        aryBetas = aryBetas[aryLgcVar, :]
        aryRat = aryRat[aryLgcVar]

    # %% Load stimulus aperture and create model responses to stimuli

    # Load stimulus aperture
    aryStmApr = np.load(strStmApr)
    # Which dimensions does the representation have in pixel space?
    tplStmApr = aryStmApr.shape[:2]

    # Convert winner parameters from degrees of visual angle to pixel
    vecIntX, vecIntY, vecIntSd = rmp_deg_pixel_xys(aryIntGssPrm[:, 0],
                                                   aryIntGssPrm[:, 1],
                                                   aryIntGssPrm[:, 2],
                                                   tplStmApr,
                                                   cfg.varExtXmin,
                                                   cfg.varExtXmax,
                                                   cfg.varExtYmin,
                                                   cfg.varExtYmax)

    aryIntGssPrmPxl = np.column_stack((vecIntX, vecIntY, vecIntSd))

    # Create 2D Gauss model responses to spatial conditions.
    print('---Create 2D Gauss model responses to spatial conditions')
    aryMdlRsp = crt_mdl_rsp(aryStmApr, tplStmApr, aryIntGssPrmPxl, cfg.varPar)

    # If supsur flag was provided, also create responses with supsur params
    # and combine positive center response with negative surround response
    if lstRat is not None:
        aryIntGssPrmPxlSur = np.copy(aryIntGssPrmPxl)
        # Adjust pRF sizes using the ratio of pRF sizes
        aryIntGssPrmPxlSur[:, 2] = np.multiply(aryIntGssPrmPxlSur[:, 2],
                                               aryRat)
        aryMdlRspSur = crt_mdl_rsp(aryStmApr, tplStmApr, aryIntGssPrmPxlSur,
                                   cfg.varPar)
        # Now the responses of the center and the surround need to be combined
        # in a meaningful way. One way this could be done is to take the ratio
        # of gain parameters that were found when fitting (i.e. betas)
        varGainRat = np.divide(aryBetas[:, 0], aryBetas[:, 1])
        aryMdlRsp = np.subtract(aryMdlRsp,
                                np.multiply(varGainRat, aryMdlRspSur))

    # %% Convolve time courses with hrf function

    # First temporally upsamle the model response
    aryMdlRspUps = np.repeat(aryMdlRsp, cfg.varTmpOvsmpl, axis=-1)
    # Convolve with hrf function
    arySimRsp = crt_prf_tc(aryMdlRspUps, aryMdlRsp.shape[-1], cfg.varTr,
                           cfg.varTmpOvsmpl, 1, tplStmApr, cfg.varPar)
    # Squeeze simulated reponse. This step is necessary because crt_prf_tc is,
    # in principle, capable of convolving with deriavtes of canonical function
    if arySimRsp.shape[1] > 1:
        print('***WARNING: pyprf_sim expects 1 hrf function, currently***')
    arySimRsp = np.squeeze(arySimRsp)
    # Save memory by deleting upsampled time course
    del(aryMdlRspUps)

    # %% Add auto-correlated noise

    if lgcNoise:
        print('***Adding noise feature not yet implemented***')

    # %% Export simulated prf, and if desired neural, time courses as nii

    # List with name suffices of output images:
    lstNiiNames = ['_SimPrfTc']

    # Create full path names from nii file names and output path
    lstNiiNames = [cfg.strPathOut + strNii + '.nii.gz' for strNii in
                   lstNiiNames]

    # export beta parameter as a single 4D nii file
    print('---Save simulated pRF time courses')
    export_nii(arySimRsp, lstNiiNames, aryLgcMsk, aryLgcVar, tplNiiShp,
               aryAff, hdrMsk, outFormat='4D')
    print('------Done.')

    if lgcRtnNrl:

        # List with name suffices of output images:
        lstNiiNames = ['_SimNrlTc']

        # Create full path names from nii file names and output path
        lstNiiNames = [cfg.strPathOut + strNii + '.nii.gz' for strNii in
                       lstNiiNames]

        # export beta parameter as a single 4D nii file
        print('---Save simulated neural time courses')
        export_nii(aryMdlRsp, lstNiiNames, aryLgcMsk, aryLgcVar, tplNiiShp,
                   aryAff, hdrMsk, outFormat='4D')
        print('------Done.')
