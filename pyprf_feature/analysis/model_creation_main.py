# -*- coding: utf-8 -*-
"""pRF model creation."""

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

import numpy as np
from pyprf_feature.analysis.utils_general import cls_set_config
from pyprf_feature.analysis.model_creation_utils import (crt_mdl_prms,
                                                         crt_mdl_rsp,
                                                         crt_nrl_tc,
                                                         crt_prf_tc)


def model_creation(dicCnfg):
    """
    Create or load pRF model time courses.

    Parameters
    ----------
    dicCnfg : dict
        Dictionary containing config parameters.

    Returns
    -------
    aryPrfTc : np.array
        4D numpy array with pRF time course models, with following dimensions:
        `aryPrfTc[x-position, y-position, SD, volume]`.
    """
    # *************************************************************************
    # *** Load parameters from config file

    # Load config parameters from dictionary into namespace:
    cfg = cls_set_config(dicCnfg)
    # *************************************************************************

    if cfg.lgcCrteMdl:

        # *********************************************************************
        # *** Load spatial condition information

        print('------Load spatial condition information')

        arySptExpInf = np.load(cfg.strSptExpInf)

        # Since we assume scientific convention and orientation of images where
        # the x-axisfalls on the height and the y-axis falls on the width
        # dimension of the screen and we assume that the first dimension that
        # the user provides index x and the second y and since python is column
        # major (i.e. first indexes columns, only then rows), we need to rotate
        # arySptExpInf by 90 degrees rightward.
        arySptExpInf = np.rot90(arySptExpInf, k=3)

        # *********************************************************************

        # *********************************************************************
        # *** Load temporal condition information

        print('------Load temporal condition information')

        aryTmpExpInf = np.load(cfg.strTmpExpInf)
        # *********************************************************************

        # *********************************************************************
        # *** Create model parameter combination, for now in pixel.
        aryMdlParams = crt_mdl_prms((int(cfg.varVslSpcSzeX),
                                     int(cfg.varVslSpcSzeY)), cfg.varNum1,
                                    cfg.varExtXmin, cfg.varExtXmax,
                                    cfg.varNum2, cfg.varExtYmin,
                                    cfg.varExtYmax, cfg.varNumPrfSizes,
                                    cfg.varPrfStdMin, cfg.varPrfStdMax,
                                    kwUnt='pix', kwCrd=cfg.strKwCrd)

        # *********************************************************************

        # *********************************************************************
        # *** Create 2D Gauss model responses to spatial conditions.

        print('------Create 2D Gauss model responses to spatial conditions')

        aryMdlRsp = crt_mdl_rsp(arySptExpInf, (int(cfg.varVslSpcSzeX),
                                               int(cfg.varVslSpcSzeY)),
                                aryMdlParams, cfg.varPar)
        del(arySptExpInf)

        # *********************************************************************

        # *********************************************************************
        # *** Create neural time courses in upsampled space

        print('------Create temporally upsampled neural time courses')

        aryNrlTc = crt_nrl_tc(aryMdlRsp, aryTmpExpInf, cfg.varTr,
                              cfg.varNumVol, cfg.varTmpOvsmpl)
        del(aryTmpExpInf)
        del(aryMdlRsp)

        # *********************************************************************

        # *********************************************************************
        # *** Convolve every neural time course model with hrf function(s)

        print('------Create pRF time course models by HRF convolution')

        aryPrfTc = crt_prf_tc(aryNrlTc, cfg.varNumVol, cfg.varTr,
                              cfg.varTmpOvsmpl, cfg.switchHrfSet,
                              (int(cfg.varVslSpcSzeX), int(cfg.varVslSpcSzeY)),
                              cfg.varPar)
        del(aryNrlTc)

        # *********************************************************************

        # *********************************************************************
        # *** Save pRF time course models

        print('------Save pRF time course models to disk')

        # The data will come out of the convolution process with an extra
        # dimension, since in principle different basis functions in addition
        # to the canonical HRF can be used. But for now the model fitting can
        # only handle option 1 (canonical convolution). Therefore, we
        # check that user has set the switchHrfSet in the csv file to 1
        strErrMsg = 'Stopping program. ' + \
            'Only canonical hrf fitting is currently supported. ' + \
            'Set switchHrfSet equal to 1 in csv file in order to continue. '
        assert cfg.switchHrfSet == 1, strErrMsg
        # we reduce the dimensions by squeezing
        aryPrfTc = np.squeeze(aryPrfTc)

        # Save the 4D array as '*.npy' file:
        np.save(cfg.strPathMdl, aryPrfTc)
        # Save the corresponding model parameters
        np.save(cfg.strPathMdl + "_params", aryMdlParams)
        del(aryMdlParams)

        # *********************************************************************

    else:

        # *********************************************************************
        # %% Load existing pRF time course models

        print('------Load pRF time course models from disk')

        # Load the file:
        aryPrfTc = np.load((cfg.strPathMdl + '.npy'))

        # Check whether pRF time course model matrix has the expected
        # dimensions:
        vecPrfTcShp = aryPrfTc.shape

        # Logical test for correct dimensions:
        strErrMsg = ('---Error: Dimensions of specified pRF time course ' +
                     'models do not agree with specified model parameters')
        assert vecPrfTcShp[0] == cfg.varNum1 * \
            cfg.varNum2 * cfg.varNumPrfSizes and \
            vecPrfTcShp[1] == cfg.varNumVol, strErrMsg

    # *************************************************************************

    return aryPrfTc
