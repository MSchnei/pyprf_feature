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
                                                         crt_prf_ftr_tc,
                                                         )


def model_creation(dicCnfg, varRat=None, strPathHrf=None):
    """
    Create or load pRF model time courses.

    Parameters
    ----------
    dicCnfg : dict
        Dictionary containing config parameters.
    varRat : float, default None
        Ratio of size suppressive surround to size of center pRF
    strPathHrf : str or None:
        Path to npy file with custom hrf parameters. If None, default
        parameters will be used.

    Returns
    -------
    aryPrfTc : np.array
        4D numpy array with pRF time course models, with following dimensions:
        'aryPrfTc[x-position, y-position, SD, volume]'.
    lgcMdlInc : np.array, boolean
        Logical to only include models with pRF center on stimulated area.

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

        # Here we assume scientific convention and orientation of images where
        # the origin should fall in the lower left corner, the x-axis occupies
        # the width and the y-axis occupies the height dimension of the screen.
        # We also assume that the first dimension that the user provides
        # indexes x and the second indexes the y-axis. Since python is column
        # major (i.e. first indexes columns, only then rows), we need to rotate
        # arySptExpInf by 90 degrees rightward. This will insure that with the
        # 0th axis we index the scientific x-axis and higher values move us to
        # the right on that x-axis. It will also ensure that the 1st
        # python axis indexes the scientific y-axis and higher values will
        # move us up.
        arySptExpInf = np.rot90(arySptExpInf, k=3)

        # *********************************************************************

        # *********************************************************************
        # *** Load temporal condition information

        print('------Load temporal condition information')

        aryTmpExpInf = np.load(cfg.strTmpExpInf)
        # add fourth column to make it appropriate for pyprf_feature
        if aryTmpExpInf.shape[-1] == 3:
            print('---------Added fourth column')
            vecNewCol = np.greater(aryTmpExpInf[:, 0], 0).astype(np.float16)
            aryTmpExpInf = np.concatenate(
                (aryTmpExpInf, np.expand_dims(vecNewCol, axis=1)), axis=1)

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

        # If desired by user, also create model parameters for supp surround
        if varRat is not None:
            aryMdlParamsSur = np.copy(aryMdlParams)
            aryMdlParamsSur[:, 2] = aryMdlParamsSur[:, 2] * varRat

        # *********************************************************************

        # *********************************************************************
        # *** Create 2D Gauss model responses to spatial conditions.

        print('------Create 2D Gauss model responses to spatial conditions')

        aryMdlRsp = crt_mdl_rsp(arySptExpInf, (int(cfg.varVslSpcSzeX),
                                               int(cfg.varVslSpcSzeY)),
                                aryMdlParams, cfg.varPar)

        # If desired by user, also create model responses for supp surround
        if varRat is not None:
            aryMdlRspSur = crt_mdl_rsp(arySptExpInf, (int(cfg.varVslSpcSzeX),
                                                      int(cfg.varVslSpcSzeY)),
                                       aryMdlParamsSur, cfg.varPar)

        # Delete array to save memory
        del(arySptExpInf)

        # *********************************************************************

        # *********************************************************************
        # *** Create prf time course models

        print('------Create prf time course models')

        # Check whether path to npy file with hrf parameters was provided
        if strPathHrf is not None:
            print('---------Load custom hrf parameters')
            aryCstPrm = np.load(strPathHrf)
            dctPrm = {}
            dctPrm['peak_delay'] = aryCstPrm[0]
            dctPrm['under_delay'] = aryCstPrm[1]
            dctPrm['peak_disp'] = aryCstPrm[2]
            dctPrm['under_disp'] = aryCstPrm[3]
            dctPrm['p_u_ratio'] = aryCstPrm[4]
        # If not, set dctPrm to None, which will result in default hrf params
        else:
            print('---------Use default hrf parameters')
            dctPrm = None

        aryPrfTc = crt_prf_ftr_tc(aryMdlRsp, aryTmpExpInf, cfg.varNumVol,
                                  cfg.varTr, cfg.varTmpOvsmpl,
                                  cfg.switchHrfSet, (int(cfg.varVslSpcSzeX),
                                                     int(cfg.varVslSpcSzeY)),
                                  cfg.varPar, dctPrm=dctPrm)

        # If desired by user, create prf time course models for supp surround
        if varRat is not None:
            print('---------Add suppressive surround')
            aryPrfTcSur = crt_prf_ftr_tc(aryMdlRspSur, aryTmpExpInf,
                                         cfg.varNumVol, cfg.varTr,
                                         cfg.varTmpOvsmpl, cfg.switchHrfSet,
                                         (int(cfg.varVslSpcSzeX),
                                          int(cfg.varVslSpcSzeY)),
                                         cfg.varPar, dctPrm=dctPrm)
            # Concatenate aryPrfTc and aryPrfTcSur
            aryPrfTc = np.concatenate((aryPrfTc, aryPrfTcSur), axis=1)

        # *********************************************************************

        # *********************************************************************
        # *** Save pRF time course models, corresponding params and responses

        print('------Save pRF time course models to disk')

        # Prepare file name extensions
        strNmeExtMdl = ''
        strNmeExtPrm = '_params'
        strNmeExtRsp = '_mdlRsp'

        # Check whether extensions need to be modified with ratio name
        if varRat is not None:
            strNmeExtMdl = strNmeExtMdl + '_' + str(varRat)
            strNmeExtPrm = strNmeExtPrm + '_' + str(varRat)
            strNmeExtRsp = strNmeExtRsp + '_' + str(varRat)
            # Also include model the parameters and responses of the surround
            # For the pRF time course models, the surround is included above
            aryMdlParams = np.stack((aryMdlParams, aryMdlParamsSur),
                                    axis=1)
            aryMdlRsp = np.stack((aryMdlRsp, aryMdlRspSur),
                                 axis=1)

        # Save pRF time course models
        np.save(cfg.strPathMdl + strNmeExtMdl, aryPrfTc)
        # Save the corresponding model parameters
        np.save(cfg.strPathMdl + strNmeExtPrm, aryMdlParams)
        # Save the corresponding model responses
        np.save(cfg.strPathMdl + strNmeExtRsp, aryMdlRsp)

        del(aryMdlParams)
        del(aryMdlRsp)

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
            cfg.varNum2 * cfg.varNumPrfSizes, strErrMsg
        assert vecPrfTcShp[-1] == cfg.varNumVol, strErrMsg
        # Check number of feature. If fitting is performed with sup surround,
        # number of features will be twice as many as simple fitting
        if varRat is None:
            assert vecPrfTcShp[1] == cfg.switchHrfSet, strErrMsg
        else:
            assert vecPrfTcShp[1] == cfg.switchHrfSet*2, strErrMsg

    # *************************************************************************

    return aryPrfTc
