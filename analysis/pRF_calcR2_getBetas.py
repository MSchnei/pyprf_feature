# -*- coding: utf-8 -*-

"""Procedure to calculate betas and R^2 for the best model.""" 

# Part of py_pRF_motion library
# Copyright (C) 2016  Marian Schneider
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

import numpy as np
import sklearn


def getBetas(idxPrc,
             aryPrfTc,
             lstAllMdlInd,
             aryFuncChnk,
             aryBstIndChnk,
             betaSw,
             queOut):
    """Calculate voxel betas and R^2 for the best model.

    Parameters
    ----------
    idxPrc : TODO
        (?)
    aryPrfTc : np.array, shape (?)
        Population receptive field time courses.
    lstAllMdlInd : list
        List of the indices of all models.
    aryFuncChnk : TODO
        Chunk of something(?)
    aryBstIndChnk : np.array, shape (?)
        Points for every voxel to the index of the best model
    betaSw : str, iterator, or np.array, shape (?)
        Best beta correlation coefficients found in training.
    queOut : TODO
        Queue output (?)

    Notes
    -----
    This is done after fitting with cross validation, since during the
    fitting process, we never fit the model to the entire data.
    """

    # get number of motion directions
    varNumMtnDrctns = aryPrfTc.shape[3]
    varNumVoxChnk = aryBstIndChnk.shape[0]
    # prepare array for best beta weights
    if type(betaSw) is sklearn.model_selection._split.KFold:
        aryEstimMtnCrvTrn = np.zeros((varNumVoxChnk, varNumMtnDrctns,
                                      betaSw.get_n_splits()), dtype='float32')
        aryEstimMtnCrvTst = np.zeros((varNumVoxChnk, varNumMtnDrctns,
                                      betaSw.get_n_splits()), dtype='float32')
        resTrn = np.zeros((varNumVoxChnk, betaSw.get_n_splits()),
                          dtype='float32')
        resTst = np.zeros((varNumVoxChnk, betaSw.get_n_splits()),
                          dtype='float32')
        aryErrorTrn = np.zeros((varNumVoxChnk), dtype='float32')
        aryErrorTst = np.zeros((varNumVoxChnk), dtype='float32')
        contrast = np.array([
                     [1, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0],
                     [0, 0, 1, 0, 0],
                     [0, 0, 0, 1, 0],
                     ])
        denomTrn = np.zeros((varNumVoxChnk, betaSw.get_n_splits(),
                             len(contrast)), dtype='float32')
        denomTst = np.zeros((varNumVoxChnk, betaSw.get_n_splits(),
                             len(contrast)), dtype='float32')

    elif type(betaSw) is np.ndarray and betaSw.dtype == 'bool':
        aryEstimMtnCrvTrn = np.zeros((varNumVoxChnk, varNumMtnDrctns,
                                      ), dtype='float32')
        aryEstimMtnCrvTst = np.zeros((varNumVoxChnk, varNumMtnDrctns,
                                      ), dtype='float32')
        resTrn = np.zeros((varNumVoxChnk),
                          dtype='float32')
        resTst = np.zeros((varNumVoxChnk),
                          dtype='float32')
        aryErrorTrn = np.zeros((varNumVoxChnk), dtype='float32')
        aryErrorTst = np.zeros((varNumVoxChnk), dtype='float32')
        contrast = np.array([
                     [1, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0],
                     [0, 0, 1, 0, 0],
                     [0, 0, 0, 1, 0],
                     ])
        denomTrn = np.zeros((varNumVoxChnk,
                             len(contrast)), dtype='float32')
        denomTst = np.zeros((varNumVoxChnk,
                             len(contrast)), dtype='float32')

    else:
        aryEstimMtnCrv = np.zeros((varNumVoxChnk, varNumMtnDrctns),
                                  dtype='float32')
    # prepare array for best residuals
    vecBstRes = np.zeros(varNumVoxChnk, dtype='float32')
    vecBstRes[:] = np.inf
    # prepare counter to check that every voxel is matched to one winner mdl
    vecLgcCounter = np.zeros(varNumVoxChnk, dtype='float32')

    # We reshape the voxel time courses, so that time goes down the column
    aryFuncChnk = aryFuncChnk.T

    # Change type to float 32:
    aryFuncChnk = aryFuncChnk.astype(np.float32)
    aryPrfTc = aryPrfTc.astype(np.float32)

    # Prepare status indicator if this is the first of the parallel processes:
    if idxPrc == 0:

        # We create a status indicator for the time consuming pRF model finding
        # algorithm. Number of steps of the status indicator:
        varStsStpSze = 20

        # Number of pRF models to fit:
        varNumMdls = len(lstAllMdlInd)

        # Vector with pRF values at which to give status feedback:
        vecStatPrf = np.linspace(0,
                                 varNumMdls,
                                 num=(varStsStpSze+1),
                                 endpoint=True)
        vecStatPrf = np.ceil(vecStatPrf)
        vecStatPrf = vecStatPrf.astype(int)

        # Vector with corresponding percentage values at which to give status
        # feedback:
        vecStatPrc = np.linspace(0,
                                 100,
                                 num=(varStsStpSze+1),
                                 endpoint=True)
        vecStatPrc = np.ceil(vecStatPrc)
        vecStatPrc = vecStatPrc.astype(int)

        # Counter for status indicator:
        varCntSts01 = 0
        varCntSts02 = 0

    # Loop through pRF models:
    for idx, mdlInd in enumerate(lstAllMdlInd):

        # Status indicator (only used in the first of the parallel
        # processes):
        if idxPrc == 0:

            # Status indicator:
            if varCntSts02 == vecStatPrf[varCntSts01]:

                # Prepare status message:
                strStsMsg = ('---------Progress: ' +
                             str(vecStatPrc[varCntSts01]) +
                             ' % --- ' +
                             str(vecStatPrf[varCntSts01]) +
                             ' pRF models out of ' +
                             str(varNumMdls))

                print(strStsMsg)

                # Only increment counter if the last value has not been
                # reached yet:
                if varCntSts01 < varStsStpSze:
                    varCntSts01 = varCntSts01 + int(1)

        # check whether any model had this particular x, y, sigma combination
        # as its best model
        lgcTemp = [aryBstIndChnk == idx][0]

        if np.greater(np.sum(lgcTemp), 0):
            # get current design matrix
            aryDsgnTmp = aryPrfTc[mdlInd].T

            if betaSw is 'train':  # training

                aryTmpPrmEst, aryTmpRes = np.linalg.lstsq(
                    aryDsgnTmp, aryFuncChnk[:, lgcTemp])[0:2]
                aryEstimMtnCrv[lgcTemp, :] = aryTmpPrmEst.T
                vecBstRes[lgcTemp] = aryTmpRes

            elif type(betaSw) is np.ndarray and betaSw.dtype == 'float':
                # get beta weights for axis of motion tuning curves
                aryEstimMtnCrv[lgcTemp, :] = np.linalg.lstsq(
                    aryDsgnTmp, aryFuncChnk[:, lgcTemp])[0].T
                # calculate prediction
                aryPredTc = np.dot(aryDsgnTmp,
                                   betaSw[lgcTemp, :].T)
                # Sum of squares:
                vecBstRes[lgcTemp] = np.sum((aryFuncChnk[:, lgcTemp] -
                                             aryPredTc) ** 2,  axis=0)

            elif type(betaSw) is np.ndarray and betaSw.dtype == 'bool':
                # get beta weights for training
                betas, resTrn[lgcTemp] = np.linalg.lstsq(
                    aryDsgnTmp[betaSw, :], aryFuncChnk[betaSw][:, lgcTemp])[0:2]
                aryEstimMtnCrvTrn[lgcTemp, :] = betas.T
                # get beta weights for validation
                betas, resTst[lgcTemp] = np.linalg.lstsq(
                    aryDsgnTmp[~betaSw, :], aryFuncChnk[~betaSw][:, lgcTemp])[0:2]
                aryEstimMtnCrvTrn[lgcTemp, :] = betas.T
                # calculate CC for training
                aryCcTrn = np.linalg.pinv(
                    np.dot(aryDsgnTmp[betaSw, :].T,
                           aryDsgnTmp[betaSw, :]))
                aryCcTst = np.linalg.pinv(
                    np.dot(aryDsgnTmp[~betaSw, :].T,
                           aryDsgnTmp[~betaSw, :]))
                # calculate Error for training
                aryErrorTrn[lgcTemp] = np.var(
                    np.subtract(aryFuncChnk[betaSw][:, lgcTemp],
                                np.dot(aryDsgnTmp[betaSw, :],
                                       aryEstimMtnCrvTrn[lgcTemp, :].T)), axis=0)
                # calculate Error for test
                aryErrorTst[lgcTemp] = np.var(
                    np.subtract(aryFuncChnk[~betaSw][:, lgcTemp],
                                np.dot(aryDsgnTmp[~betaSw, :],
                                       aryEstimMtnCrvTst[lgcTemp, :].T)), axis=0)
                # calculate denominator for training
                for indContr, contr in enumerate(contrast):
                    denomTrn[lgcTemp, indContr] = np.sqrt(
                        aryErrorTrn[lgcTemp] * np.dot(
                            np.dot(contr, aryCcTrn), contr.T))
                    denomTst[lgcTemp, indContr] = np.sqrt(
                        aryErrorTst[lgcTemp] * np.dot(
                            np.dot(contr, aryCcTst), contr.T))

            elif type(betaSw) is sklearn.model_selection._split.KFold:
                for idxCV, (idxTrn, idxVal) in enumerate(betaSw.split(aryDsgnTmp)):
                    # get beta weights for training
                    betas, resTrn[lgcTemp, idxCV] = np.linalg.lstsq(
                        aryDsgnTmp[idxTrn], aryFuncChnk[idxTrn][:, lgcTemp])[0:2]
                    aryEstimMtnCrvTrn[lgcTemp, :, idxCV] = betas.T
                    # get beta weights for validation
                    betas, resTst[lgcTemp, idxCV] = np.linalg.lstsq(
                        aryDsgnTmp[idxVal], aryFuncChnk[idxVal][:, lgcTemp])[0:2]
                    aryEstimMtnCrvTst[lgcTemp, :, idxCV] = betas.T
                    # calculate CC for training
                    aryCcTrn = np.linalg.pinv(
                        np.dot(aryDsgnTmp[idxTrn].T,
                               aryDsgnTmp[idxTrn]))
                    aryCcTst = np.linalg.pinv(
                        np.dot(aryDsgnTmp[idxVal].T,
                               aryDsgnTmp[idxVal]))
                    # calculate Error for training
                    aryErrorTrn[lgcTemp] = np.var(
                        np.subtract(aryFuncChnk[idxTrn][:, lgcTemp],
                                    np.dot(aryDsgnTmp[idxTrn],
                                           aryEstimMtnCrvTrn[lgcTemp, :, idxCV].T)), axis=0)
                    # calculate Error for test
                    aryErrorTst[lgcTemp] = np.var(
                        np.subtract(aryFuncChnk[idxVal][:, lgcTemp],
                                    np.dot(aryDsgnTmp[idxVal],
                                           aryEstimMtnCrvTst[lgcTemp, :, idxCV].T)), axis=0)
                    # calculate denominator for training
                    for indContr, contr in enumerate(contrast):
                        denomTrn[lgcTemp, idxCV, indContr] = np.sqrt(
                            aryErrorTrn[lgcTemp] * np.dot(
                                np.dot(contr, aryCcTrn), contr.T))
                        denomTst[lgcTemp, idxCV, indContr] = np.sqrt(
                            aryErrorTst[lgcTemp] * np.dot(
                                np.dot(contr, aryCcTst), contr.T))

            # increase logical counter to verify later that every voxel
            # was visited only once
            vecLgcCounter[lgcTemp] += 1

        # Status indicator (only used in the first of the parallel
        # processes):
        if idxPrc == 0:
            # Increment status indicator counter:
            varCntSts02 = varCntSts02 + 1

    # check that every voxel was visited only once
    strErrMsg = ('It looks like at least voxel was revisted more than once. ' +
                 'Check whether the R2 was calculated correctly')
    assert np.sum(vecLgcCounter) == len(vecLgcCounter), strErrMsg

    if type(betaSw) is sklearn.model_selection._split.KFold:

        # calculate t-values
        aryTvalsTrn = np.empty((varNumVoxChnk, contrast.shape[0],
                                betaSw.get_n_splits()))
        aryTvalsTst = np.empty((varNumVoxChnk, contrast.shape[0],
                                betaSw.get_n_splits()))

        for ind1, contr in enumerate(contrast):
            for ind2 in range(betaSw.get_n_splits()):
                aryTvalsTrn[:, ind1, ind2] = np.divide(
                    np.dot(contr, aryEstimMtnCrvTrn[:, :, ind2].T),
                    denomTrn[:, ind2, ind1])
                aryTvalsTst[:, ind1, ind2] = np.divide(
                    np.dot(contr, aryEstimMtnCrvTst[:, :, ind2].T),
                    denomTst[:, ind2, ind1])

        # Output list:
        lstOut = [idxPrc,
                  aryEstimMtnCrvTrn,
                  aryEstimMtnCrvTst,
                  aryTvalsTrn,
                  aryTvalsTst,
                  ]
        queOut.put(lstOut)

    elif type(betaSw) is np.ndarray and betaSw.dtype == 'bool':
        # calculate t-values
        aryTvalsTrn = np.empty((varNumVoxChnk, contrast.shape[0],
                                ))
        aryTvalsTst = np.empty((varNumVoxChnk, contrast.shape[0],
                                ))

        for ind1, contr in enumerate(contrast):
                aryTvalsTrn[:, ind1] = np.divide(
                    np.dot(contr, aryEstimMtnCrvTrn.T),
                    denomTrn[:, ind1])
                aryTvalsTst[:, ind1] = np.divide(
                    np.dot(contr, aryEstimMtnCrvTst.T),
                    denomTst[:, ind1])

        # Output list:
        lstOut = [idxPrc,
                  aryEstimMtnCrvTrn,
                  aryEstimMtnCrvTst,
                  aryTvalsTrn,
                  aryTvalsTst,
                  ]
        queOut.put(lstOut)

    else:
        # After finding the best fitting model for each voxel, we still have to
        # calculate the coefficient of determination (R-squared) for each voxel. We
        # start by calculating the total sum of squares (i.e. the deviation of the
        # data from the mean). The mean of each time course:
        vecFuncMean = np.mean(aryFuncChnk, axis=0)
        # Deviation from the mean for each datapoint:
        vecFuncDev = np.subtract(aryFuncChnk, vecFuncMean[None, :])
        # Sum of squares:
        vecSsTot = np.sum(np.power(vecFuncDev,
                                   2.0),
                          axis=0)
        # Coefficient of determination:
        vecBstR2 = np.subtract(1.0,
                               np.divide(vecBstRes,
                                         vecSsTot))
        # Output list:
        lstOut = [idxPrc,
                  vecBstR2,
                  aryEstimMtnCrv]
        queOut.put(lstOut)
