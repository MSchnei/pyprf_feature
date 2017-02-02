# -*- coding: utf-8 -*-

"""Main procedures for populaton receptive field (pRF) finding."""

# Part of py_pRF_motion library
# Copyright (C) 2016  Ingo Marquardt
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


# %%
def funcFindPrf(idxPrc, vecMdlXpos, vecMdlYpos, vecMdlSd, aryFuncChnk,
                aryPrfTc, lgcCython, queOut):
    """Find best pRF model for voxel time course (no cross-validation).

        Parameters
        ----------
        idxPrc : TODO
        vecMdlXpos : TODO
        vecMdlYpos : TODO
        vecMdlSd : TODO
        aryFuncChnk : TODO
        aryPrfTc : TODO
        lgcCython : TODO
        queOut : TODO
        
    """
    # Number of voxels to be fitted in this chunk:
    varNumVoxChnk = aryFuncChnk.shape[0]

    # Number of volumes:
    varNumVol = aryFuncChnk.shape[1]

    # number of motion directions
    varNumMtnDrctns = aryPrfTc.shape[3]

    # Vectors for pRF finding results [number-of-voxels times one]:
    vecBstXpos = np.zeros(varNumVoxChnk)
    vecBstYpos = np.zeros(varNumVoxChnk)
    vecBstSd = np.zeros(varNumVoxChnk)
    if lgcCython:
        vecBstBetas = np.zeros((varNumVoxChnk, varNumMtnDrctns),
                               dtype='float32')
    else:
        vecBstBetas = np.zeros((varNumVoxChnk, varNumMtnDrctns+1),
                               dtype='float32')

    # Vector that will hold the temporary residuals from the model fitting:
    vecBstRes = np.zeros(varNumVoxChnk, dtype='float32')
    vecBstRes[:] = np.inf

    # We reshape the voxel time courses, so that time goes down the column,
    # i.e. from top to bottom.
    aryFuncChnk = aryFuncChnk.T

    # Prepare data for cython (i.e. accelerated) least squares finding:
    if lgcCython:
        # Instead of fitting a constant term, we subtract the mean from the
        # data and from the model ("FSL style") First, we subtract the mean
        # over time from the data:
        aryFuncChnkTmean = np.array(np.mean(aryFuncChnk, axis=0), ndmin=2)
        aryFuncChnk = np.subtract(aryFuncChnk, aryFuncChnkTmean[0, None])
        # Secondly, we subtract the mean over time form the pRF model time
        # courses. The array has four dimensions, the 4th is time (one to three
        # are x-position, y-position, and pRF size (SD)).
        aryPrfTcTmean = np.mean(aryPrfTc, axis=3)
        aryPrfTc = np.subtract(aryPrfTc, aryPrfTcTmean[:, :, :, None])
    # Otherwise, create constant term for numpy least squares finding:
    else:
        # Constant term for the model:
        vecConst = np.ones((varNumVol), dtype=np.float32)

    # Change type to float 32:
    aryFuncChnk = aryFuncChnk.astype(np.float32)
    aryPrfTc = aryPrfTc.astype(np.float32)

    # Prepare status indicator if this is the first of the parallel processes:
    if idxPrc == 0:

        # We create a status indicator for the time consuming pRF model finding
        # algorithm. Number of steps of the status indicator:
        varStsStpSze = 20

        # Number of pRF models to fit:
        varNumMdls = (len(vecMdlXpos) * len(vecMdlYpos) * len(vecMdlSd))

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
    for idxX, mdlPrmX in enumerate(vecMdlXpos):

        for idxY, mdlPrmY in enumerate(vecMdlYpos):

            for idxSd, mdlPrmSd in enumerate(vecMdlSd):

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

                # Cython version:
                if lgcCython:
                    # currently not implemented
                    raise ValueError("Cython currently not implemented")

                # Numpy version:
                else:

                    # Current pRF time course model:
                    vecMdlTc = aryPrfTc[idxX, idxY, idxSd, :, :]

                    # We create a design matrix including the current pRF time
                    # course model, and a constant term:
                    aryDsgn = np.vstack([vecMdlTc,
                                         vecConst]).T

                    # Calculate the least-squares solution for all voxels:
                    vecTmpBetas, vecTmpRes = np.linalg.lstsq(aryDsgn,
                                                             aryFuncChnk)[0:2]

                # Check whether current residuals are lower than previously
                # calculated ones:
                vecLgcTmpRes = np.less(vecTmpRes, vecBstRes)

                # Replace best x and y position values, and SD values.
                vecBstXpos[vecLgcTmpRes] = mdlPrmX
                vecBstYpos[vecLgcTmpRes] = mdlPrmY
                vecBstSd[vecLgcTmpRes] = mdlPrmSd
                vecBstBetas[vecLgcTmpRes, :] = vecTmpBetas[:, vecLgcTmpRes].T

                # Replace best residual values:
                vecBstRes[vecLgcTmpRes] = vecTmpRes[vecLgcTmpRes]

                # Status indicator (only used in the first of the parallel
                # processes):
                if idxPrc == 0:

                    # Increment status indicator counter:
                    varCntSts02 = varCntSts02 + 1

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
              vecBstXpos,
              vecBstYpos,
              vecBstSd,
              vecBstR2,
              vecBstBetas]

    queOut.put(lstOut)


# %%
def funcFindPrfXval(idxPrc, vecMdlXpos, vecMdlYpos, vecMdlSd, lstFuncTrnChnk,
                    lstFuncTstChnk, lstPrfMdlsTrn, lstPrfMdlsTst, lgcCython,
                    varNumXval, queOut):
    """Find best pRF model for voxel timecourse (with cross-validation).

    Parameters
    ----------
    idxPrc : TODO
    vecMdlXpos : TODO
    vecMdlYpos : TODO
    vecMdlSd : TODO
    lstFuncTrnChnk : TODO
    lstFuncTstChnk : TODO
    lstPrfMdlsTrn : TODO
    lstPrfMdlsTst : TODO
    lgcCython : TODO
    varNumXval : TODO
    queOut : TODO

    """
    # Number of voxels to be fitted in this chunk:
    assert lstFuncTrnChnk[0].shape[0] == lstFuncTstChnk[0].shape[0]
    varNumVoxChnk = lstFuncTrnChnk[0].shape[0]

    # Vectors for pRF finding results [number-of-voxels times one]:
    vecBstXpos = np.zeros(varNumVoxChnk)
    vecBstYpos = np.zeros(varNumVoxChnk)
    vecBstSd = np.zeros(varNumVoxChnk)

    # Vector that will hold the temporary residuals from the model fitting:
    vecBstRes = np.zeros(varNumVoxChnk).astype(np.float32)
    vecBstRes[:] = np.inf

    # vector for temporary residuals of cross validation
    vecTmpResXVal = np.zeros((varNumVoxChnk, varNumXval), dtype='float32')
    vecTmpResXVal[:] = np.inf

    # We reshape the voxel time courses, so that time goes down the column,
    # i.e. from top to bottom.
    for ind in np.arange(varNumXval):
        lstFuncTrnChnk[ind] = lstFuncTrnChnk[ind].T
        lstFuncTstChnk[ind] = lstFuncTstChnk[ind].T

    # Prepare status indicator if this is the first of the parallel processes:
    if idxPrc == 0:

        # We create a status indicator for the time consuming pRF model finding
        # algorithm. Number of steps of the status indicator:
        varStsStpSze = 20

        # Number of pRF models to fit:
        varNumMdls = (len(vecMdlXpos) * len(vecMdlYpos) * len(vecMdlSd))

        # Vector with pRF values at which to give status feedback:
        vecStatPrf = np.linspace(0,
                                 varNumMdls,
                                 num=(varStsStpSze1),
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
    for idxX, mdlPrmX in enumerate(vecMdlXpos):

        for idxY, mdlPrmY in enumerate(vecMdlYpos):

            for idxSd, mdlPrmSd in enumerate(vecMdlSd):

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

                # Cython version:
                if lgcCython:
                    # currently not implemented
                    raise ValueError("Cython currently not implemented")

                # Numpy version:
                else:
                    for idxXval in range(0, varNumXval):
                        # Get pRF time course model for training and test:
                        # transpose so that time runds down the column
                        aryDsgnTrn = lstPrfMdlsTrn[idxXval][
                            idxX, idxY, idxSd, :, :].T
                        aryDsgnTst = lstPrfMdlsTst[idxXval][
                            idxX, idxY, idxSd, :, :].T

                        # Calculate the least-squares solution for all voxels
                        # and get parameter estimates from the training fit
                        aryTmpPrmEst = np.linalg.lstsq(aryDsgnTrn,
                                                       lstFuncTrnChnk[
                                                           idxXval])[0]
                        # calculate predicted model fit based on training data
                        aryTmpMdlTc = np.dot(aryDsgnTst, aryTmpPrmEst)
                        # calculate residual sum of squares between test data
                        # and predicted model fit based on training data
                        vecTmpResXVal[:, idxXval] = np.sum(
                            (np.subtract(lstFuncTstChnk[idxXval],
                                         aryTmpMdlTc))**2, axis=0)

                    vecTmpRes = np.mean(vecTmpResXVal, axis=1)

                # Check whether current residuals are lower than previously
                # calculated ones:
                vecLgcTmpRes = np.less(vecTmpRes, vecBstRes)

                # Replace best x and y position values, and SD values.
                vecBstXpos[vecLgcTmpRes] = mdlPrmX
                vecBstYpos[vecLgcTmpRes] = mdlPrmY
                vecBstSd[vecLgcTmpRes] = mdlPrmSd

                # Replace best residual values:
                vecBstRes[vecLgcTmpRes] = vecTmpRes[vecLgcTmpRes]

                # Status indicator (only used in the first of the parallel
                # processes):
                if idxPrc == 0:

                    # Increment status indicator counter:
                    varCntSts02 = varCntSts02 + 1

    # Output list:
    lstOut = [idxPrc,
              vecBstXpos,
              vecBstYpos,
              vecBstSd]

    queOut.put(lstOut)
