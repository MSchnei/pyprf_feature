# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 17:12:06 2016

@author: marian
"""
import numpy as np


def getBetas(idxPrc,
             vecMdlXpos,
             vecMdlYpos,
             vecMdlSd,
             aryPrfTc,
             aryFuncChnk,
             aryBstMdls,
             queOut):

    # get number of motion directions
    varNumMtnDrctns = aryPrfTc.shape[3]
    varNumVoxChnk = aryBstMdls.shape[0]
    # prepare array for best beta weights
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
                                     str(len(aryBstMdls)))

                        print(strStsMsg)

                        # Only increment counter if the last value has not been
                        # reached yet:
                        if varCntSts01 < varStsStpSze:
                            varCntSts01 = varCntSts01 + int(1)

                # check whether any model had this particular x, y, sigma
                # combination as its best model
                tmpBstMdl = np.array([mdlPrmX, mdlPrmY, mdlPrmSd])
                lgcTemp = [np.sum(np.isclose(aryBstMdls - tmpBstMdl, 0),
                                  axis=1) == 3][0]

                if np.greater(np.sum(lgcTemp), 0):
                    aryDsgnTmp = aryPrfTc[idxX, idxY, idxSd, :, :].T
                    aryTmpPrmEst, aryTmpRes = np.linalg.lstsq(
                        aryDsgnTmp, aryFuncChnk[:, lgcTemp])[0:2]
                    aryEstimMtnCrv[lgcTemp, :] = aryTmpPrmEst.T
                    vecBstRes[lgcTemp] = aryTmpRes
                    vecLgcCounter[lgcTemp] += 1

                # Status indicator (only used in the first of the parallel
                # processes):
                if idxPrc == 0:
                    # Increment status indicator counter:
                    varCntSts02 = varCntSts02 + 1

    # check that every voxel was visited only once
    assert np.sum(vecLgcCounter) == len(vecLgcCounter)
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
