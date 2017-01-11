# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 17:12:06 2016

@author: marian
"""
import numpy as np


def getBetas(idxPrc,
             aryFitMdlParams,
             aryDsgn,
             estimPrfChunk,
             simRespChunk,
             queOut):

    aryEstimMtnCrv = np.empty((estimPrfChunk.shape[0], 9), dtype='float32')

    # Prepare status indicator if this is the first of the parallel processes:
    if idxPrc == 0:

        # We create a status indicator for the time consuming pRF model finding
        # algorithm. Number of steps of the status indicator:
        varStsStpSze = 20

        # Vector with pRF values at which to give status feedback:
        vecStatPrf = np.linspace(0,
                                 len(aryFitMdlParams),
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

    for ind, aryEst in enumerate(aryFitMdlParams):

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
                             str(len(aryFitMdlParams)))

                print(strStsMsg)

                # Only increment counter if the last value has not been
                # reached yet:
                if varCntSts01 < varStsStpSze:
                    varCntSts01 = varCntSts01 + int(1)

        # find simulated models that got estimated to have the parameters
        # of the current fit model
        lgcTemp = [np.sum(estimPrfChunk - aryEst, axis=1) == 0][0]
        if np.greater(np.sum(lgcTemp), 0):
            aryDsgnTemp = aryDsgn[ind, :, :]
            yTemp = simRespChunk[lgcTemp, :].T
            aryTmpPrmEst = np.linalg.lstsq(aryDsgnTemp, yTemp)[0]
            aryEstimMtnCrv[lgcTemp, :] = aryTmpPrmEst.T

        # Status indicator (only used in the first of the parallel
        # processes):
        if idxPrc == 0:
            # Increment status indicator counter:
            varCntSts02 = varCntSts02 + 1

    # Output list:
    lstOut = [idxPrc,
              aryEstimMtnCrv]
    queOut.put(lstOut)
