# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 11:49:18 2016

@author: marian
"""

from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
import os
import tables
import numpy as np
import scipy as sp
from scipy.stats import gamma


# ***  Define functions


# %%
def funcGauss1D(x, mu, sig):
    """ Create 1D Gaussian. Source:
    http://mathworld.wolfram.com/GaussianFunction.html
    """
    arrOut = np.exp(-np.power((x - mu)/sig, 2.)/2)
    # normalize
#    arrOut = arrOut/(np.sqrt(2.*np.pi)*sig)
    # normalize (laternative)
    arrOut = arrOut/np.sum(arrOut)
    return arrOut


# %%
def funcGauss2D(varSizeX, varSizeY, varPosX, varPosY, varSd):

    """ Create 2D Gaussian kernel. Source:
    http://mathworld.wolfram.com/GaussianFunction.html
    """

    varSizeX = int(varSizeX)
    varSizeY = int(varSizeY)

    # aryX and aryY are in reversed order, this seems to be necessary:
    aryY, aryX = sp.mgrid[0:varSizeX,
                          0:varSizeY]

    # The actual creation of the Gaussian array:
    aryGauss = (
        (
            np.power((aryX - varPosX), 2.0) +
            np.power((aryY - varPosY), 2.0)
        ) /
        (2.0 * np.power(varSd, 2.0))
        )
    aryGauss = np.exp(-aryGauss)
    # normalize
    # aryGauss = aryGauss/(2*np.pi*np.power(varSd, 2))

    return aryGauss


# %%
def makeMotDirTuneCurves(x, mu, sig):
    # find middle of x
    centre = np.round(len(x)/2)
    dif = np.subtract(centre, mu).astype(int)
    # initialise the curve in the centre so it will be symmetric
    arrOut = funcGauss1D(x, mu+dif, sig)
    # roll back so that the peak of the gaussian will be over mu
    arrOut = np.roll(arrOut, -dif)

    return arrOut


# %%
def makeAoMTuneCurves(x, mu1, mu2, sig):
    arr1 = makeMotDirTuneCurves(x, mu1, sig)
    arr2 = makeMotDirTuneCurves(x, mu2, sig)
    arrOut = 0.5 * arr1 + 0.5 * arr2
    return arrOut


# %%
def funcHrf(varNumVol, varTr):

    """ Create double gamma function. Source:
    http://www.jarrodmillman.com/rcsds/lectures/convolution_background.html
    """

    vecX = np.arange(0, varNumVol, 1)

    # Expected time of peak of HRF [s]:
    varHrfPeak = 6.0 / varTr
    # Expected time of undershoot of HRF [s]:
    varHrfUndr = 12.0 / varTr
    # Scaling factor undershoot (relative to peak):
    varSclUndr = 0.35

    # Gamma pdf for the peak
    vecHrfPeak = gamma.pdf(vecX, varHrfPeak)
    # Gamma pdf for the undershoot
    vecHrfUndr = gamma.pdf(vecX, varHrfUndr)
    # Combine them
    vecHrf = vecHrfPeak - varSclUndr * vecHrfUndr

    # Scale maximum of HRF to 1.0:
    vecHrf = np.divide(vecHrf, np.max(vecHrf))

    return vecHrf


# %%
def funcConvPar(aryDm,
                vecHrf,
                varNumVol):

    """
    Function for convolution of pixel-wise 'design matrix' with HRF model.
    """
    # In order to avoid an artefact at the end of the time series, we have to
    # concatenate an empty array to both the design matrix and the HRF model
    # before convolution.
    aryDm = np.concatenate((aryDm, np.zeros((aryDm.shape[0], 100))), axis=1)
    vecHrf = np.concatenate((vecHrf, np.zeros((100,))))

    aryDmConv = np.empty((aryDm.shape[0], varNumVol))
    for idx in range(0, aryDm.shape[0]):
        vecDm = aryDm[idx, :]
        # Convolve design matrix with HRF model:
        aryDmConv[idx, :] = np.convolve(vecDm, vecHrf,
                                        mode='full')[:varNumVol]
    return aryDmConv

# %%
def simulateGaussNoise(mean,
                       SDNoise,
                       varNumTP,
                       varNumCNR,
                       ):
    sims = np.empty((varNumCNR, varNumTP))
    for ind, noise in enumerate(SDNoise):
        sims[ind, :] = np.random.normal(mean, noise, varNumTP)
        # 0 is the mean of the normal distribution you are choosing from
        # 1 is the standard deviation of the normal distribution
        # varNumVol is the number of elements you get in array noise
    return sims


# %%
def simulateAR1(n,
                beta,
                sigma,
                c,
                burnin,
                varNumCNR,
                varNumTP,
                ):
    """
    Simulates an AR(1) model using the parameters beta, c, and sigma.
    Returns an array with length n

    n := number of time points
    beta := correlation weight
    sigma := standard deviation of the noise, can be a vector
    c := constant added to the noise, default 0

    based on:
    source1: https://github.com/ndronen/misc/blob/master/python/ar1.py
    source2: http://stats.stackexchange.com/questions/22742/
             problem-simulating-ar2-process
    source3: https://kurtverstegen.wordpress.com/2013/12/07/simulation/
    """
    # Output array with noise time courses
    noise = np.empty((varNumCNR, varNumTP))
    if burnin == 1:
        burnin = 100
        n = n + burnin

    noiseTemp = c + sp.random.normal(0, 1, n)
    sims = np.zeros(n)
    sims[0] = noiseTemp[0]
    for i in range(1, n):
        sims[i] = beta*sims[i-1] + noiseTemp[i]
    sims = sims[burnin:]
    noise = sigma[:, np.newaxis]*sp.stats.mstats.zscore(sims)
    return noise


# %%
def simulateAR2(n,
                betas,
                sigma,
                c,
                burnin,
                varNumCNR,
                varNumTP,
                ):
    """
    Simulates an AR(2) model using the parameters beta, c, and sigma.
    Returns an array with length n

    n := number of time points
    beta := correlation weights, here two weights must be provided
    sigma := standard deviation of the noise, can be a vector
    c := constant added to the noise, default 0

    based on:
    source1: https://github.com/ndronen/misc/blob/master/python/ar1.py
    source2: http://stats.stackexchange.com/questions/22742/
             problem-simulating-ar2-process
    source3: https://kurtverstegen.wordpress.com/2013/12/07/simulation/
    """
    # Output array with noise time courses
    noise = np.empty((varNumCNR, varNumTP))
    if burnin == 1:
        burnin = 100*len(betas)
        n = n + burnin
    noiseTemp = c + sp.random.normal(0, 1, n)
    sims = np.zeros(n)
    sims[0] = noiseTemp[0]
    sims[1] = noiseTemp[1]
    for i in range(2, n):
        sims[i] = betas[0]*sims[i-1] + betas[1]*sims[i-2] + noiseTemp[i]
    sims = sims[burnin:]
    noise = sigma[:, np.newaxis]*sp.stats.mstats.zscore(sims)
    return noise


# %%

def funcNrlTc(idxPrc,
              varPixX,
              varPixY,
              NrlMdlChunk,
              varNumTP,
              aryCond,  # tngCrvBrds
              path,
              varNumNrlMdls,
              varPar,
              queOut):
    """
    Function for creating neural time course models.
    This function is the easiest way of creating neural time courses, only
    assuming 0 and 1 (this was used by early versions of pRF_find.py).
    """

#    # if hd5 method is used: open file for reading
#    filename = 'aryBoxCar' + str(idxPrc) + '.hdf5'
#    hdf5_path = os.path.join(path, filename)
#    fileH = tables.openFile(hdf5_path, mode='r')

    # Output array with pRF model time courses at all modelled standard
    # deviations for current pixel position:
    aryOut = np.empty((len(NrlMdlChunk), varNumTP),
                      dtype='float32')

    # Prepare status indicator if this is the first of the parallel processes:
    if idxPrc == 1:

        # We create a status indicator for the time consuming pRF model finding
        # algorithm. Number of steps of the status indicator:
        varStsStpSze = 20

        # Number of pRF models to fit:
        varNumLoops = varNumNrlMdls/varPar

        # Vector with pRF values at which to give status feedback:
        vecStatus = np.linspace(0,
                                varNumLoops,
                                num=(varStsStpSze+1),
                                endpoint=True)
        vecStatus = np.ceil(vecStatus)
        vecStatus = vecStatus.astype(int)

        # Vector with corresponding percentage values at which to give status
        # feedback:
        vecStatusPrc = np.linspace(0,
                                   100,
                                   num=(varStsStpSze+1),
                                   endpoint=True)
        vecStatusPrc = np.ceil(vecStatusPrc)
        vecStatusPrc = vecStatusPrc.astype(int)

        # Counter for status indicator:
        varCntSts01 = 0
        varCntSts02 = 0

    # Loop through all Gauss parameters that are in this chunk
    for idx, NrlMdlTrpl in enumerate(NrlMdlChunk):

        # Status indicator (only used in the first of the parallel
        # processes):
        if idxPrc == 1:

            # Status indicator:
            if varCntSts02 == vecStatus[varCntSts01]:

                # Prepare status message:
                strStsMsg = ('---------Progress: ' +
                             str(vecStatusPrc[varCntSts01]) +
                             ' % --- ' +
                             str(vecStatus[varCntSts01]) +
                             ' loops out of ' +
                             str(varNumLoops))

                print(strStsMsg)

                # Only increment counter if the last value has not been
                # reached yet:
                if varCntSts01 < varStsStpSze:
                    varCntSts01 = varCntSts01 + int(1)

        # x pos of Gauss model: NrlMdlTrpl[0]
        # y pos of Gauss model: NrlMdlTrpl[1]
        # std of Gauss model: NrlMdlTrpl[2]
        # index of tng crv model: NrlMdlTrpl[3]
        varTmpX = int(np.around(NrlMdlTrpl[0], 0))
        varTmpY = int(np.around(NrlMdlTrpl[1], 0))

        # Create pRF model (2D):
        aryGauss = funcGauss2D(varPixX,
                               varPixY,
                               varTmpX,
                               varTmpY,
                               NrlMdlTrpl[2])

        # Multiply pixel-wise box car model with Gaussian pRF models:
        aryNrlTcTmp = np.multiply(aryCond, aryGauss[:, :, None])

        # Calculate sum across x- and y-dimensions - the 'area under the
        # Gaussian surface'. This is essentially an unscaled version of the
        # neural time course model (i.e. not yet scaled for the size of
        # the pRF).
        aryNrlTcTmp = np.sum(aryNrlTcTmp, axis=(0, 1))

        # Normalise the nrl time course model to the size of the pRF. This
        # gives us the ratio of 'activation' of the pRF at each time point,
        # or, in other words, the neural time course model.
        aryNrlTcTmp = np.divide(aryNrlTcTmp,
                                np.sum(aryGauss, axis=(0, 1)))

        # Put model time courses into the function's output array:
        aryOut[idx, :] = aryNrlTcTmp

        # Status indicator (only used in the first of the parallel
        # processes):
        if idxPrc == 1:
            # Increment status indicator counter:
            varCntSts02 = varCntSts02 + 1

    # Output list:
    lstOut = [idxPrc,
              aryOut,
              ]

    queOut.put(lstOut)


# %%
def funcNrlTcMotPred(idxPrc,
                     varPixX,
                     varPixY,
                     NrlMdlChunk,
                     varNumTP,
                     aryBoxCar,  # aryCond
                     path,
                     varNumNrlMdls,
                     varNumMtDrctn,
                     varPar,
                     queOut):
    """
    Function for creating neural time course models.
    This function should be used to create neural models if different
    predictors for every motion direction are included.
    """

#    # if hd5 method is used: open file for reading
#    filename = 'aryBoxCar' + str(idxPrc) + '.hdf5'
#    hdf5_path = os.path.join(path, filename)
#    fileH = tables.openFile(hdf5_path, mode='r')

    # Output array with pRF model time courses at all modelled standard
    # deviations for current pixel position:
    aryOut = np.empty((len(NrlMdlChunk), varNumTP, varNumMtDrctn),
                      dtype='float32')

    # Prepare status indicator if this is the first of the parallel processes:
    if idxPrc == 1:

        # We create a status indicator for the time consuming pRF model finding
        # algorithm. Number of steps of the status indicator:
        varStsStpSze = 20

        # Number of pRF models to fit:
        varNumLoops = varNumNrlMdls/varPar

        # Vector with pRF values at which to give status feedback:
        vecStatus = np.linspace(0,
                                varNumLoops,
                                num=(varStsStpSze+1),
                                endpoint=True)
        vecStatus = np.ceil(vecStatus)
        vecStatus = vecStatus.astype(int)

        # Vector with corresponding percentage values at which to give status
        # feedback:
        vecStatusPrc = np.linspace(0,
                                   100,
                                   num=(varStsStpSze+1),
                                   endpoint=True)
        vecStatusPrc = np.ceil(vecStatusPrc)
        vecStatusPrc = vecStatusPrc.astype(int)

        # Counter for status indicator:
        varCntSts01 = 0
        varCntSts02 = 0

    # Loop through all Gauss parameters that are in this chunk
    for idx, NrlMdlTrpl in enumerate(NrlMdlChunk):

        # Status indicator (only used in the first of the parallel
        # processes):
        if idxPrc == 1:

            # Status indicator:
            if varCntSts02 == vecStatus[varCntSts01]:

                # Prepare status message:
                strStsMsg = ('---------Progress: ' +
                             str(vecStatusPrc[varCntSts01]) +
                             ' % --- ' +
                             str(vecStatus[varCntSts01]) +
                             ' loops out of ' +
                             str(varNumLoops))

                print(strStsMsg)

                # Only increment counter if the last value has not been
                # reached yet:
                if varCntSts01 < varStsStpSze:
                    varCntSts01 = varCntSts01 + int(1)

        # x pos of Gauss model: NrlMdlTrpl[0]
        # y pos of Gauss model: NrlMdlTrpl[1]
        # std of Gauss model: NrlMdlTrpl[2]
        # index of tng crv model: NrlMdlTrpl[3]
        varTmpX = int(np.around(NrlMdlTrpl[0], 0))
        varTmpY = int(np.around(NrlMdlTrpl[1], 0))

        # Create pRF model (2D):
        aryGauss = funcGauss2D(varPixX,
                               varPixY,
                               varTmpX,
                               varTmpY,
                               NrlMdlTrpl[2])

        # Multiply pixel-wise box car model with Gaussian pRF models:
        aryNrlTcTmp = np.multiply(aryBoxCar, aryGauss[:, :, None, None])

        # Calculate sum across x- and y-dimensions - the 'area under the
        # Gaussian surface'. This is essentially an unscaled version of the
        # neural time course model (i.e. not yet scaled for the size of
        # the pRF).
        aryNrlTcTmp = np.sum(aryNrlTcTmp, axis=(0, 1))

        # Normalise the nrl time course model to the size of the pRF. This
        # gives us the ratio of 'activation' of the pRF at each time point,
        # or, in other words, the neural time course model.
        aryNrlTcTmp = np.divide(aryNrlTcTmp,
                                np.sum(aryGauss, axis=(0, 1)))

        # Put model time courses into the function's output array:
        aryOut[idx, :, :] = aryNrlTcTmp

        # Status indicator (only used in the first of the parallel
        # processes):
        if idxPrc == 1:
            # Increment status indicator counter:
            varCntSts02 = varCntSts02 + 1

    # Output list:
    lstOut = [idxPrc,
              aryOut,
              ]

    queOut.put(lstOut)


# %%
# this function should be used to create neural models if tuning curve models
# for motion direction are included (currently used by pRF_simulateTC.py)
def funcNrlTcTngCrv(idxPrc,
                    varPixX,
                    varPixY,
                    NrlMdlChunk,
                    varNumTP,
                    aryBoxCar,  # tngCrvBrds
                    path,
                    varNumNrlMdls,
                    varPar,
                    queOut):
    """
    Function for creating neural time course models.
    """

#    # if hd5 method is used: open file for reading
#    filename = 'aryBoxCar' + str(idxPrc) + '.hdf5'
#    hdf5_path = os.path.join(path, filename)
#    fileH = tables.openFile(hdf5_path, mode='r')

    # Output array with pRF model time courses at all modelled standard
    # deviations for current pixel position:
    aryOut = np.empty((len(NrlMdlChunk), varNumTP),
                      dtype='float32')

    # Prepare status indicator if this is the first of the parallel processes:
    if idxPrc == 1:

        # We create a status indicator for the time consuming pRF model finding
        # algorithm. Number of steps of the status indicator:
        varStsStpSze = 20

        # Number of pRF models to fit:
        varNumLoops = varNumNrlMdls/varPar

        # Vector with pRF values at which to give status feedback:
        vecStatus = np.linspace(0,
                                varNumLoops,
                                num=(varStsStpSze+1),
                                endpoint=True)
        vecStatus = np.ceil(vecStatus)
        vecStatus = vecStatus.astype(int)

        # Vector with corresponding percentage values at which to give status
        # feedback:
        vecStatusPrc = np.linspace(0,
                                   100,
                                   num=(varStsStpSze+1),
                                   endpoint=True)
        vecStatusPrc = np.ceil(vecStatusPrc)
        vecStatusPrc = vecStatusPrc.astype(int)

        # Counter for status indicator:
        varCntSts01 = 0
        varCntSts02 = 0

    # Loop through all Gauss parameters that are in this chunk
    for idx, NrlMdlQrt in enumerate(NrlMdlChunk):

        # Status indicator (only used in the first of the parallel
        # processes):
        if idxPrc == 1:

            # Status indicator:
            if varCntSts02 == vecStatus[varCntSts01]:

                # Prepare status message:
                strStsMsg = ('---------Progress: ' +
                             str(vecStatusPrc[varCntSts01]) +
                             ' % --- ' +
                             str(vecStatus[varCntSts01]) +
                             ' loops out of ' +
                             str(varNumLoops))

                print(strStsMsg)

                # Only increment counter if the last value has not been
                # reached yet:
                if varCntSts01 < varStsStpSze:
                    varCntSts01 = varCntSts01 + int(1)

        # x pos of Gauss model: NrlMdlQrt[0]
        # y pos of Gauss model: NrlMdlQrt[1]
        # std of Gauss model: NrlMdlQrt[2]
        # index of tng crv model: NrlMdlQrt[3]
        varTmpX = int(np.around(NrlMdlQrt[0], 0))
        varTmpY = int(np.around(NrlMdlQrt[1], 0))

        # Create pRF model (2D):
        aryGauss = funcGauss2D(varPixX,
                               varPixY,
                               varTmpX,
                               varTmpY,
                               NrlMdlQrt[2])

        # pull temporary box car models
        idxTngCrv = int(NrlMdlQrt[3])
        aryBoxCarTemp = aryBoxCar[:, :, idxTngCrv, :]

#        # if the hd5 method is used:
#        aryBoxCarTemp = fileH.root.aryBoxCar[
#            tngCrvBrds[idxTngCrv]:tngCrvBrds[idxTngCrv+1]
#            ].reshape(varPixX, varPixY, varNumTP)

        # Multiply pixel-wise box car model with Gaussian pRF models:
        aryNrlTcTmp = np.multiply(aryBoxCarTemp, aryGauss[:, :, None])

        # Calculate sum across x- and y-dimensions - the 'area under the
        # Gaussian surface'. This is essentially an unscaled version of the
        # neural time course model (i.e. not yet scaled for the size of
        # the pRF).
        aryNrlTcTmp = np.sum(aryNrlTcTmp, axis=(0, 1))

        # Normalise the nrl time course model to the size of the pRF. This
        # gives us the ratio of 'activation' of the pRF at each time point,
        # or, in other words, the neural time course model.
        aryNrlTcTmp = np.divide(aryNrlTcTmp,
                                np.sum(aryGauss, axis=(0, 1)))

        # Put model time courses into the function's output array:
        aryOut[idx, :] = aryNrlTcTmp

        # Status indicator (only used in the first of the parallel
        # processes):
        if idxPrc == 1:
            # Increment status indicator counter:
            varCntSts02 = varCntSts02 + 1

    # Output list:
    lstOut = [idxPrc,
              aryOut,
              ]

    queOut.put(lstOut)


# %%
def funcFindPrf(idxPrc,
                aryFuncChnk,
                aryPrfTc,
                aryMdls,
                queOut):

    """
    Function for finding best pRF model for voxel time course.
    This function should be used if there is only one predictor.
    """

    # Number of voxels to be fitted in this chunk:
    varNumVoxChnk = aryFuncChnk.shape[0]

    # Number of volumes:
    varNumVol = aryFuncChnk.shape[1]

    # Vectors for pRF finding results [number-of-voxels times one]:
    vecBstXpos = np.zeros(varNumVoxChnk)
    vecBstYpos = np.zeros(varNumVoxChnk)
    vecBstSd = np.zeros(varNumVoxChnk)
    # vecBstR2 = np.zeros(varNumVoxChnk)

    # Vector for best R-square value. For each model fit, the R-square value is
    # compared to this, and updated if it is lower than the best-fitting
    # solution so far. We initialise with an arbitrary, high value
    vecBstRes = np.add(np.zeros(varNumVoxChnk),
                       100000.0)

    # We reshape the voxel time courses, so that time goes down the column,
    # i.e. from top to bottom.
    aryFuncChnk = aryFuncChnk.T

    # Constant term for the model:
    vecConst = np.ones((varNumVol), dtype=np.float32)

    # Change type to float 32:
    aryFuncChnk = aryFuncChnk.astype(np.float32)
    aryPrfTc = aryPrfTc.astype(np.float32)

    # Number of pRF models to fit:
    varNumMdls = len(aryMdls)

    # Prepare status indicator if this is the first of the parallel processes:
    if idxPrc == 0:

        # We create a status indicator for the time consuming pRF model finding
        # algorithm. Number of steps of the status indicator:
        varStsStpSze = 20

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
    for idxMdls in range(0, varNumMdls):

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

        # Current pRF time course model:
        vecMdlTc = aryPrfTc[idxMdls, :].flatten()

        # We create a design matrix including the current pRF time
        # course model, and a constant term:
        aryDsgn = np.vstack([vecMdlTc,
                             vecConst]).T

        # Calculation of the ratio of the explained variance (R square)
        # for the current model for all voxel time courses.

#                print('------------np.linalg.lstsq on pRF: ' +
#                      str(idxX) +
#                      'x ' +
#                      str(idxY) +
#                      'y ' +
#                      str(idxSd) +
#                      'z --- START')
#                varTmeTmp01 = time.time()

        # Change type to float32:
        # aryDsgn = aryDsgn.astype(np.float32)

        # Calculate the least-squares solution for all voxels:
        vecTmpRes = np.linalg.lstsq(aryDsgn, aryFuncChnk)[1]

#                varTmeTmp02 = time.time()
#                varTmeTmp03 = np.around((varTmeTmp02 - varTmeTmp01),
#                                        decimals=2)
#                print('------------np.linalg.lstsq on pRF: ' +
#                      str(idxX) +
#                      'x ' +
#                      str(idxY) +
#                      'y ' +
#                      str(idxSd) +
#                      'z --- DONE elapsed time: ' +
#                      str(varTmeTmp03) +
#                      's')

        # Check whether current residuals are lower than previously
        # calculated ones:
        vecLgcTmpRes = np.less(vecTmpRes, vecBstRes)

        # Replace best x and y position values, and SD values.
        vecBstXpos[vecLgcTmpRes] = aryMdls[idxMdls][0]
        vecBstYpos[vecLgcTmpRes] = aryMdls[idxMdls][1]
        vecBstSd[vecLgcTmpRes] = aryMdls[idxMdls][2]

        # Replace best residual values:
        vecBstRes[vecLgcTmpRes] = vecTmpRes[vecLgcTmpRes]

#                varTmeTmp04 = time.time()
#                varTmeTmp05 = np.around((varTmeTmp04 - varTmeTmp02),
#                                        decimals=2)
#                print('------------selection of best-fitting pRF model: ' +
#                      str(idxX) +
#                      'x ' +
#                      str(idxY) +
#                      'y ' +
#                      str(idxSd) +
#                      'z --- elapsed time: ' +
#                      str(varTmeTmp05) +
#                      's')

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
              vecBstR2]

    queOut.put(lstOut)


# %%
def funcFindPrfMltpPrd(idxPrc,
                       aryFuncChnk,
                       aryPrfTc,
                       aryMdls,
                       queOut):

    """
    Function for finding best pRF model for voxel time course.
    This function should be used if there are several predictors.
    """

    # Number of voxels to be fitted in this chunk:
    varNumVoxChnk = aryFuncChnk.shape[0]

    # Number of volumes:
    varNumVol = aryFuncChnk.shape[1]

    # Vectors for pRF finding results [number-of-voxels times one]:
    vecBstXpos = np.zeros(varNumVoxChnk)
    vecBstYpos = np.zeros(varNumVoxChnk)
    vecBstSd = np.zeros(varNumVoxChnk)
    # vecBstR2 = np.zeros(varNumVoxChnk)

    # Vector for best R-square value. For each model fit, the R-square value is
    # compared to this, and updated if it is lower than the best-fitting
    # solution so far. We initialise with an arbitrary, high value
    vecBstRes = np.add(np.zeros(varNumVoxChnk),
                       100000.0)

    # We reshape the voxel time courses, so that time goes down the column,
    # i.e. from top to bottom.
    aryFuncChnk = aryFuncChnk.T

    # Constant term for the model:
    vecConst = np.ones((varNumVol), dtype=np.float32)

    # Change type to float 32:
    aryFuncChnk = aryFuncChnk.astype(np.float32)
    aryPrfTc = aryPrfTc.astype(np.float32)

    # Number of pRF models to fit:
    varNumMdls = len(aryMdls)

    # Prepare status indicator if this is the first of the parallel processes:
    if idxPrc == 0:

        # We create a status indicator for the time consuming pRF model finding
        # algorithm. Number of steps of the status indicator:
        varStsStpSze = 20

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
    for idxMdls in range(0, varNumMdls):

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

        # Current pRF time course model:
        vecMdlTc = aryPrfTc[idxMdls, :, :]

        # We create a design matrix including the current pRF time
        # course model, and a constant term:
        aryDsgn = np.vstack([vecMdlTc,
                             vecConst]).T

        # Calculation of the ratio of the explained variance (R square)
        # for the current model for all voxel time courses.

#                print('------------np.linalg.lstsq on pRF: ' +
#                      str(idxX) +
#                      'x ' +
#                      str(idxY) +
#                      'y ' +
#                      str(idxSd) +
#                      'z --- START')
#                varTmeTmp01 = time.time()

        # Change type to float32:
        # aryDsgn = aryDsgn.astype(np.float32)

        # Calculate the least-squares solution for all voxels:
        vecTmpRes = np.linalg.lstsq(aryDsgn, aryFuncChnk)[1]

#                varTmeTmp02 = time.time()
#                varTmeTmp03 = np.around((varTmeTmp02 - varTmeTmp01),
#                                        decimals=2)
#                print('------------np.linalg.lstsq on pRF: ' +
#                      str(idxX) +
#                      'x ' +
#                      str(idxY) +
#                      'y ' +
#                      str(idxSd) +
#                      'z --- DONE elapsed time: ' +
#                      str(varTmeTmp03) +
#                      's')

        # Check whether current residuals are lower than previously
        # calculated ones:
        vecLgcTmpRes = np.less(vecTmpRes, vecBstRes)

        # Replace best x and y position values, and SD values.
        vecBstXpos[vecLgcTmpRes] = aryMdls[idxMdls][0]
        vecBstYpos[vecLgcTmpRes] = aryMdls[idxMdls][1]
        vecBstSd[vecLgcTmpRes] = aryMdls[idxMdls][2]

        # Replace best residual values:
        vecBstRes[vecLgcTmpRes] = vecTmpRes[vecLgcTmpRes]

#                varTmeTmp04 = time.time()
#                varTmeTmp05 = np.around((varTmeTmp04 - varTmeTmp02),
#                                        decimals=2)
#                print('------------selection of best-fitting pRF model: ' +
#                      str(idxX) +
#                      'x ' +
#                      str(idxY) +
#                      'y ' +
#                      str(idxSd) +
#                      'z --- elapsed time: ' +
#                      str(varTmeTmp05) +
#                      's')

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
              vecBstR2]

    queOut.put(lstOut)


# %%
def funcFindPrfMltpPrdXVal(idxPrc,
                           aryFuncChnkTrn,
                           aryFuncChnkTst,
                           aryPrfMdlsTrnConv,
                           aryPrfMdlsTstConv,
                           aryMdls,
                           queOut):
    """
    Function for finding best pRF model for voxel time course.
    This function should be used if there are several predictors.
    """

    # Number of voxels to be fitted in this chunk:
    varNumVoxChnk = aryFuncChnkTrn.shape[0]

    # Number of volumes:
    varNumVolTrn = aryFuncChnkTrn.shape[2]
    varNumVolTst = aryFuncChnkTst.shape[2]

    # get number of cross validations
    varNumXval = aryPrfMdlsTrnConv.shape[2]

    # Vectors for pRF finding results [number-of-voxels times one]:
    vecBstXpos = np.zeros(varNumVoxChnk)
    vecBstYpos = np.zeros(varNumVoxChnk)
    vecBstSd = np.zeros(varNumVoxChnk)
    # vecBstR2 = np.zeros(varNumVoxChnk)

    # Vector for temporary residuals values that are obtained during
    # the different loops of cross validation
    vecTmpResXVal = np.empty((varNumVoxChnk, varNumXval), dtype='float32')

    # Vector for best residual values.
    vecBstRes = np.add(np.zeros(varNumVoxChnk),
                       100000.0)

    # Constant term for the model:
    vecConstTrn = np.ones((varNumVolTrn), dtype=np.float32)
    vecConstTst = np.ones((varNumVolTst), dtype=np.float32)

    # Change type to float 32:
    aryPrfMdlsTrnConv = aryPrfMdlsTrnConv.astype(np.float32)
    aryPrfMdlsTstConv = aryPrfMdlsTstConv.astype(np.float32)

    # Number of pRF models to fit:
    varNumMdls = len(aryMdls)

    # Prepare status indicator if this is the first of the parallel processes:
    if idxPrc == 0:

        # We create a status indicator for the time consuming pRF model finding
        # algorithm. Number of steps of the status indicator:
        varStsStpSze = 20

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
    for idxMdls in range(0, varNumMdls):

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

        # Loop through different cross validations
        for idxXval in range(0, varNumXval):
            # Current pRF time course model:
            vecMdlTrn = aryPrfMdlsTrnConv[idxMdls, :, idxXval, :]
            vecMdlTst = aryPrfMdlsTstConv[idxMdls, :, idxXval, :]

            # We create a design matrix including the current pRF time
            # course model, and a constant term:
            aryDsgnTrn = np.vstack([vecMdlTrn,
                                    vecConstTrn]).T

            aryDsgnTst = np.vstack([vecMdlTst,
                                    vecConstTst]).T

            # Calculate the least-squares solution for all voxels
            # and get parameter estimates from the training fit
            aryTmpPrmEst = np.linalg.lstsq(aryDsgnTrn,
                                           aryFuncChnkTrn[:, idxXval, :].T)[0]
            # calculate predicted model fit based on training data
            aryTmpMdlTc = np.dot(aryDsgnTst, aryTmpPrmEst)
            # calculate residual sum of squares between test data and
            # predicted model fit based on training data
            vecTmpResXVal[:, idxXval] = np.sum(
                (np.subtract(aryFuncChnkTst[:, idxXval, :].T,
                             aryTmpMdlTc))**2, axis=0)

        vecTmpRes = np.mean(vecTmpResXVal, axis=1)
        # Check whether current residuals are lower than previously
        # calculated ones:
        vecLgcTmpRes = np.less(vecTmpRes, vecBstRes)

        # Replace best x and y position values, and SD values.
        vecBstXpos[vecLgcTmpRes] = aryMdls[idxMdls][0]
        vecBstYpos[vecLgcTmpRes] = aryMdls[idxMdls][1]
        vecBstSd[vecLgcTmpRes] = aryMdls[idxMdls][2]

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
              vecBstSd,
              ]

    queOut.put(lstOut)
