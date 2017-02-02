# -*- coding: utf-8 -*-
"""functions for creating pRF model time courses"""

# Part of py_pRF_motion library
# Copyright (C) 2016  Marian Schneider, Ingo Marquardt
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

from __future__ import division
import numpy as np
import scipy as sp
import pickle
from scipy.interpolate import griddata
from scipy.stats import vonmises_line
from pRF_hrfutils import spmt, dspmt, ddspmt, cnvlTc, cnvlTcOld, createHrf
import multiprocessing as mp


def loadPng(varNumVol, tplPngSize, strPathPng):
    """
    This function loads PNG files.
    """
    print('------Load PNGs')
    # Create list of png files to load:
    lstPngPaths = [None] * varNumVol
    for idx01 in range(0, varNumVol):
        lstPngPaths[idx01] = (strPathPng + str(idx01) + '.png')

    # Load png files. The png data will be saved in a numpy array of the
    # following order: aryPngData[x-pixel, y-pixel, PngNumber]. The
    # sp.misc.imread function actually contains three values per pixel (RGB),
    # but since the stimuli are black-and-white, any one of these is sufficient
    # and we discard the others.
    aryPngData = np.zeros((tplPngSize[0],
                           tplPngSize[1],
                           varNumVol))
    for idx01 in range(0, varNumVol):
        aryPngData[:, :, idx01] = sp.misc.imread(lstPngPaths[idx01])

    # Convert RGB values (0 to 255) to integer ones and zeros:
    aryPngData = (aryPngData > 0).astype(int)

    return aryPngData


def loadPrsOrd(vecRunLngth, strPathPresOrd, vecVslStim):
    """
    This function loads presentation order of motion directions.
    """
    print('------Load presentation order of motion directions')
    aryPresOrd = np.empty((0))
    for idx01 in range(0, len(vecRunLngth)):
        # reconstruct file name
        # ---> consider: some runs were shorter than others(replace next row)
        filename1 = (strPathPresOrd + str(vecVslStim[idx01]) +
                     '.pickle')
        # filename1 = (strPathPresOrd + str(idx01+1) + '.pickle')
        # load array
        with open(filename1, 'rb') as handle:
            array1 = pickle.load(handle)
        tempCond = array1["Conditions"]
        tempCond = tempCond[:vecRunLngth[idx01], 1]
        # add temp array to aryPresOrd
        aryPresOrd = np.concatenate((aryPresOrd, tempCond), axis=0)
    aryPresOrd = aryPresOrd.astype(int)

    return aryPresOrd


def crtBoxCarWeights(lgcVonMises, lgcAoM, varNumMtDrctn, varKappa=0):
    """
    This function creates box car weights.
    If lgcVonMises is set to False, this will create weights of 0 and 1
    If lgcVonMises is set to True, this takes into account that a voxel which
    prefers rightward motion might still react to rightward and upward motion
    """
    if lgcVonMises:  # assign weights according to von mises pdf
        print('------Create box car weights from von Mises distribution')

        x = np.linspace(-np.pi, np.pi, 360)
        mus = np.arange(0, 360, 45)
        # get von Mises distributions
        aryCrvV1 = np.empty((len(x), len(mus)), dtype='float64')
        for ind, mu in enumerate(mus-180):
            aryCrvV1[:, ind] = np.roll(vonmises_line.pdf(x, varKappa), mu)
        # sample relevant indices
        vecIndV1 = np.empty((len(mus), len(mus)))
        for ind in np.arange(len(mus)):
            vecIndV1[:, ind] = aryCrvV1[mus, ind]
        # normalize
        vecIndV1 = vecIndV1 / np.sum(vecIndV1, axis=0)

        if lgcAoM:  # reduce presented motion direction from 8 to 4?
            vecIndV1 = (vecIndV1[:varNumMtDrctn, varNumMtDrctn:] +
                        vecIndV1[:varNumMtDrctn, :varNumMtDrctn])
    else:  # assume binary 0 and 1 weights
        print('------Create binary box car weights')
        vecIndV1 = np.eye(varNumMtDrctn)

    return vecIndV1


def crtPwBoxCarFn(varNumVol, aryPngData, aryPresOrd, vecIndV1, vecMtDrctn):
    """
    This function creates pixel-wise boxcar functions.
    """
    print('------Create pixel-wise boxcar functions')
    aryBoxCar = np.empty(aryPngData.shape[0:2] + (len(vecMtDrctn),) +
                         (varNumVol,), dtype='float32')
    # append a row of zeros for fixation periods
    vecIndV1 = np.append(np.zeros((1, vecIndV1.shape[1])), vecIndV1, 0)
    for ind, num in enumerate(vecMtDrctn):
        vecTmpWeights = vecIndV1[:, ind][aryPresOrd]
        mtplTempMtDrctn = aryPngData * vecTmpWeights[None, None, :]
        aryBoxCar[:, :, ind, :] = mtplTempMtDrctn
    return aryBoxCar


def cnvlPwBoxCarFn(aryBoxCar,
                   varNumVol,
                   varTr,
                   tplPngSize,
                   varPar,
                   switchHrfSet,
                   varNumMtDrctn,
                   lgcOldSchoolHrf,
                   ):
    """
    Create pixel-wise HRF model time courses.

    After concatenating all stimulus frames (png files) into an array, this
    stimulus array is effectively a boxcar design matrix with zeros if no
    stimulus was present at that pixel at that frame, and ones if a stimulus
    was present. In this function, we convolve this boxcar design matrix with
    an HRF model.
    """
    print('------Convolve every pixel box car function with hrf function(s)')

    # Create hrf time course function:
    if switchHrfSet == 3:
        lstHrf = [spmt, dspmt, ddspmt]
    elif switchHrfSet == 2:
        lstHrf = [spmt, dspmt]
    elif switchHrfSet == 1:
        lstHrf = [spmt]

    # Reshape png data:
    aryBoxCar = np.reshape(aryBoxCar,
                           ((aryBoxCar.shape[0] * aryBoxCar.shape[1] *
                            aryBoxCar.shape[2]), aryBoxCar.shape[3]))

    # Put input data into chunks:
    lstBoxCar = np.array_split(aryBoxCar, varPar)
    # We don't need the original array with the input data anymore:
    del(aryBoxCar)

    # Create a queue to put the results in:
    queOut = mp.Queue()

    # Empty list for processes:
    lstPrcs = [None] * varPar

    # Empty list for results of parallel processes:
    lstConv = [None] * varPar

    print('---------Creating parallel processes')

    if lgcOldSchoolHrf:
        for idxPrc in range(0, varPar):
            lstPrcs[idxPrc] = mp.Process(target=cnvlTcOld,
                                         args=(idxPrc,
                                               lstBoxCar[idxPrc],
                                               varTr,
                                               varNumVol,
                                               queOut)
                                         )
    else:
        # Create processes:
        for idxPrc in range(0, varPar):
            lstPrcs[idxPrc] = mp.Process(target=cnvlTc,
                                         args=(idxPrc,
                                               lstBoxCar[idxPrc],
                                               lstHrf,
                                               varTr,
                                               varNumVol,
                                               queOut)
                                         )

        # Daemon (kills processes when exiting):
        lstPrcs[idxPrc].Daemon = True

    # Start processes:
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc].start()

    # Collect results from queue:
    for idxPrc in range(0, varPar):
        lstConv[idxPrc] = queOut.get(True)

    # Join processes:
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc].join()

    print('---------Collecting results from parallel processes')

    # Put output into correct order:
    lstConv = sorted(lstConv)

    # Concatenate convolved pixel time courses (into the same order as they
    # were entered into the analysis):
    aryBoxCarConv = np.zeros((0, switchHrfSet, varNumVol))
    for idxRes in range(0, varPar):
        aryBoxCarConv = np.concatenate((aryBoxCarConv, lstConv[idxRes][1]),
                                       axis=0)
    del(lstConv)

    # Reshape results:
    aryBoxCarConv = np.reshape(aryBoxCarConv,
                               [tplPngSize[0],
                                tplPngSize[1],
                                varNumMtDrctn,
                                varNumVol])

    # Return:
    return aryBoxCarConv


def rsmplInHighRes(aryBoxCarConv,
                   tplPngSize,
                   tplVslSpcHighSze,
                   varNumMtDrctn,
                   varNumVol):
    """
    Resample pixel-time courses in high-res visual space.

    The Gaussian sampling of the pixel-time courses takes place in the
    super-sampled visual space. Here we take the convolved pixel-time courses
    into this space, for each time point (volume).
    """

    print('------Resample pixel-time courses in high-res visual space')

    # Array for super-sampled pixel-time courses:
    aryBoxCarConvHigh = np.zeros((tplVslSpcHighSze[0],
                                  tplVslSpcHighSze[1],
                                  varNumMtDrctn,
                                  varNumVol))

    # Loop through volumes:
    for idxMtn in range(0, varNumMtDrctn):

        for idxVol in range(0, varNumVol):

            # Range for the coordinates:
            vecRange = np.arange(0, tplPngSize[0])

            # The following array describes the coordinates of the pixels in
            # the flattened array (i.e. "vecOrigPixVal"). In other words, these
            # are the row and column coordinates of the original pizel values.
            crd2, crd1 = np.meshgrid(vecRange, vecRange)
            aryOrixPixCoo = np.column_stack((crd1.flatten(), crd2.flatten()))

            # The following vector will contain the actual original pixel
            # values:

            vecOrigPixVal = aryBoxCarConv[:, :, idxMtn, idxVol]
            vecOrigPixVal = vecOrigPixVal.flatten()

            # The sampling interval for the creation of the super-sampled pixel
            # data (complex numbers are used as a convention for inclusive
            # intervals in "np.mgrid()").:

            varStpSzeX = np.complex(tplVslSpcHighSze[0])
            varStpSzeY = np.complex(tplVslSpcHighSze[1])

            # The following grid has the coordinates of the points at which we
            # would like to re-sample the pixel data:
            aryPixGridX, aryPixGridY = np.mgrid[0:tplPngSize[0]:varStpSzeX,
                                                0:tplPngSize[1]:varStpSzeY]

            # The actual resampling:
            aryResampled = griddata(aryOrixPixCoo,
                                    vecOrigPixVal,
                                    (aryPixGridX, aryPixGridY),
                                    method='nearest')

            # Put super-sampled pixel time courses into array:
            aryBoxCarConvHigh[:, :, idxMtn, idxVol] = aryResampled

    return aryBoxCarConvHigh


def func2DGaussHrf(varSizeX,
                   varSizeY,
                   varPosX,
                   varPosY,
                   varSd,
                   switchHrfSet,
                   varTr):
    """Create 3D kernel: 2D Gaussian with hrf in 3rd dimension"""
    varPosX = 50
    varPosY = 50
    hrf = np.ones(16)
    varSizeX = 128
    varSizeY = 128
    varSd = 20
    switchHrfSet = 1

    varSizeX = int(varSizeX)
    varSizeY = int(varSizeY)

    # Create hrf time course function:
    if switchHrfSet == 3:
        lstHrf = [spmt, dspmt, ddspmt]
    elif switchHrfSet == 2:
        lstHrf = [spmt, dspmt]
    elif switchHrfSet == 1:
        lstHrf = [spmt]

    # get hrf shapes
    lstBse = createHrf(lstHrf, varTr, varOvsmpl=1)

    aryY, aryX, aryT = np.meshgrid(np.arange(varSizeX),
                                   np.arange(varSizeY),
                                   np.arange(len(lstBse[0])),)

    # Step 1: calculate spatial Gaussian
    # this is a 2D Gaussian that moves with every movie frame at speed v

    # normalization for spatial gaussian
    a = 1 / (2*varSd**2*np.pi)
    # denominator spatial Gaussian
    b = 1 / (2*varSd**2)
    # calculate the Gaussian
    ary2DGauss = a * np.exp(
        -b * (np.square(aryX-varPosX) + np.square(aryY-varPosY))
        )

    # Step 2: build a 3D array that has an hrf shape in the time direction
    hrf = lstBse[0]
    aryHrf = np.multiply(aryT, hrf[None, None, :])
    # Step 3: combine
    aryGauss3D = ary2DGauss * aryHrf

    return aryGauss3D

def cnvlWith3DGauss(aryBoxCar, aryGauss3D):
    aryBoxCar
    temp = aryBoxCar[:, :, 0, :]
    cnvlPwBoxCarFn

    from scipy import signal


def funcGauss(varSizeX,
              varSizeY,
              varPosX,
              varPosY,
              varSd):
    """Create 2D Gaussian kernel."""
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

    return aryGauss


def funcPrfTc(idxPrc,
              aryMdlParamsChnk,
              tplVslSpcHighSze,
              varNumVol,
              aryPngDataHigh,
              queOut):
    """Create pRF time course models."""
    # Number of combinations of model parameters in the current chunk:
    varChnkSze = np.size(aryMdlParamsChnk, axis=0)

    # Determine number of motion directions
    varNumMtnDrtn = aryPngDataHigh.shape[2]

    # Output array with pRF model time courses:
    aryOut = np.zeros([varChnkSze, varNumMtnDrtn, varNumVol])

    # Loop through different motion directions:
    for idxMtn in range(0, varNumMtnDrtn):
        # Loop through combinations of model parameters:
        for idxMdl in range(0, varChnkSze):

            # Depending on the relation between the number of x- and y-pos
            # at which to create pRF models and the size of the super-sampled
            # visual space, the indicies need to be rounded:
            varTmpX = np.around(aryMdlParamsChnk[idxMdl, 1], 0)
            varTmpY = np.around(aryMdlParamsChnk[idxMdl, 2], 0)
            varTmpSd = np.around(aryMdlParamsChnk[idxMdl, 3], 0)

            # Create pRF model (2D):
            aryGauss = funcGauss(tplVslSpcHighSze[0],
                                 tplVslSpcHighSze[1],
                                 varTmpX,
                                 varTmpY,
                                 varTmpSd)

            # Multiply super-sampled pixel-time courses with Gaussian pRF
            # models:
            aryPrfTcTmp = np.multiply(aryPngDataHigh[:, :, idxMtn, :],
                                      aryGauss[:, :, None])

            # Calculate sum across x- and y-dimensions - the 'area under the
            # Gaussian surface'. This is essentially an unscaled version of the
            # pRF time course model (i.e. not yet scaled for size of the pRF).
            aryPrfTcTmp = np.sum(aryPrfTcTmp, axis=(0, 1))

            # Normalise the pRF time course model to the size of the pRF. This
            # gives us the ratio of 'activation' of the pRF at each time point,
            # or, in other words, the pRF time course model.
            aryPrfTcTmp = np.divide(aryPrfTcTmp,
                                    np.sum(aryGauss, axis=(0, 1)))

            # Put model time courses into the function's output array:
            aryOut[idxMdl, idxMtn, :] = aryPrfTcTmp

    # Put column with the indicies of model-parameter-combinations into the
    # output array (in order to be able to put the pRF model time courses into
    # the correct order after the parallelised function):
    lstOut = [idxPrc,
              aryOut]

    # Put output to queue:
    queOut.put(lstOut)