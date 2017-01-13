# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 11:42:53 2017

@author: marian
"""

import numpy as np
import scipy as sp
import pickle
from scipy.interpolate import griddata
from pRF_hrfutils import spmt, dspmt, ddspmt, cnvlTc
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


def crtPwBoxCarFn(varNumVol, aryPngData, aryPresOrd, vecMtDrctn):
    """
    This function creates pixel-wise boxcar functions.
    """
    print('------Create pixel-wise boxcar functions')
    aryBoxCar = np.empty(aryPngData.shape[0:2] + (len(vecMtDrctn),) +
                         (varNumVol,), dtype='int64')
    for ind, num in enumerate(vecMtDrctn):
        aryCondTemp = np.zeros((aryPngData.shape), dtype='int64')
        lgcTempMtDrctn = [aryPresOrd == num][0]
        aryCondTemp[:, :, lgcTempMtDrctn] = np.copy(
            aryPngData[:, :, lgcTempMtDrctn])
        aryBoxCar[:, :, ind, :] = aryCondTemp

    return aryBoxCar


def cnvlPwBoxCarFn(aryBoxCar,
                   varNumVol,
                   varTr,
                   tplPngSize,
                   varPar,
                   switchHrfSet,
                   varNumMtDrctn,
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
