# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 11:42:53 2017

@author: marian
"""

import numpy as np
import scipy as sp
from scipy.stats import gamma
import pickle
import pRF_hrfutils as hrf
import multiprocessing as mp
from pRF_utilities import funcHrf, funcConvPar

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




from scipy.interpolate import interp1d
def cnvlTc(idxPrc,
           aryBoxCarChunk,
           lstHrf,
           varTr,
           varNumVol,
           varOvsmpl=10,
           varHrfLen=32,
           queOut):
    """
    Convolution of time courses with HRF model.
    """

    # *** prepare hrf time courses for convolution
    print("---------Prepare hrf time courses for convolution")
    # get frame times, i.e. start point of every volume in seconds
    vecFrms = np.arange(0, varTr * varNumVol, varTr)
    # get supersampled frames times, i.e. start point of every volume in
    # seconds, since convolution takes place in temp. upsampled space
    vecFrmTms = np.arange(0, varTr * varNumVol, varTr / varOvsmpl)
    # get resolution of supersampled frame times
    varRes = varTr / float(varOvsmpl)

    # prepare empty list that will contain the arrays with hrf time courses
    lstBse = []
    for hrfFn in lstHrf:
        # needs to be a multiple of oversample
        vecTmpBse = hrfFn(np.linspace(0, varHrfLen,
                                      (varHrfLen // varTr) * varOvsmpl))
        lstBse.append(vecTmpBse)

    # *** prepare pixel time courses for convolution
    print("---------Prepare pixel time courses for convolution")

    # adjust the input, if necessary, such that input is 2D, with last dim time
    tplInpShp = aryBoxCarChunk.shape
    aryBoxCarChunk = aryBoxCarChunk.reshape((-1, aryBoxCarChunk.shape[-1]))

    # Prepare an empty array for ouput
    aryConv = np.zeros((aryBoxCarChunk.shape[0], len(lstHrf),
                        aryBoxCarChunk.shape[1]))

    print("---------Convolve")
    # Each time course is convolved with the HRF separately, because the
    # numpy convolution function can only be used on one-dimensional data.
    # Thus, we have to loop through time courses:
    for idxTc in range(0, aryConv.shape[0]):

        # Extract the current time course:
        vecTc = aryBoxCarChunk[idxTc, :]

        # upsample the pixel time course, so that it matches the hrf time crs
        vecTcUps = np.zeros(int(varNumVol * varTr/varRes))
        vecOns = vecFrms[vecTc.astype(bool)]
        vecInd = np.round(vecOns / varRes).astype(np.int)
        vecTcUps[vecInd] = 1.

        # *** convolve
        for indBase, base in enumerate(lstBse):
            # perform the convolution
            col = np.convolve(base, vecTcUps, mode='full')[:vecTcUps.size]
            # get function for downsampling
            f = interp1d(vecFrmTms, col)
            # downsample to original space and assign to ary
            aryConv[idxTc, indBase, :] = f(vecFrms)

    # determine output shape
    varNumBase
    tplOutShp = tplInpShp[:-1] + (len(lstHrf), ) + (tplInpShp[-1], )

    # Create list containing the convolved timecourses, and the process ID:
    lstOut = [idxPrc,
              aryConv.reshape(tplOutShp)]

    # Put output to queue:
    queOut.put(lstOut)


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
        lstHrf = [hrf.spmt, hrf.dspmt, hrf.ddspmt]
    elif switchHrfSet == 2:
        lstHrf = [hrf.spmt, hrf.dspmt]
    elif switchHrfSet == 1:
        lstHrf = [hrf.spmt]

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
                                varNumMtDrctn * switchHrfSet,
                                varNumVol])

    # Return:
    return aryBoxCarConv
