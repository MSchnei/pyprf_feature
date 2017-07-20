# -*- coding: utf-8 -*-
"""function for creating pRF model time courses"""

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

import numpy as np
import scipy as sp
import pickle
from PIL import Image
from scipy.interpolate import griddata
from pRF_hrfutils import spmt, dspmt, ddspmt, cnvlTc, cnvlTcOld
import multiprocessing as mp


def loadPng(varNumVol, tplPngSize, strPathPng):
    """This function loads PNG files.
    Parameters
    ----------
    varNumVol : float
        Number of volumes, i.e. number of time points in all runs.
    tplPngSize : tuple
        Shape of the stimulus image (i.e. png).
    strPathPng: str
        Path to the folder cointaining the png files.
    Returns
    -------
    aryPngData : 2d numpy array, shape [png_x, png_y, n_vols]
        Stack of stimulus data.

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
        aryPngData[:, :, idx01] = np.array(Image.open(lstPngPaths[idx01]))

    # Convert RGB values (0 to 255) to integer ones and zeros:
    aryPngData = (aryPngData > 0).astype(int)

    return aryPngData


def loadPrsOrd(vecRunLngth, strPathPresOrd, vecVslStim):
    """This function loads presentation order of motion directions.
    Parameters
    ----------
    vecRunLngth : list
        Number of volumes in every run
    strPathPresOrd : str
        Path to the npy vector containing order of presented motion directions.
    vecVslStim: list
        Key of (stimulus) condition presented in every run
    Returns
    -------
    aryPresOrdAprt : 1d numpy array, shape [n_vols]
        Presentation order of aperture position.
    aryPresOrdMtn : 1d numpy array, shape [n_vols]
        Presentation order of motion direction.
    """

    print('------Load presentation order of motion directions')
    aryPresOrd = np.empty((0, 2))
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
        tempCond = tempCond[:vecRunLngth[idx01], :]
        # add temp array to aryPresOrd
        aryPresOrd = np.concatenate((aryPresOrd, tempCond), axis=0)
    aryPresOrdAprt = aryPresOrd[:, 0].astype(int)
    aryPresOrdMtn = aryPresOrd[:, 1].astype(int)

    return aryPresOrdAprt, aryPresOrdMtn


def crtPwBoxCarFn(varNumVol, aryPngData, aryPresOrd, vecMtDrctn):
    """This function creates pixel-wise boxcar functions.
    Parameters
    ----------
    input1 : 2d numpy array, shape [n_samples, n_measurements]
        Description of input 1.
    input2 : float, positive
      Description of input 2.
    Returns
    -------
    data : 2d numpy array, shape [n_samples, n_measurements]
        Closed data.
    Reference
    ---------
    [1]
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


def crtGauss2D(varSizeX, varSizeY, varPosX, varPosY, varSd):
    """Create 2D Gaussian kernel.
    Parameters
    ----------
    varSizeX : int, positive
        Width of the visual field.
    varSizeY : int, positive
        Height of the visual field..
    varPosX : int, positive
        X position of centre of 2D Gauss.
    varPosY : int, positive
        Y position of centre of 2D Gauss.
    varSd : float, positive
        Standard deviation of 2D Gauss.
    Returns
    -------
    aryGauss : 2d numpy array, shape [varSizeX, varSizeY]
        2d Gaussian.
    Reference
    ---------
    [1]
    """
    varSizeX = int(varSizeX)
    varSizeY = int(varSizeY)

    # aryX and aryY are in reversed order, this seems to be necessary:
    aryY, aryX = sp.mgrid[0:varSizeX,
                          0:varSizeY]

    # The actual creation of the Gaussian array:
    aryGauss = (
        (np.square((aryX - varPosX)) + np.square((aryY - varPosY))) /
        (2.0 * np.square(varSd))
        )
    aryGauss = np.exp(-aryGauss) / (2 * np.pi * np.square(varSd))

    return aryGauss


def cnvlGauss2D(idxPrc, aryBoxCar, aryMdlParamsChnk, tplPngSize, varNumVol,
                queOut):
    """Spatially convolve boxcar functions with 2D Gaussian.
    Parameters
    ----------
    idxPrc : 2d numpy array, shape [n_samples, n_measurements]
        Description of input 1.
    aryBoxCar : float, positive
      Description of input 2.
    aryMdlParamsChnk : 2d numpy array, shape [n_samples, n_measurements]
        Description of input 1.
    tplPngSize : float, positive
      Description of input 2.
    varNumVol : 2d numpy array, shape [n_samples, n_measurements]
        Description of input 1.
    queOut : float, positive
      Description of input 2.
    Returns
    -------
    data : 2d numpy array, shape [n_samples, n_measurements]
        Closed data.
    Reference
    ---------
    [1]
    """
    # Number of combinations of model parameters in the current chunk:
    varChnkSze = np.size(aryMdlParamsChnk, axis=0)

    # Determine number of motion directions
    varNumMtnDrtn = aryBoxCar.shape[2]

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
            aryGauss = crtGauss2D(tplPngSize[0],
                                  tplPngSize[1],
                                  varTmpX,
                                  varTmpY,
                                  varTmpSd)

            # Multiply pixel-time courses with Gaussian pRF models:
            aryPrfTcTmp = np.multiply(aryBoxCar[:, :, idxMtn, :],
                                      aryGauss[:, :, None])

            # Calculate sum across x- and y-dimensions - the 'area under the
            # Gaussian surface'. This is essentially an unscaled version of the
            # pRF time course model (i.e. not yet scaled for size of the pRF).
            aryPrfTcTmp = np.sum(aryPrfTcTmp, axis=(0, 1))

            # Put model time courses into function's output with 2d Gaussian
            # arrray:
            aryOut[idxMdl, idxMtn, :] = aryPrfTcTmp

    # Put column with the indicies of model-parameter-combinations into the
    # output array (in order to be able to put the pRF model time courses into
    # the correct order after the parallelised function):
    lstOut = [idxPrc,
              aryOut]

    # Put output to queue:
    queOut.put(lstOut)


def crtPrfNrlTc(aryBoxCar, varNumMtDrctn, varNumVol, tplPngSize, varNumX,
                varExtXmin,  varExtXmax, varNumY, varExtYmin, varExtYmax,
                varNumPrfSizes, varPrfStdMin, varPrfStdMax, varPar):
    """Create neural model time courses from pixel-wise boxcar functions.
    Parameters
    ----------
    aryBoxCar : 4d numpy array, shape [n_x_pix, n_y_pix, n_mtn_dir, n_vol]
        Description of input 1.
    varNumMtDrctn : float, positive
        Description of input 2.
    varNumVol : float, positive
        Description of input 2.
    tplPngSize : tuple
        Description of input 2.
    varNumX : float, positive
        Description of input 2.
    varExtXmin : float, positive
        Description of input 2.
    varExtXmax : float, positive
        Description of input 2.
    varNumY : float, positive
        Description of input 2.
    varExtYmin : float, positive
        Description of input 2.
    varExtYmax : float, positive
        Description of input 2.
    varNumPrfSizes : float, positive
        Description of input 2.
    varPrfStdMin : float, positive
        Description of input 2.
    varPrfStdMax : float, positive
        Description of input 2.
    varPar : float, positive
        Description of input 2.
    Returns
    -------
    aryNrlTc : 5d numpy array, shape [n_x_pos, n_y_pos, n_sd, n_mtn_dir, n_vol]
        Closed data.
    Reference
    ---------
    [1]
    """
    print('------Create neural time course models')

    # Vector with the x-indicies of the positions in the super-sampled visual
    # space at which to create pRF models.
    vecX = np.linspace(0, (tplPngSize[0] - 1), varNumX, endpoint=True)

    # Vector with the y-indicies of the positions in the super-sampled visual
    # space at which to create pRF models.
    vecY = np.linspace(0, (tplPngSize[1] - 1), varNumY, endpoint=True)

    # We calculate the scaling factor from degrees of visual angle to pixels
    # separately for the x- and the y-directions (the two should be the same).
    varDgr2PixX = tplPngSize[0] / (varExtXmax - varExtXmin)
    varDgr2PixY = tplPngSize[1] / (varExtYmax - varExtYmin)

    # Check whether varDgr2PixX and varDgr2PixY are similar:
    strErrMsg = 'ERROR. The ratio of X and Y dimensions in stimulus ' + \
        'space (in degrees of visual angle) and the ratio of X and Y ' + \
        'dimensions in the upsampled visual space do not agree'
    assert 0.5 > np.absolute((varDgr2PixX - varDgr2PixY)), strErrMsg

    # Vector with pRF sizes to be modelled (still in degree of visual angle):
    vecPrfSd = np.linspace(varPrfStdMin, varPrfStdMax, varNumPrfSizes,
                           endpoint=True)

    # We multiply the vector containing pRF sizes with the scaling factors.
    # Now the vector with the pRF sizes can be used directly for creation of
    # Gaussian pRF models in visual space.
    vecPrfSd = np.multiply(vecPrfSd, varDgr2PixX)

    # Number of pRF models to be created (i.e. number of possible combinations
    # of x-position, y-position, and standard deviation):
    varNumMdls = varNumX * varNumY * varNumPrfSizes

    # Array for the x-position, y-position, and standard deviations for which
    # pRF model time courses are going to be created, where the columns
    # correspond to: (0) an index starting from zero, (1) the x-position, (2)
    # the y-position, and (3) the standard deviation. The parameters are in
    # units of the upsampled visual space.
    aryMdlParams = np.zeros((varNumMdls, 4))

    # Counter for parameter array:
    varCntMdlPrms = 0

    # Put all combinations of x-position, y-position, and standard deviations
    # into the array:

    # Loop through x-positions:
    for idxX in range(0, varNumX):

        # Loop through y-positions:
        for idxY in range(0, varNumY):

            # Loop through standard deviations (of Gaussian pRF models):
            for idxSd in range(0, varNumPrfSizes):

                # Place index and parameters in array:
                aryMdlParams[varCntMdlPrms, 0] = varCntMdlPrms
                aryMdlParams[varCntMdlPrms, 1] = vecX[idxX]
                aryMdlParams[varCntMdlPrms, 2] = vecY[idxY]
                aryMdlParams[varCntMdlPrms, 3] = vecPrfSd[idxSd]

                # Increment parameter index:
                varCntMdlPrms = varCntMdlPrms + 1

    # The long array with all the combinations of model parameters is put into
    # separate chunks for parallelisation, using a list of arrays.
    lstMdlParams = np.array_split(aryMdlParams, varPar)

    # Create a queue to put the results in:
    queOut = mp.Queue()

    # Empty list for results from parallel processes (for pRF model time course
    # results):
    lstPrfTc = [None] * varPar

    # Empty list for processes:
    lstPrcs = [None] * varPar

    print('---------Creating parallel processes')

    # Create processes:
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc] = mp.Process(target=cnvlGauss2D,
                                     args=(idxPrc, aryBoxCar,
                                           lstMdlParams[idxPrc], tplPngSize,
                                           varNumVol, queOut)
                                     )
        # Daemon (kills processes when exiting):
        lstPrcs[idxPrc].Daemon = True

    # Start processes:
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc].start()

    # Collect results from queue:
    for idxPrc in range(0, varPar):
        lstPrfTc[idxPrc] = queOut.get(True)

    # Join processes:
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc].join()

    print('---------Collecting results from parallel processes')
    # Put output arrays from parallel process into one big array
    lstPrfTc = sorted(lstPrfTc)
    aryPrfTc = np.empty((0, varNumMtDrctn, varNumVol))
    for idx in range(0, varPar):
        aryPrfTc = np.concatenate((aryPrfTc, lstPrfTc[idx][1]), axis=0)

    # check that all the models were collected correctly
    assert aryPrfTc.shape[0] == varNumMdls

    # Clean up:
    del(aryMdlParams)
    del(lstMdlParams)
    del(lstPrfTc)

    # Array representing the low-resolution visual space, of the form
    # aryPrfTc[x-position, y-position, pRF-size, varNum Vol], which will hold
    # the pRF model time courses.
    aryNrlTc = np.zeros([varNumX, varNumY, varNumPrfSizes, varNumMtDrctn,
                         varNumVol])

    # We use the same loop structure for organising the pRF model time courses
    # that we used for creating the parameter array. Counter:
    varCntMdlPrms = 0

    # Put all combinations of x-position, y-position, and standard deviations
    # into the array:

    # Loop through x-positions:
    for idxX in range(0, varNumX):

        # Loop through y-positions:
        for idxY in range(0, varNumY):

            # Loop through standard deviations (of Gaussian pRF models):
            for idxSd in range(0, varNumPrfSizes):

                # Put the pRF model time course into its correct position in
                # the 4D array, leaving out the first column (which contains
                # the index):
                aryNrlTc[idxX, idxY, idxSd, :, :] = aryPrfTc[
                    varCntMdlPrms, :, :]

                # Increment parameter index:
                varCntMdlPrms = varCntMdlPrms + 1

    return aryNrlTc


def cnvlPwBoxCarFn(aryNrlTc, varNumVol, varTr, tplPngSize, varNumMtDrctn,
                   switchHrfSet, lgcOldSchoolHrf, varPar,):
    """ Create 2D Gaussian kernel.
    Parameters
    ----------
    aryNrlTc : 5d numpy array, shape [n_x_pos, n_y_pos, n_sd, n_mtn_dir, n_vol]
        Description of input 1.
    varNumVol : float, positive
        Description of input 2.
    varTr : float, positive
        Description of input 1.
    tplPngSize : tuple
        Description of input 1.
    varNumMtDrctn : int, positive
        Description of input 1.
    switchHrfSet :
        Description of input 1.
    lgcOldSchoolHrf : int, positive
        Description of input 1.
    varPar : int, positive
        Description of input 1.
    Returns
    -------
    data : 2d numpy array, shape [n_samples, n_measurements]
        Closed data.
    Reference
    ---------
    [1]
    """
    print('------Convolve every pixel box car function with hrf function(s)')

    # Create hrf time course function:
    if switchHrfSet == 3:
        lstHrf = [spmt, dspmt, ddspmt]
    elif switchHrfSet == 2:
        lstHrf = [spmt, dspmt]
    elif switchHrfSet == 1:
        lstHrf = [spmt]

    # adjust the input, if necessary, such that input is 2D, with last dim time
    tplInpShp = aryNrlTc.shape
    aryNrlTc = np.reshape(aryNrlTc, (-1, aryNrlTc.shape[-1]))

    # Put input data into chunks:
    lstNrlTc = np.array_split(aryNrlTc, varPar)

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
                                               lstNrlTc[idxPrc],
                                               varTr,
                                               varNumVol,
                                               queOut)
                                         )
            # Daemon (kills processes when exiting):
            lstPrcs[idxPrc].Daemon = True

    else:
        # Create processes:
        for idxPrc in range(0, varPar):
            lstPrcs[idxPrc] = mp.Process(target=cnvlTc,
                                         args=(idxPrc,
                                               lstNrlTc[idxPrc],
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
    # Concatenate convolved pixel time courses (into the same order
    aryNrlTcConv = np.zeros((0, switchHrfSet, varNumVol))
    for idxRes in range(0, varPar):
        aryNrlTcConv = np.concatenate((aryNrlTcConv, lstConv[idxRes][1]),
                                      axis=0)
    # clean up
    del(aryNrlTc)
    del(lstConv)

    # Reshape results:
    tplOutShp = tplInpShp[:-2] + (varNumMtDrctn * len(lstHrf), ) + \
        (tplInpShp[-1], )

    # Return:
    return np.reshape(aryNrlTcConv, tplOutShp)


def rsmplInHighRes(aryBoxCarConv,
                   tplPngSize,
                   tplVslSpcHighSze,
                   varNumMtDrctn,
                   varNumVol):
    """ Resample pixel-time courses in high-res visual space.
    Parameters
    ----------
    input1 : 2d numpy array, shape [n_samples, n_measurements]
        Description of input 1.
    input2 : float, positive
      Description of input 2.
    Returns
    -------
    data : 2d numpy array, shape [n_samples, n_measurements]
        Closed data.
    Reference
    ---------
    [1]
    """

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
