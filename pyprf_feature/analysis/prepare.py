# -*- coding: utf-8 -*-
"""Preparation of fMRI data and pRF model time courses."""

# Part of py_pRF_mapping library
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
from scipy.ndimage.filters import gaussian_filter1d
from pyprf_feature.analysis.utils_general import load_nii


def prep_models(aryPrfTc, varSdSmthTmp=2.0, lgcPrint=True):
    """
    Prepare pRF model time courses.

    Parameters
    ----------
    aryPrfTc : np.array
        4D numpy array with pRF time course models, with following dimensions:
        `aryPrfTc[x-position, y-position, SD, volume]`.
    varSdSmthTmp : float
        Extent of temporal smoothing that is applied to functional data and
        pRF time course models, [SD of Gaussian kernel, in seconds]. If `zero`,
        no temporal smoothing is applied.
    lgcPrint : boolean
        Whether print statements should be executed.

    Returns
    -------
    aryPrfTc : np.array
        4D numpy array with prepared pRF time course models, same
        dimensions as input (`aryPrfTc[x-position, y-position, SD, volume]`).
    """
    if lgcPrint:
        print('------Prepare pRF time course models')

    # Define temporal smoothing of pRF time course models
    def funcSmthTmp(aryPrfTc, varSdSmthTmp, lgcPrint=True):
        """Apply temporal smoothing to fMRI data & pRF time course models.

        Parameters
        ----------
        aryPrfTc : np.array
            4D numpy array with pRF time course models, with following
            dimensions: `aryPrfTc[x-position, y-position, SD, volume]`.
        varSdSmthTmp : float, positive
            Extent of temporal smoothing that is applied to functional data and
            pRF time course models, [SD of Gaussian kernel, in seconds]. If
            `zero`, no temporal smoothing is applied.
        lgcPrint : boolean
            Whether print statements should be executed.

        Returns
        -------
        aryPrfTc : np.array
            4D numpy array with prepared pRF time course models, same dimension
            as input (`aryPrfTc[x-position, y-position, SD, volume]`).
        """

        # adjust the input, if necessary, such that input is 2D, with last
        # dim time
        tplInpShp = aryPrfTc.shape
        aryPrfTc = aryPrfTc.reshape((-1, aryPrfTc.shape[-1]))

        # For the filtering to perform well at the ends of the time series, we
        # set the method to 'nearest' and place a volume with mean intensity
        # (over time) at the beginning and at the end.
        aryPrfTcMean = np.mean(aryPrfTc, axis=-1, keepdims=True).reshape(-1, 1)

        aryPrfTc = np.concatenate((aryPrfTcMean, aryPrfTc, aryPrfTcMean),
                                  axis=-1)

        # In the input data, time goes from left to right. Therefore, we apply
        # the filter along axis=1.
        aryPrfTc = gaussian_filter1d(aryPrfTc.astype('float32'), varSdSmthTmp,
                                     axis=-1, order=0, mode='nearest',
                                     truncate=4.0)

        # Remove mean-intensity volumes at the beginning and at the end:
        aryPrfTc = aryPrfTc[..., 1:-1]

        # Output array:
        return aryPrfTc.reshape(tplInpShp).astype('float16')

    # Perform temporal smoothing of pRF time course models
    if 0.0 < varSdSmthTmp:
        if lgcPrint:
            print('---------Temporal smoothing on pRF time course models')
            print('------------SD tmp smooth is: ' + str(varSdSmthTmp))
        aryPrfTc = funcSmthTmp(aryPrfTc, varSdSmthTmp)

    # Z-score the prf time course models
    if lgcPrint:
        print('---------Zscore the pRF time course models')
    # De-mean the prf time course models:
    aryPrfTc = np.subtract(aryPrfTc, np.mean(aryPrfTc, axis=-1)[..., None])

    # Standardize the prf time course models:
    # In order to avoid devision by zero, only divide those voxels with a
    # standard deviation greater than zero:
    aryTmpStd = np.std(aryPrfTc, axis=-1)
    aryTmpLgc = np.greater(aryTmpStd, np.array([0.0]))
    aryPrfTc[aryTmpLgc, :] = np.divide(aryPrfTc[aryTmpLgc, :],
                                       aryTmpStd[aryTmpLgc, None])

    return aryPrfTc


def prep_func(strPathNiiMask, lstPathNiiFunc, varAvgThr=100.,
              varVarThr=0.0001, strPrePro='demean'):
    """
    Load & prepare functional data.

    Parameters
    ----------
    strPathNiiMask: str
        Path to mask used to restrict pRF model finding. Only voxels with
        a value greater than zero in the mask are considered.
    lstPathNiiFunc : list
        List of paths of functional data (nii files).
    varAvgThr : float, positive, default = 100.
        Float. Voxels that have at least one run with a mean lower than this
        (before demeaning) will be excluded from model fitting.
    varVarThr : float, positive, default = 0.0001
        Float. Voxels that have at least one run with a variance lower than
        this (after demeaning) will be excluded from model fitting.
    strPrePro : string, default 'demean'
        Preprocessing that will be applied to the data.
        By default they are demeaned.

    Returns
    -------
    aryLgcMsk : np.array
        3D numpy array with logial values. Externally supplied mask (e.g grey
        matter mask). Voxels that are `False` in the mask are excluded.
    vecLgcIncl : np.array
        1D numpy array containing logical values. One value per voxel after
        mask has been applied. If `True`, the variance and mean of the voxel's
        time course are greater than the provided thresholds in all runs and
        the voxel is included in the output array (`aryFunc`). If `False`, the
        variance or mean of the voxel's time course is lower than threshold in
        at least one run and the voxel has been excluded from the output
        (`aryFunc`). This is to avoid problems in the subsequent model fitting.
        This array is necessary to put results into original dimensions after
        model fitting.
    hdrMsk : nibabel-header-object
        Nii header of mask.
    aryAff : np.array
        Array containing 'affine', i.e. information about spatial positioning
        of mask nii data.
    aryFunc : np.array
        2D numpy array containing prepared functional data, of the form
        aryFunc[voxelCount, time].
    tplNiiShp : tuple
        Spatial dimensions of input nii data (number of voxels in x, y, z
        direction). The data are reshaped during preparation, this
        information is needed to fit final output into original spatial
        dimensions.

    Notes
    -----
    Functional data is loaded from disk. The functional data is reshaped, into
    the form aryFunc[voxel, time]. A mask is applied (externally supplied, e.g.
    a grey matter mask). Subsequently, the functional data is de-meaned.
    """
    print('------Load & prepare nii data')

    # Load mask (to restrict model fitting):
    aryMask, hdrMsk, aryAff = load_nii(strPathNiiMask)

    # Mask is loaded as float32, but is better represented as integer:
    aryMask = np.array(aryMask).astype(np.int16)

    # Number of non-zero voxels in mask:
    # varNumVoxMsk = int(np.count_nonzero(aryMask))

    # Dimensions of nii data:
    tplNiiShp = aryMask.shape

    # Total number of voxels:
    varNumVoxTlt = (tplNiiShp[0] * tplNiiShp[1] * tplNiiShp[2])

    # Reshape mask:
    aryMask = np.reshape(aryMask, varNumVoxTlt)

    # List for arrays with functional data (possibly several runs):
    lstFunc = []

    # List for averages of the individual runs (before demeaning)
    lstFuncAvg = []

    # List for variances of the individual runs (after demeaning)
    lstFuncVar = []

    # Number of runs:
    varNumRun = len(lstPathNiiFunc)

    # Loop through runs and load data:
    for idxRun in range(varNumRun):

        print(('---------Prepare run ' + str(idxRun + 1)))

        # Load 4D nii data:
        aryTmpFunc, _, _ = load_nii(lstPathNiiFunc[idxRun])

        # Dimensions of nii data (including temporal dimension; spatial
        # dimensions need to be the same for mask & functional data):
        tplNiiShp = aryTmpFunc.shape

        # Reshape functional nii data, from now on of the form
        # aryTmpFunc[voxelCount, time]:
        aryTmpFunc = np.reshape(aryTmpFunc, [varNumVoxTlt, tplNiiShp[3]])

        # Apply mask:
        print('------------Mask')
        aryLgcMsk = np.greater(aryMask.astype(np.int16),
                               np.array([0], dtype=np.int16)[0])
        aryTmpFunc = aryTmpFunc[aryLgcMsk, :]

        # save the mean of the run
        lstFuncAvg.append(np.mean(aryTmpFunc, axis=1, dtype=np.float32))

        # also save the variance of the run
        lstFuncVar.append(np.var(aryTmpFunc, axis=1, dtype=np.float32))

        # De-mean functional data:
        if strPrePro == 'demean':
            print('------------Demean')
            aryTmpFunc = np.subtract(aryTmpFunc,
                                     np.mean(aryTmpFunc,
                                             axis=1,
                                             dtype=np.float32)[:, None])
        elif strPrePro == 'zscore':
            print('------------Zscore')
            aryTmpFunc = np.subtract(aryTmpFunc,
                                     np.mean(aryTmpFunc,
                                             axis=1,
                                             dtype=np.float32)[:, None])

            # Standardize the data time courses:
            # In order to avoid devision by zero, only divide
            # those voxels with a standard deviation greater
            # than zero:
            aryTmpStd = np.std(aryTmpFunc, axis=-1)
            aryTmpLgc = np.greater(aryTmpStd, np.array([0.0]))
            aryTmpFunc[aryTmpLgc, :] = np.divide(aryTmpFunc[aryTmpLgc, :],
                                                 aryTmpStd[aryTmpLgc, None])

        # Put prepared functional data of current run into list:
        lstFunc.append(aryTmpFunc)
        del(aryTmpFunc)

    # Put functional data from separate runs into one array. 2D array of the
    # form aryFunc[voxelCount, time]
    aryFunc = np.concatenate(lstFunc, axis=1).astype(np.float32, copy=False)
    del(lstFunc)

    # Put the averages (before demeaning) from the separate runs into one
    # array. 2D array of the form aryFuncVar[voxelCount, nr of runs]
    aryFuncAvg = np.stack(lstFuncAvg, axis=1).astype(np.float32, copy=False)
    del(lstFuncAvg)

    # Put the variance (after demeaning) from the separate runs into one array.
    # 2D array of the form aryFuncVar[voxelCount, nr of runs]
    aryFuncVar = np.stack(lstFuncVar, axis=1).astype(np.float32, copy=False)
    del(lstFuncVar)

    # Especially if data were recorded in different sessions, there can
    # sometimes be voxels that have close to zero signal in runs from one
    # session but regular signal in the runs from another session. These voxels
    # are very few, are located at the edge of the functional and can cause
    # problems during model fitting. They are therefore excluded.

    # Is the mean greater than threshold?
    aryLgcAvg = np.greater(aryFuncAvg,
                           np.array([varAvgThr]).astype(np.float32)[0])
    # Mean needs to be greater than threshold in every single run
    vecLgcAvg = np.all(aryLgcAvg, axis=1)

    # Voxels that are outside the brain and have no, or very little, signal
    # should not be included in the pRF model finding. We take the variance
    # over time and exclude voxels with a suspiciously low variance, if they
    # have low variance in at least one run. Because the data given into the
    # cython or GPU function has float32 precision, we calculate the variance
    # on data with float32 precision.

    # Is the variance greater than threshold?
    aryLgcVar = np.greater(aryFuncVar,
                           np.array([varVarThr]).astype(np.float32)[0])
    # Variance needs to be greater than threshold in every single run
    vecLgcVar = np.all(aryLgcVar, axis=1)

    # Are there any nan values in the functional time series?
    vecLgcNan = np.invert(np.any(np.isnan(aryFunc), axis=1))

    # combine the logical vectors for exclusion resulting from low variance and
    # low mean signal time course
    vecLgcIncl = np.logical_and(vecLgcAvg, vecLgcVar)

    # combine logical vectors for mean/variance with vector for nan exclsion
    vecLgcIncl = np.logical_and(vecLgcIncl, vecLgcNan)

    # Array with functional data for which conditions (mask inclusion and
    # cutoff value) are fullfilled:
    aryFunc = aryFunc[vecLgcIncl, :]

    # print info about the exclusion of voxels
    print('---------Minimum mean threshold for voxels applied at: ' +
          str(varAvgThr))
    print('---------Minimum variance threshold for voxels applied at:  ' +
          str(varVarThr))
    print('---------Number of voxels excluded due to low mean or variance: ' +
          str(np.sum(np.invert(vecLgcIncl))))

    return aryLgcMsk, vecLgcIncl, hdrMsk, aryAff, aryFunc, tplNiiShp
