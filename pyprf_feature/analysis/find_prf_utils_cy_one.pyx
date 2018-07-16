# -*- coding: utf-8 -*-
"""Cythonised least squares GLM model fitting with 1 predictor."""

# Part of pyprf_feature library
# Copyright (C) 2018  Omer Faruk Gulban & Ingo Marquardt & Marian Schneider
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


# *****************************************************************************
# *** Import modules & adjust cython settings for speedup

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport pow, sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
# *****************************************************************************


# *****************************************************************************
# *** Main function least squares solution, no cross-validation, 1 predictor

cpdef tuple cy_lst_sq_one(
    np.ndarray[np.float32_t, ndim=1] vecPrfTc,
    np.ndarray[np.float32_t, ndim=2] aryFuncChnk):
    """
    Cythonised least squares GLM model fitting.

    Parameters
    ----------
    vecPrfTc : np.array
        1D numpy array, at float32 precision, containing a single pRF model
        time course (along time dimension).
    aryFuncChnk : np.array
        2D numpy array, at float32 precision, containing a chunk of functional
        data (i.e. voxel time courses). Dimensionality: aryFuncChnk[time,
        voxel].

    Returns
    -------
    vecRes : np.array
        1D numpy array with model residuals for all voxels in the chunk of
        functional data. Dimensionality: vecRes[voxel]

    Notes
    -----
    Computes the least-squares solution for the model fit between the pRF time
    course model, and all voxel time courses. Assumes removal of the mean from
    the functional data and the model. Needs to be compiled before execution
    (see `cython_leastsquares_setup.py`).
    """

    cdef:
        float varVarY = 0
        float[:] vecPrfTc_view = vecPrfTc
        unsigned long varNumVoxChnk, idxVox
        unsigned int idxVol, varNumVols

    # Number of voxels in the input data chunk:
    varNumVoxChnk = int(aryFuncChnk.shape[1])

    # Define 1D array for results (i.e. for residuals of least squares
    # solution):
    cdef np.ndarray[np.float32_t, ndim=1] vecRes = np.zeros(varNumVoxChnk,
                                                            dtype=np.float32)

    cdef np.ndarray[np.float32_t, ndim=1] vecPe = np.zeros(varNumVoxChnk,
                                                           dtype=np.float32)
    # Memory view on array for results:
    cdef float[:] vecRes_view = vecRes

    # Memory view on array for parameter estimates:
    cdef float[:] vecPe_view = vecPe

    # Memory view on numpy array with functional data:
    cdef float [:, :] aryFuncChnk_view = aryFuncChnk

    # Calculate variance of pRF model time course (i.e. variance in the model):
    varNumVols = int(vecPrfTc.shape[0])
    for idxVol in range(varNumVols):
        varVarY += vecPrfTc_view[idxVol] ** 2

    # Call optimised cdef function for calculation of residuals:
    vecRes_view, vecPe_view = func_cy_res(vecPrfTc_view,
                                          aryFuncChnk_view,
                                          vecRes_view,
                                          vecPe_view,
                                          varNumVoxChnk,
                                          varNumVols,
                                          varVarY)

    # Convert memory view to numpy array before returning it:
    vecRes = np.asarray(vecRes_view)
    vecPe = np.asarray(vecPe_view)

    return vecPe.reshape(1, -1), vecRes
# *****************************************************************************

# *****************************************************************************
# *** Function fast calculation residuals, no cross-validation, 1 predictor

cdef (float[:], float[:]) func_cy_res(float[:] vecPrfTc_view,
                                      float[:, :] aryFuncChnk_view,
                                      float[:] vecRes_view,
                                      float[:] vecPe_view,
                                      unsigned long varNumVoxChnk,
                                      unsigned int varNumVols,
                                      float varVarY):

    cdef:
        float varCovXy, varRes, varSlope, varXhat
        unsigned int idxVol
        unsigned long idxVox

    # Loop through voxels:
    for idxVox in range(varNumVoxChnk):

        # Covariance and residuals of current voxel:
        varCovXy = 0
        varRes = 0

        # Loop through volumes and calculate covariance between the model and
        # the current voxel:
        for idxVol in range(varNumVols):
            varCovXy += (aryFuncChnk_view[idxVol, idxVox]
                         * vecPrfTc_view[idxVol])
        # Obtain the slope of the regression of the model on the data:
        varSlope = varCovXy / varVarY

        # Loop through volumes again in order to calculate the error in the
        # prediction:
        for idxVol in range(varNumVols):
            # The predicted voxel time course value:
            varXhat = vecPrfTc_view[idxVol] * varSlope
            # Mismatch between prediction and actual voxel value (variance):
            varRes += (aryFuncChnk_view[idxVol, idxVox] - varXhat) ** 2

        vecRes_view[idxVox] = varRes
        vecPe_view[idxVox] = varSlope

    # Return memory view:
    return vecRes_view, vecPe_view


# *****************************************************************************

# *****************************************************************************
# *** Main function least squares solution, with cross-validation, 1 predictor

cpdef np.ndarray[np.float32_t, ndim=2] cy_lst_sq_xval_one(
    np.ndarray[np.float32_t, ndim=1] vecPrfTc,
    np.ndarray[np.float32_t, ndim=2] aryFuncChnk,
    np.ndarray[np.int32_t, ndim=2] aryIdxTrn,
    np.ndarray[np.int32_t, ndim=2] aryIdxTst
    ):
    """
    Cythonised least squares GLM model fitting with cross validation.

    Parameters
    ----------
    vecPrfTc : np.array
        1D numpy array, at float32 precision, containing a single pRF model
        time course (along time dimension).
    aryFuncChnk : np.array
        2D numpy array, at float32 precision, containing a chunk of functional
        data (i.e. voxel time courses). Dimensionality: aryFuncChnk[time,
        voxel].
    aryIdxTrn : np.array
        2D numpy array, at int32 precision, containing a trainings indices for
        cross-validation.
    aryIdxTst : np.array
        2D numpy array, at int32 precision, containing a test indices for
        cross-validation.

    Returns
    -------
    aryResXval : np.array
        2D numpy array with cross validation error for all voxels in the chunk of
        functional data and all cross validation folds.
        Dimensionality: aryResXval[voxel, varNumXval]

    Notes
    -----
    Computes the least-squares solution for the model fit between the pRF time
    course model, and all voxel time courses with k-fold cross validation.
    Assumes removal of the mean from the functional data and the model.
    Needs to be compiled before execution (see `cython_leastsquares_setup.py`).
    """
    cdef:
        float[:] vecPrfTc_view = vecPrfTc
        float [:, :] aryFuncChnk_view = aryFuncChnk
        int [:, :] aryIdxTrn_view = aryIdxTrn
        int [:, :] aryIdxTst_view = aryIdxTst
        unsigned long varNumVoxChnk, idxVox
        unsigned int idxVol, idxXval, varNumXval, varNumVolTrn, varNumVolTst
        int[:] vecIdxTrn

    # Number of voxels in the input data chunk:
    varNumVoxChnk = int(aryFuncChnk.shape[1])
    # Number of cross-validations:
    varNumXval = int(aryIdxTrn.shape[1])
    # Number of training volumes
    varNumVolTrn = int(aryIdxTrn.shape[0])
    # Number of testing volumes
    varNumVolTst = int(aryIdxTst.shape[0])

    # Define 2D array for residuals (here crossvalidation error) of least
    # squares solution), initialized with all zeros here:
    cdef np.ndarray[np.float32_t, ndim=2] aryResXval = np.zeros((varNumVoxChnk,
                                                                 varNumXval),
                                                                dtype=np.float32)

    # Memory view on array for residuals (here crossvalidation error)
    cdef float[:, :] aryResXval_view = aryResXval

    # Define 1D array for variances in training model time courses across folds,
    # initialized with all zeros here
    cdef np.ndarray[np.float32_t, ndim=1] vecVarY = np.zeros(varNumXval,
                                                             dtype=np.float32)
    # Memory view on array for variances in training model time courses:
    cdef float[:] vecVarY_view = vecVarY

    # Calculate variance of training pRF model time course (i.e. variance in
    # the model) - separately for every fold:
    for idxXval in range(varNumXval):
        # get vector with volumes for training
        vecIdxTrn = aryIdxTrn_view[:, idxXval]
        for idxVol in vecIdxTrn:
            vecVarY_view[idxXval] += vecPrfTc_view[idxVol] ** 2

    # Call optimised cdef function for calculation of residuals:
    aryResXval_view = func_cy_res_xval(vecPrfTc_view,
                                       aryFuncChnk_view,
                                       aryIdxTrn_view,
                                       aryIdxTst_view,
                                       aryResXval_view,
                                       varNumXval,
                                       varNumVoxChnk,
                                       varNumVolTrn,
                                       varNumVolTst,
                                       vecVarY_view)

    # Convert memory view to numpy array before returning it:
    aryResXval = np.asarray(aryResXval_view)

    return aryResXval

# *****************************************************************************

# *****************************************************************************
# *** Function fast calculation residuals, with cross-validation, 1 predictor

cdef float[:, :] func_cy_res_xval(float[:] vecPrfTc_view,
                                  float[:, :] aryFuncChnk_view,
                                  int[:, :] aryIdxTrn_view,
                                  int[:, :] aryIdxTst_view,
                                  float[:, :] aryResXval_view,
                                  unsigned int varNumXval,
                                  unsigned long varNumVoxChnk,
                                  unsigned int varNumVolTrn,
                                  unsigned int varNumVolTst,
                                  float[:] vecVarY_view):

    cdef:
        float varVarY, varCovXy, varRes, varSlope, varXhat
        unsigned int idxVol, idxXval, idxItr
        unsigned long idxVox

    # Loop through cross-validations
    for idxXval in range(varNumXval):

        # Loop through voxels:
        for idxVox in range(varNumVoxChnk):

            # Covariance and residuals of current voxel:
            varCovXy = 0
            varRes = 0

            # Loop through trainings volumes and calculate covariance between
            # the training model and the current voxel:
            for idxItr in range(varNumVolTrn):
                # get the training volume
                idxVol = aryIdxTrn_view[idxItr, idxXval]
                # calculate covariance
                varCovXy += (aryFuncChnk_view[idxVol, idxVox]
                             * vecPrfTc_view[idxVol])
            # Get the variance of the training model time courses for this fold
            varVarY = vecVarY_view[idxXval]
            # Obtain the slope of the regression of the model on the data:
            varSlope = varCovXy / varVarY

            # Loop through test volumes and calculate the predicted time course
            # value and the mismatch between prediction and actual voxel value
            for idxItr in range(varNumVolTst):
                # get the test volume
                idxVol = aryIdxTst_view[idxItr, idxXval]
                # The predicted voxel time course value:
                varXhat = vecPrfTc_view[idxVol] * varSlope
                # Mismatch between prediction and actual voxel value (variance):
                varRes += (aryFuncChnk_view[idxVol, idxVox] - varXhat) ** 2

            aryResXval_view[idxVox, idxXval] = varRes

    # Return memory view
    return aryResXval_view

# *****************************************************************************
