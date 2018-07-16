#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Perform least-squares fitting with cross-validation using numpy"""

import numpy as np


def np_lst_sq(vecMdl, aryFuncChnk):
    """Least squares fitting in numpy without cross-validation.

    Notes
    -----
    This is just a wrapper function for np.linalg.lstsq to keep piping
    consistent.

    """
    aryTmpBts, vecTmpRes = np.linalg.lstsq(vecMdl,
                                           aryFuncChnk,
                                           rcond=-1)[:2]

    return aryTmpBts, vecTmpRes


def np_lst_sq_xval(vecMdl, aryFuncChnk, aryIdxTrn, aryIdxTst):
    """Least squares fitting in numpy with cross-validation.
    """

    varNumXval = aryIdxTrn.shape[-1]

    varNumVoxChnk = aryFuncChnk.shape[-1]

    # pre-allocate ary to collect cross-validation
    # error for every xval fold
    aryResXval = np.empty((varNumVoxChnk,
                           varNumXval),
                          dtype=np.float32)

    # loop over cross-validation folds
    for idxXval in range(varNumXval):
        # Get pRF time course models for trn and tst:
        vecMdlTrn = vecMdl[aryIdxTrn[:, idxXval], :]
        vecMdlTst = vecMdl[aryIdxTst[:, idxXval], :]
        # Get functional data for trn and tst:
        aryFuncChnkTrn = aryFuncChnk[
            aryIdxTrn[:, idxXval], :]
        aryFuncChnkTst = aryFuncChnk[
            aryIdxTst[:, idxXval], :]

        # Numpy linalg.lstsq is used to calculate the
        # parameter estimates of the current model:
        vecTmpPe = np.linalg.lstsq(vecMdlTrn,
                                   aryFuncChnkTrn,
                                   rcond=-1)[0]

        # calculate model prediction time course
        aryMdlPrdTc = np.dot(vecMdlTst, vecTmpPe)

        # calculate residual sum of squares between
        # test data and model prediction time course
        aryResXval[:, idxXval] = np.sum(
            (np.subtract(aryFuncChnkTst,
                         aryMdlPrdTc))**2, axis=0)

    return aryResXval
