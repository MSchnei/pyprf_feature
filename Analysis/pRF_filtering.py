# -*- coding: utf-8 -*-
"""Main function for preprocessing of data & models"""

# Part of py_pRF_motion library
# Copyright (C) 2016  Ingo Marquardt, Marian Schneider
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
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import gaussian_filter1d


# %% Spatial smoothing of fMRI data
def funcSmthSpt(aryFuncChnk, varSdSmthSpt):
    """
    Apply spatial smoothing to the input data.

    The extent of smoothing needs to be specified as an input parameter.
    """
    varNdim = aryFuncChnk.ndim

    # Number of time points in this chunk:
    varNumVol = aryFuncChnk.shape[-1]

    # Loop through volumes:
    if varNdim == 4:
        for idxVol in range(0, varNumVol):

            aryFuncChnk[:, :, :, idxVol] = gaussian_filter(
                aryFuncChnk[:, :, :, idxVol],
                varSdSmthSpt,
                order=0,
                mode='nearest',
                truncate=4.0)
    elif varNdim == 5:
        varNumMtnDrctns = aryFuncChnk.shape[3]
        for idxVol in range(0, varNumVol):
            for idxMtn in range(0, varNumMtnDrctns):
                aryFuncChnk[:, :, :, idxMtn, idxVol] = gaussian_filter(
                    aryFuncChnk[:, :, :, idxMtn, idxVol],
                    varSdSmthSpt,
                    order=0,
                    mode='nearest',
                    truncate=4.0)

    # Output list:
    return aryFuncChnk


# %% Temporal smoothing of fMRI data & pRF time course models
def funcSmthTmp(aryFuncChnk, varSdSmthTmp):
    """
    Apply temporal smoothing to the input data.

    The extend of smoothing needs to be specified as an input parameter.
    """
    # For the filtering to perform well at the ends of the time series, we
    # set the method to 'nearest' and place a volume with mean intensity
    # (over time) at the beginning and at the end.
    aryFuncChnkMean = np.mean(aryFuncChnk,
                              axis=1,
                              keepdims=True)

    aryFuncChnk = np.concatenate((aryFuncChnkMean,
                                  aryFuncChnk,
                                  aryFuncChnkMean), axis=1)

    # In the input data, time goes from left to right. Therefore, we apply
    # the filter along axis=1.
    aryFuncChnk = gaussian_filter1d(aryFuncChnk,
                                    varSdSmthTmp,
                                    axis=1,
                                    order=0,
                                    mode='nearest',
                                    truncate=4.0)

    # Remove mean-intensity volumes at the beginning and at the end:
    aryFuncChnk = aryFuncChnk[:, 1:-1]

    # Output list:
    return aryFuncChnk


# *************************************************************************
