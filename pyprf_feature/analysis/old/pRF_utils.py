# -*- coding: utf-8 -*-

"""Basic utility functions.""" 

# Part of py_pRF_motion library
# Copyright (C) 2016  Marian Schneider
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

import os
import numpy as np
import nibabel as nb
from scipy import stats


# %% Load voxel responses

def loadNiiData(lstNiiFls,
                strPathNiiMask=None,
                strPathNiiFunc=None):
    """load nii data.

        Parameters
        ----------
        lstNiiFls : list, list of str with nii file names
        strPathNiiMask : str, path to nii file with mask (optional)
        strPathNiiFunc : str, parent path to nii files (optional)
        Returns
        -------
        aryFunc : np.array
            Nii data   
    """
    print('---------Loading nii data')
    # check whether  a mask is available
    if strPathNiiMask is not None:
        aryMask = nb.load(strPathNiiMask).get_data().astype('bool')
    # check a parent path is available that needs to be preprended to nii files
    if strPathNiiFunc is not None:
        lstNiiFls = [os.path.join(strPathNiiFunc, i) for i in lstNiiFls]

    aryFunc = []
    for idx, path in enumerate(lstNiiFls):
        print('------------Loading run: ' + str(idx+1))
        # Load 4D nii data:
        niiFunc = nb.load(path).get_data()
        # append to list
        if strPathNiiMask is not None:
            aryFunc.append(niiFunc[aryMask, :])
        else:
            aryFunc.append(niiFunc)
    # concatenate arrys in list along time dimension
    aryFunc = np.concatenate(aryFunc, axis=-1)
    # set to type float32
    aryFunc = aryFunc.astype('float32')

    return aryFunc


def saveNiiData(data,
                objNii,
                fileNameNii,
                aryMask=None):

    if aryMask is not None:
        # save R2 as nii
        aryData = np.zeros((objNii.shape))
        aryData[aryMask] = data
    else:
        aryData = data

    niiOut = nb.Nifti1Image(aryData,
                            objNii.affine,
                            objNii.header
                            )
    # Save nii:
    nb.save(niiOut, fileNameNii)


def calcR2(predTst, yTest, axis=0):
    """calculate coefficient of determination. Assumes that axis=0 is time

        Parameters
        ----------
        predTst : np.array, predicted reponse for yTest
        yTest : np.array, acxtually observed response for yTest
        Returns
        -------
        aryFunc : np.array
            R2
    """
    rss = np.sum((yTest - predTst) ** 2, axis=axis)
    tss = np.sum((yTest - yTest.mean()) ** 2, axis=axis)

    return 1 - rss/tss


def calcFstats(predTst, yTest, p, axis=0):
    """calculate coefficient of determination. Assumes that axis=0 is time

        Parameters
        ----------
        predTst : np.array, predicted reponse for yTest
        yTest : np.array, acxtually observed response for yTest
        p: float, number of predictors
        Returns
        -------
        aryFunc : np.array
            R2
    """
    rss = np.sum((yTest - predTst) ** 2, axis=axis)
    tss = np.sum((yTest - yTest.mean()) ** 2, axis=axis)
    # derive number of measurements
    n = yTest.shape[0]
    # calculate Fvalues
    vecFvals = ((tss - rss)/p)/(rss/(n-p-1))
    # calculate corresponding po values
    df1 = p - 1
    df2 = n-1
    vecPvals = stats.f.cdf(vecFvals, df1, df2)

    return vecFvals, vecPvals


def calcMse(predTst, yTest, axis=0):
    """calculate mean squared error. Assumes that axis=0 is time

        Parameters
        ----------
        predTst : np.array, predicted reponse for yTest
        yTest : np.array, acxtually observed response for yTest
        Returns
        -------
        aryFunc : np.array
            MSE
    """
    return np.mean((yTest - predTst) ** 2, axis=axis)
