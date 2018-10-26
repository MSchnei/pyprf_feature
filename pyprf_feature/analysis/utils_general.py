# -*- coding: utf-8 -*-
"""General functions supporting pRF fitting."""

# Part of pyprf_feature library
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

import os
import numpy as np
import scipy as sp
import nibabel as nb


def load_nii(strPathIn, varSzeThr=5000.0):
    """
    Load nii file.

    Parameters
    ----------
    strPathIn : str
        Path to nii file to load.
    varSzeThr : float
        If the nii file is larger than this threshold (in MB), the file is
        loaded volume-by-volume in order to prevent memory overflow. Default
        threshold is 1000 MB.

    Returns
    -------
    aryNii : np.array
        Array containing nii data. 32 bit floating point precision.
    objHdr : header object
        Header of nii file.
    aryAff : np.array
        Array containing 'affine', i.e. information about spatial positioning
        of nii data.

    Notes
    -----
    If the nii file is larger than the specified threshold (`varSzeThr`), the
    file is loaded volume-by-volume in order to prevent memory overflow. The
    reason for this is that nibabel imports data at float64 precision, which
    can lead to a memory overflow even for relatively small files.
    """
    # Load nii file (this does not load the data into memory yet):
    objNii = nb.load(strPathIn)

    # Get size of nii file:
    varNiiSze = os.path.getsize(strPathIn)

    # Convert to MB:
    varNiiSze = np.divide(float(varNiiSze), 1000000.0)

    # Load volume-by-volume or all at once, depending on file size:
    if np.greater(varNiiSze, float(varSzeThr)):

        # Load large nii file

        print(('---------Large file size ('
              + str(np.around(varNiiSze))
              + ' MB), reading volume-by-volume'))

        # Get image dimensions:
        tplSze = objNii.shape

        # Create empty array for nii data:
        aryNii = np.zeros(tplSze, dtype=np.float32)

        # Loop through volumes:
        for idxVol in range(tplSze[3]):
            aryNii[..., idxVol] = np.asarray(
                  objNii.dataobj[..., idxVol]).astype(np.float32)

    else:

        # Load small nii file

        # Load nii file (this doesn't load the data into memory yet):
        objNii = nb.load(strPathIn)

        # Load data into array:
        aryNii = np.asarray(objNii.dataobj).astype(np.float32)

    # Get headers:
    objHdr = objNii.header

    # Get 'affine':
    aryAff = objNii.affine

    # Output nii data (as numpy array), header, and 'affine':
    return aryNii, objHdr, aryAff


def load_res_prm(lstFunc, lstFlsMsk=None):
    """Load result parameters from multiple nii files, with optional mask.

    Parameters
    ----------
    lstFunc : list,
        list of str with file names of 3D or 4D nii files
    lstFlsMsk : list, optional
        list of str with paths to 3D nii files that can act as mask/s
    Returns
    -------
    lstPrmAry : list
        The list will contain as many numpy arrays as masks were provided.
        Each array is 2D with shape [nr voxel in mask, nr nii files in lstFunc]
    objHdr : header object
        Header of nii file.
    aryAff : np.array
        Array containing 'affine', i.e. information about spatial positioning
        of nii data.

    """

    # load parameter/functional maps into a list
    lstPrm = []
    for ind, path in enumerate(lstFunc):
        aryFnc = load_nii(path)[0].astype(np.float32)
        if aryFnc.ndim == 3:
            lstPrm.append(aryFnc)
        # handle cases where nii array is 4D, in this case split arrays up in
        # 3D arrays and appenbd those
        elif aryFnc.ndim == 4:
            for indAx in range(aryFnc.shape[-1]):
                lstPrm.append(aryFnc[..., indAx])

    # load mask/s if available
    if lstFlsMsk is not None:
        lstMsk = [None] * len(lstFlsMsk)
        for ind, path in enumerate(lstFlsMsk):
            aryMsk = load_nii(path)[0].astype(np.bool)
            lstMsk[ind] = aryMsk
    else:
        print('------------No masks were provided')

    if lstFlsMsk is None:
        # if no mask was provided we just flatten all parameter array in list
        # and return resulting list
        lstPrmAry = [ary.flatten() for ary in lstPrm]
    else:
        # if masks are available, we loop over masks and then over parameter
        # maps to extract selected voxels and parameters
        lstPrmAry = [None] * len(lstFlsMsk)
        for indLst, aryMsk in enumerate(lstMsk):
            # prepare array that will hold parameter values of selected voxels
            aryPrmSel = np.empty((np.sum(aryMsk), len(lstPrm)),
                                 dtype=np.float32)
            # loop over different parameter maps
            for indAry, aryPrm in enumerate(lstPrm):
                # get voxels specific to this mask
                aryPrmSel[:, indAry] = aryPrm[aryMsk, ...]
            # put array away in list, if only one parameter map was provided
            # the output will be squeezed
            lstPrmAry[indLst] = aryPrmSel

    # also get header object and affine array
    # we simply take it for the first functional nii file, cause that is the
    # only file that has to be provided by necessity
    objHdr, aryAff = load_nii(lstFunc[0])[1:]

    return lstPrmAry, objHdr, aryAff


def export_nii(ary2dNii, lstNiiNames, aryLgcMsk, aryLgcVar, tplNiiShp, aryAff,
               hdrMsk, outFormat='3D'):
    """
    Export nii file(s).

    Parameters
    ----------
    ary2dNii : numpy array
        Numpy array with results to be exported to nii.
    lstNiiNames : list
        List that contains strings with the complete file names.
    aryLgcMsk : numpy array
        If the nii file is larger than this threshold (in MB), the file is
        loaded volume-by-volume in order to prevent memory overflow. Default
        threshold is 1000 MB.
    aryLgcVar : np.array
        1D numpy array containing logical values. One value per voxel after
        mask has been applied. If `True`, the variance and mean of the voxel's
        time course are greater than the provided thresholds in all runs and
        the voxel is included in the output array (`aryFunc`). If `False`, the
        variance or mean of the voxel's time course is lower than threshold in
        at least one run and the voxel has been excluded from the output
        (`aryFunc`). This is to avoid problems in the subsequent model fitting.
        This array is necessary to put results into original dimensions after
        model fitting.
    tplNiiShp : tuple
        Tuple that describes the 3D shape of the output volume
    aryAff : np.array
        Array containing 'affine', i.e. information about spatial positioning
        of nii data.
    hdrMsk : nibabel-header-object
        Nii header of mask.
    outFormat : string, either '3D' or '4D'
        String specifying whether images will be saved as seperate 3D nii
        files or one 4D nii file

    Notes
    -----
    [1] This function does not return any arrays but instead saves to disk.
    [2] Depending on whether outFormat is '3D' or '4D' images will be saved as
        seperate 3D nii files or one 4D nii file.
    """

    # Number of voxels that were included in the mask:
    varNumVoxMsk = np.sum(aryLgcMsk)

    # Number of maps in ary2dNii
    varNumMaps = ary2dNii.shape[-1]

    # Place voxels based on low-variance exlusion:
    aryPrfRes01 = np.zeros((varNumVoxMsk, varNumMaps), dtype=np.float32)
    for indMap in range(varNumMaps):
        aryPrfRes01[aryLgcVar, indMap] = ary2dNii[:, indMap]

    # Total number of voxels:
    varNumVoxTlt = (tplNiiShp[0] * tplNiiShp[1] * tplNiiShp[2])

    # Place voxels based on mask-exclusion:
    aryPrfRes02 = np.zeros((varNumVoxTlt, aryPrfRes01.shape[-1]),
                           dtype=np.float32)
    for indDim in range(aryPrfRes01.shape[-1]):
        aryPrfRes02[aryLgcMsk, indDim] = aryPrfRes01[:, indDim]

    # Reshape pRF finding results into original image dimensions:
    aryPrfRes = np.reshape(aryPrfRes02,
                           [tplNiiShp[0],
                            tplNiiShp[1],
                            tplNiiShp[2],
                            aryPrfRes01.shape[-1]])

    if outFormat == '3D':
        # Save nii results:
        for idxOut in range(0, aryPrfRes.shape[-1]):
            # Create nii object for results:
            niiOut = nb.Nifti1Image(aryPrfRes[..., idxOut],
                                    aryAff,
                                    header=hdrMsk
                                    )
            # Save nii:
            strTmp = lstNiiNames[idxOut]
            nb.save(niiOut, strTmp)

    elif outFormat == '4D':

        # adjust header
        hdrMsk.set_data_shape(aryPrfRes.shape)

        # Create nii object for results:
        niiOut = nb.Nifti1Image(aryPrfRes,
                                aryAff,
                                header=hdrMsk
                                )
        # Save nii:
        strTmp = lstNiiNames[0]
        nb.save(niiOut, strTmp)


def joinRes(lstPrfRes, varPar, idxPos, inFormat='1D'):
    """Join results from different processing units (here cores).

    Parameters
    ----------
    lstPrfRes : list
        Output of results from parallelization.
    varPar : integer, positive
        Number of cores that were used during parallelization
    idxPos : integer, positive
        List position index that we expect the results to be collected to have.
    inFormat : string
        Specifies whether input will be 1d or 2d.

    Returns
    -------
    aryOut : numpy array
        Numpy array with results collected from different cores

    """

    if inFormat == '1D':
        # initialize output array
        aryOut = np.zeros((0,))
        # gather arrays from different processing units
        for idxRes in range(0, varPar):
            aryOut = np.append(aryOut, lstPrfRes[idxRes][idxPos])

    elif inFormat == '2D':
        # initialize output array
        aryOut = np.zeros((0, lstPrfRes[0][idxPos].shape[-1]))
        # gather arrays from different processing units
        for idxRes in range(0, varPar):
            aryOut = np.concatenate((aryOut, lstPrfRes[idxRes][idxPos]),
                                    axis=0)

    return aryOut


def cmp_res_R2(lstRat, lstNiiNames, strPathOut, posR2=4, lgcDel=False):
    """"Compare results for different exponents and create winner nii.

    Parameters
    ----------
    lstRat : list
        List of floats containing the ratios that were tested for surround
        suppression.
    lstNiiNames : list
        List of names of the different pRF maps (e.g. xpos, ypos, SD)
    strPathOut : string
        Path to the parent directory where the results should be saved.
    posR2 : integer, position index
        Position index of the R2 map. Index in the list with nii names.
    lgcDel : boolean
        Should inbetween results (in form of nii files) be deleted?

    Notes
    -----
    [1] This function does not return any arrays but instead saves to disk.

    """

    print('---Compare results for different ratios')

    # Get the names of the nii files with inbetween results
    lstCmpRes = []
    for indRat in range(len(lstRat)):
        # Get strExpSve
        strExpSve = '_' + str(lstRat[indRat])
        # If ratio is marked with 0, set empty string to find reults.
        # This is the code for fitting without a surround.
        if lstRat[indRat] == 0:
            strExpSve = ''
        # Create full path names from nii file names and output path
        lstPthNames = [strPathOut + strNii + strExpSve + '.nii.gz' for
                       strNii in lstNiiNames]
        # Append list to list that contains nii names for all exponents
        lstCmpRes.append(lstPthNames)

    print('------Find ratio that yielded highest R2 per voxel')

    # Initialize winner R2 maps
    aryWnrR2 = np.zeros(nb.load(lstCmpRes[0][0]).shape)
    aryRatMap = np.zeros(nb.load(lstCmpRes[0][0]).shape)

    # Loop over R2 maps to establish which exponents wins
    for indRat, lstMaps in zip(lstRat, lstCmpRes):
        # Load R2 map for this particular exponent
        aryTmpR2 = load_nii(lstMaps[posR2])[0]
        # Get logical that tells us where current R2 map is greater than
        # previous ones
        aryLgcTmpRes = np.greater(aryTmpR2, aryWnrR2)
        # Replace values of R2, where current R2 map was greater
        aryWnrR2[aryLgcTmpRes] = np.copy(aryTmpR2[aryLgcTmpRes])
        # Remember the index of the exponent that gave rise to this new R2
        aryRatMap[aryLgcTmpRes] = indRat

    # Initialize list with winner maps
    lstRatMap = []
    for strPthMaps in lstCmpRes[-1]:
        lstRatMap.append(np.zeros(nb.load(strPthMaps).shape))

    # Compose other maps by assigning map from exponent that was greatest for
    # every voxel
    for indRat, lstMaps in zip(lstRat, lstCmpRes):
        # Find out where this exponent won in terms of R2
        lgcWinnerMap = [aryRatMap == indRat][0]
        # Loop over all the maps
        for indMap, _ in enumerate(lstMaps):
            # Load map for this particular ratio
            aryTmpMap = load_nii(lstMaps[indMap])[0]
            # Load current winner map from array
            aryCrrWnrMap = np.copy(lstRatMap[indMap])
            # Assign values in temporary map to current winner map for voxels
            # where this ratio won
            aryCrrWnrMap[lgcWinnerMap] = np.copy(aryTmpMap[lgcWinnerMap])
            lstRatMap[indMap] = aryCrrWnrMap

    print('------Export results as nii')

    # Save winner maps as nii files
    # Get header and affine array
    hdrMsk, aryAff = load_nii(lstMaps[posR2])[1:]
    # Loop over all the maps
    for indMap, aryMap in enumerate(lstRatMap):
        # Create nii object for results:
        niiOut = nb.Nifti1Image(aryMap,
                                aryAff,
                                header=hdrMsk
                                )
        # Save nii:
        strTmp = strPathOut + '_supsur' + lstNiiNames[indMap] + '.nii.gz'
        nb.save(niiOut, strTmp)

    # Save map with best ratios as nii
    niiOut = nb.Nifti1Image(aryRatMap,
                            aryAff,
                            header=hdrMsk
                            )
    # Save nii:
    strTmp = strPathOut + '_supsur' + '_Ratios' + '.nii.gz'
    nb.save(niiOut, strTmp)

    # Delete all the inbetween results, if desired by user
    if lgcDel:
        lstCmpRes = [item for sublist in lstCmpRes for item in sublist]
        print('------Delete in-between results')
        for strMap in lstCmpRes[:]:
            os.remove(strMap)


def map_crt_to_pol(aryXCrds, aryYrds):
    """Remap coordinates from cartesian to polar

    Parameters
    ----------
    aryXCrds : 1D numpy array
        Array with x coordinate values.
    aryYrds : 1D numpy array
        Array with y coordinate values.

    Returns
    -------
    aryTht : 1D numpy array
        Angle of coordinates
    aryRad : 1D numpy array
        Radius of coordinates.
    """

    aryRad = np.sqrt(aryXCrds**2+aryYrds**2)
    aryTht = np.arctan2(aryYrds, aryXCrds)

    return aryTht, aryRad


def map_pol_to_crt(aryTht, aryRad):
    """Remap coordinates from polar to cartesian

    Parameters
    ----------
    aryTht : 1D numpy array
        Angle of coordinates
    aryRad : 1D numpy array
        Radius of coordinates.

    Returns
    -------
    aryXCrds : 1D numpy array
        Array with x coordinate values.
    aryYrds : 1D numpy array
        Array with y coordinate values.
    """

    aryXCrds = aryRad * np.cos(aryTht)
    aryYrds = aryRad * np.sin(aryTht)

    return aryXCrds, aryYrds


def find_near_pol_ang(aryEmpPlrAng, aryExpPlrAng):
    """Return index of nearest expected polar angle.

    Parameters
    ----------
    aryEmpPlrAng : 1D numpy array
        Empirically found polar angle estimates
    aryExpPlrAng : 1D numpy array
        Theoretically expected polar angle estimates

    Returns
    -------
    aryXCrds : 1D numpy array
        Indices of nearest theoretically expected polar angle.
    aryYrds : 1D numpy array
        Distances to nearest theoretically expected polar angle.
    """

    dist = np.abs(np.subtract(aryEmpPlrAng[:, None],
                              aryExpPlrAng[None, :]))

    return np.argmin(dist, axis=-1), np.min(dist, axis=-1)


def rmp_rng(aryVls, varNewMin, varNewMax, varOldThrMin=None,
            varOldAbsMax=None):
    """Remap values in an array from one range to another.

    Parameters
    ----------
    aryVls : 1D numpy array
        Array with values that need to be remapped.
    varNewMin : float
        Desired minimum value of new, remapped array.
    varNewMax : float
        Desired maximum value of new, remapped array.
    varOldThrMin : float
        Theoretical minimum of old distribution. Can be specified if this
        theoretical minimum does not occur in empirical distribution but
        should be considered nontheless.
    varOldThrMin : float
        Theoretical maximum of old distribution. Can be specified if this
        theoretical maximum does not occur in empirical distribution but
        should be considered nontheless.

    Returns
    -------
    aryVls : 1D numpy array
        Array with remapped values.
    """
    if varOldThrMin is None:
        varOldMin = aryVls.min()
    else:
        varOldMin = varOldThrMin
    if varOldAbsMax is None:
        varOldMax = aryVls.max()
    else:
        varOldMax = varOldAbsMax

    aryNewVls = np.empty((aryVls.shape), dtype=aryVls.dtype)
    for ind, val in enumerate(aryVls):
        aryNewVls[ind] = (((val - varOldMin) * (varNewMax - varNewMin)) /
                          (varOldMax - varOldMin)) + varNewMin

    return aryNewVls


def crt_2D_gauss(varSizeX, varSizeY, varPosX, varPosY, varSd):
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
    [1] mathworld.wolfram.com/GaussianFunction.html
    """
    varSizeX = int(varSizeX)
    varSizeY = int(varSizeY)

    # create x and y in meshgrid:
    aryX, aryY = sp.mgrid[0:varSizeX, 0:varSizeY]

    # The actual creation of the Gaussian array:
    aryGauss = (
        (np.square((aryX - varPosX)) + np.square((aryY - varPosY))) /
        (2.0 * np.square(varSd))
        )
    aryGauss = np.exp(-aryGauss) / (2 * np.pi * np.square(varSd))

    return aryGauss


def cnvl_2D_gauss(idxPrc, aryMdlParamsChnk, arySptExpInf, tplPngSize, queOut):
    """Spatially convolve input with 2D Gaussian model.

    Parameters
    ----------
    idxPrc : int
        Process ID of the process calling this function (for CPU
        multi-threading). In GPU version, this parameter is 0 (just one thread
        on CPU).
    aryMdlParamsChnk : 2d numpy array, shape [n_models, n_model_params]
        Array with the model parameter combinations for this chunk.
    arySptExpInf : 3d numpy array, shape [n_x_pix, n_y_pix, n_conditions]
        All spatial conditions stacked along second axis.
    tplPngSize : tuple, 2.
        Pixel dimensions of the visual space (width, height).
    queOut : multiprocessing.queues.Queue
        Queue to put the results on. If this is None, the user is not running
        multiprocessing but is just calling the function
    Returns
    -------
    data : 2d numpy array, shape [n_models, n_conditions]
        Closed data.
    Reference
    ---------
    [1]
    """
    # Number of combinations of model parameters in the current chunk:
    varChnkSze = aryMdlParamsChnk.shape[0]

    # Number of conditions / time points of the input data
    varNumLstAx = arySptExpInf.shape[-1]

    # Output array with results of convolution:
    aryOut = np.zeros((varChnkSze, varNumLstAx))

    # Loop through combinations of model parameters:
    for idxMdl in range(0, varChnkSze):

        # Spatial parameters of current model:
        varTmpX = aryMdlParamsChnk[idxMdl, 0]
        varTmpY = aryMdlParamsChnk[idxMdl, 1]
        varTmpSd = aryMdlParamsChnk[idxMdl, 2]

        # Create pRF model (2D):
        aryGauss = crt_2D_gauss(tplPngSize[0],
                                tplPngSize[1],
                                varTmpX,
                                varTmpY,
                                varTmpSd)

        # Multiply pixel-time courses with Gaussian pRF models:
        aryCndTcTmp = np.multiply(arySptExpInf, aryGauss[:, :, None])

        # Calculate sum across x- and y-dimensions - the 'area under the
        # Gaussian surface'.
        aryCndTcTmp = np.sum(aryCndTcTmp, axis=(0, 1))

        # Put model time courses into function's output with 2d Gaussian
        # arrray:
        aryOut[idxMdl, :] = aryCndTcTmp

    if queOut is None:
        # if user is not using multiprocessing, return the array directly
        return aryOut

    else:
        # Put column with the indices of model-parameter-combinations into the
        # output array (in order to be able to put the pRF model time courses
        # into the correct order after the parallelised function):
        lstOut = [idxPrc,
                  aryOut]

        # Put output to queue:
        queOut.put(lstOut)


class cls_set_config(object):
    """
    Set config parameters from dictionary into local namespace.

    Parameters
    ----------
    dicCnfg : dict
        Dictionary containing parameter names (as keys) and parameter values
        (as values). For example, `dicCnfg['varTr']` contains a float, such as
        `2.94`.
    """

    def __init__(self, dicCnfg):
        """Set config parameters from dictionary into local namespace."""
        self.__dict__.update(dicCnfg)
