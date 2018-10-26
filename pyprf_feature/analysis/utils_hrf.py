# -*- coding: utf-8 -*-
"""Main functions for hrf."""

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

from __future__ import division
from functools import partial
import numpy as np
import scipy.stats as sps
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve


def spm_hrf_compat(t,
                   peak_delay=6,
                   under_delay=16,
                   peak_disp=1,
                   under_disp=1,
                   p_u_ratio=6,
                   normalize=True,
                   ):
    """SPM HRF function from sum of two gamma PDFs

    Parameters
    ----------
    t : array-like
        vector of times at which to sample HRF
    peak_delay : float, optional
        delay of peak
    peak_disp : float, optional
        width (dispersion) of peak
    under_delay : float, optional
        delay of undershoot
    under_disp : float, optional
        width (dispersion) of undershoot
    p_u_ratio : float, optional
        peak to undershoot ratio.  Undershoot divided by this value before
        subtracting from peak.
    normalize : {True, False}, optional
        If True, divide HRF values by their sum before returning. SPM does this
        by default.

    Returns
    -------
    hrf : array
        vector length ``len(t)`` of samples from HRF at times `t`

    Notes
    -----
    [1] This function is designed to be partially compatible with SPMs
    `spm_hrf.m` function.
    [2] The SPN HRF is a *peak* gamma PDF (with location `peak_delay` and
    dispersion `peak_disp`), minus an *undershoot* gamma PDF (with location
    `under_delay` and dispersion `under_disp`, and divided by the `p_u_ratio`).
    [3] See ``spm_hrf.m`` in the SPM distribution.

    References:
    -----
    [1] http://nipy.org/
    [2] https://github.com/fabianp/hrf_estimation
    """
    if len([v for v in [peak_delay, peak_disp, under_delay, under_disp]
            if v <= 0]):
        raise ValueError("delays and dispersions must be > 0")
    # gamma.pdf only defined for t > 0
    hrf = np.zeros(t.shape, dtype=np.float)
    pos_t = t[t > 0]
    peak = sps.gamma.pdf(pos_t,
                         peak_delay / peak_disp,
                         loc=0,
                         scale=peak_disp)
    undershoot = sps.gamma.pdf(pos_t,
                               under_delay / under_disp,
                               loc=0,
                               scale=under_disp)
    hrf[t > 0] = peak - undershoot / p_u_ratio
    if not normalize:
        return hrf
    return hrf / np.max(hrf)


def spmt(t, peak_delay=6, under_delay=16, peak_disp=1, under_disp=1,
         p_u_ratio=6):
    """Normalized SPM HRF function from sum of two gamma PDFs

    Parameters
    ----------
    t : array-like
        vector of times at which to sample HRF

    Returns
    -------
    hrf : array
        vector length ``len(t)`` of samples from HRF at times `t`

    Notes
    -----
    [1] This is the canonical HRF function as used in SPM. It
    has the following defaults:
        - delay of response (relative to onset) : 6s
        - delay of undershoot (relative to onset) : 16s
        - dispersion of response : 1s
        - dispersion of undershoot : 1s
        - ratio of response to undershoot : 6s
        - onset : 0s
        - length of kernel : 32s

    References:
    -----
    [1] http://nipy.org/
    [2] https://github.com/fabianp/hrf_estimation
    """
    return spm_hrf_compat(t, peak_delay=peak_delay, under_delay=under_delay,
                          peak_disp=peak_disp, under_disp=under_disp,
                          p_u_ratio=p_u_ratio, normalize=True)


def dspmt(t, peak_delay=6, under_delay=16, peak_disp=1, under_disp=1,
          p_u_ratio=6):
    """ SPM canonical HRF derivative, HRF derivative values for time values `t`

    Parameters
    ----------
    t : array-like
        vector of times at which to sample HRF

    Returns
    -------
    hrf : array
        vector length ``len(t)`` of samples from HRF at times `t`

    Notes
    -----
    [1] This is the canonical HRF derivative function as used in SPM.
    [2] It is the numerical difference of the HRF sampled at time `t` minus the
    values sampled at time `t` -1

    References:
    -----
    [1] http://nipy.org/
    [2] https://github.com/fabianp/hrf_estimation
    """
    t = np.asarray(t)
    aryRsp1 = spmt(t, peak_delay=peak_delay, under_delay=under_delay,
                   peak_disp=peak_disp, under_disp=under_disp,
                   p_u_ratio=p_u_ratio)
    aryRsp2 = spmt(t-1, peak_delay=peak_delay, under_delay=under_delay,
                   peak_disp=peak_disp, under_disp=under_disp,
                   p_u_ratio=p_u_ratio)

    return aryRsp1 - aryRsp2


def ddspmt(t, peak_delay=6, under_delay=16, peak_disp=1, under_disp=1,
           p_u_ratio=6):
    """ SPM canonical HRF dispersion derivative, values for time values `t`

    Parameters
    ----------
    t : array-like
        vector of times at which to sample HRF

    Returns
    -------
    hrf : array
        vector length ``len(t)`` of samples from HRF at times `t`

    Notes
    -----
    [1] This is the canonical HRF dispersion derivative function as used in SPM
    [2] It is the numerical difference between the HRF sampled at time `t`, and
    values at `t` for another HRF shape with a small change in the peak
    dispersion parameter (``peak_disp`` in func:`spm_hrf_compat`).

    References:
    -----
    [1] http://nipy.org/
    [2] https://github.com/fabianp/hrf_estimation
    """

    _spm_dd_func = partial(spmt, peak_delay=peak_delay,
                           under_delay=under_delay,
                           under_disp=under_disp, p_u_ratio=p_u_ratio,
                           peak_disp=1.01)

    return (spmt(t) - _spm_dd_func(t)) / 0.01


def create_boxcar(aryCnd, aryOns, aryDrt, varTr, varNumVol,
                  aryExclCnd=None, varTmpOvsmpl=1000.):
    """
    Creation of condition time courses in temporally upsampled space.

    Parameters
    ----------

    aryCnd : np.array
        1D array with condition identifiers (every condition has its own int)
    aryOns : np.array, same len as aryCnd
        1D array with condition onset times in seconds.
    aryDrt : np.array, same len as aryCnd
        1D array with condition durations of different conditions in seconds.
    varTr : float, positive
        Time to repeat (TR) of the (fMRI) experiment.
    varNumVol : float, positive
        Number of volumes of the (fMRI) data.
    aryExclCnd : array
        1D array containing condition identifiers for conditions to be excluded
    varTmpOvsmpl : float, positive
        Factor by which the time courses should be temporally upsampled.

    Returns
    -------
    aryBxCrOut : np.array, float16
        Condition time courses in temporally upsampled space.

    References:
    -----
    [1] https://github.com/fabianp/hrf_estimation

    """
    if aryExclCnd is not None:
        for cond in aryExclCnd:
            aryOns = aryOns[aryCnd != cond]
            aryDrt = aryDrt[aryCnd != cond]
            aryCnd = aryCnd[aryCnd != cond]

    resolution = varTr / float(varTmpOvsmpl)
    aryCnd = np.asarray(aryCnd)
    aryOns = np.asarray(aryOns, dtype=np.float)
    unique_conditions = np.sort(np.unique(aryCnd))
    boxcar = []

    for c in unique_conditions:
        tmp = np.zeros(int(varNumVol * varTr/resolution))
        onset_c = aryOns[aryCnd == c]
        duration_c = aryDrt[aryCnd == c]
        onset_idx = np.round(onset_c / resolution).astype(np.int)
        duration_idx = np.round(duration_c / resolution).astype(np.int)
        aux = np.arange(int(varNumVol * varTr/resolution))
        for start, dur in zip(onset_idx, duration_idx):
            lgc = np.logical_and(aux >= start, aux < start + dur)
            tmp = tmp + lgc
        assert np.all(np.less(tmp, 2))
        boxcar.append(tmp)
    aryBxCrOut = np.array(boxcar).T
    if aryBxCrOut.shape[1] == 1:
        aryBxCrOut = np.squeeze(aryBxCrOut)
    return aryBxCrOut.astype('float16')


def cnvl_tc(idxPrc, aryPrfTcChunk, lstHrf, varTr, varNumVol, varTmpOvsmpl,
            queOut, varHrfLen=32., dctPrm=None):
    """Convolution of time courses with HRF model.

    Parameters
    ----------
    idxPrc : int, positive
        Process ID of the process calling this function (for CPU
        multi-threading). In GPU version, this parameter is 0 (just one thread
        on CPU).
    aryPrfTcChunk : np.array
        2D array with model time course to be convolved with HRF.
    lstHrf : list
        List containing the different HRF functions.
    varTr : float, positive
        Time to repeat (TR) of the (fMRI) experiment.
    varNumVol : float, positive
        Number of volumes of the (fMRI) data.
    varTmpOvsmpl : float, positive
        Factor by which the time courses should be temporally upsampled.
    queOut : multiprocessing.queues.Queue
        Queue to put the results on.
    varHrfLen : float, positive, default=32
        Length of the HRF time course in seconds.
    dctPrm : dictionary, default None
        Dictionary with customized hrf parameters. If this is None, default
        hrf parameters will be used.

    Returns
    -------
    lstOut : list
        int, positive : Process ID of the process calling this function.
        2D np.array, float16 : Model time course convolved with HRF.

    References:
    -----
    [1] https://github.com/fabianp/hrf_estimation

    """

    # Adjust the input, if necessary, such that input is 2D, with last dim time
    tplInpShp = aryPrfTcChunk.shape
    aryPrfTcChunk = aryPrfTcChunk.reshape((-1, aryPrfTcChunk.shape[-1]))

    # Prepare list to collect hrf basis functions
    lstBse = []
    # Prepare array that contains time intervals
    aryTme = np.linspace(0, varHrfLen, (varHrfLen // varTr) * varTmpOvsmpl)
    for fnHrf in lstHrf:
        # If hrf parameter dictionary is None, run with default parameters
        if dctPrm is None:
            vecTmpBse = fnHrf(aryTme)
        # Otherwise, run with custom parameters
        else:
            vecTmpBse = fnHrf(aryTme, **dctPrm)
        # Normalise HRF so that the sum of values is 1 (see FSL)
        # otherwise, after convolution values for predictors are very high
        vecTmpBse = np.divide(vecTmpBse, np.sum(vecTmpBse))

        lstBse.append(vecTmpBse)

    # Get frame times, i.e. start point of every volume in seconds
    vecFrms = np.arange(0, varTr * varNumVol, varTr)
    # Get supersampled frames times, i.e. start point of every volume in
    # upsampled res, since convolution takes place in temp. upsampled space
    vecFrmTms = np.arange(0, varTr * varNumVol, varTr / varTmpOvsmpl)

    # Prepare an empty array for ouput
    aryConv = np.zeros((aryPrfTcChunk.shape[0], len(lstHrf), varNumVol),
                       dtype=np.float16)
    # Each time course is convolved with the HRF separately, because the
    # numpy convolution function can only be used on one-dimensional data.
    # Thus, we have to loop through time courses:
    for idxTc in range(0, aryConv.shape[0]):

        # Extract the current time course (already in upsampled space):
        vecTcUps = aryPrfTcChunk[idxTc, :]

        # *** convolve
        for indBase, base in enumerate(lstBse):
            # Make sure base and vecTcUps are float64 to avoid overflow
            base = base.astype(np.float64)
            vecTcUps = vecTcUps.astype(np.float64)
            # Perform the convolution (previously: np.convolve)
            col = fftconvolve(base, vecTcUps, mode='full')[:vecTcUps.size]
            # Get function for downsampling
            f = interp1d(vecFrmTms, col)
            # Downsample to original resoltuion to match res of data
            # take the value from the centre of each volume's period (see FSL)
            aryConv[idxTc, indBase, :] = f(vecFrms + varTr/2.
                                           ).astype(np.float16)

    # Determine output shape
    tplOutShp = tplInpShp[:-1] + (len(lstHrf), ) + (varNumVol, )

    if queOut is None:
        # if user is not using multiprocessing, return the array directly
        return aryConv.reshape(tplOutShp)

    else:
        # Create list containing the convolved timecourses, and the process ID:
        lstOut = [idxPrc,
                  aryConv.reshape(tplOutShp)]

        # Put output to queue:
        queOut.put(lstOut)
