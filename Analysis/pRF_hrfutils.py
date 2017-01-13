# -*- coding: utf-8 -*-
# This module was copied from nipy
"""
Created on Fri Jan 13 12:15:40 2017

@author: marian
"""

from __future__ import division
from functools import partial
import numpy as np
import scipy.stats as sps

# SPMs HRF
def spm_hrf_compat(t,
                   peak_delay=6,
                   under_delay=16,
                   peak_disp=1,
                   under_disp=1,
                   p_u_ratio=6,
                   normalize=True,
                  ):
    """ SPM HRF function from sum of two gamma PDFs

    This function is designed to be partially compatible with SPMs `spm_hrf.m`
    function.

    The SPN HRF is a *peak* gamma PDF (with location `peak_delay` and dispersion
    `peak_disp`), minus an *undershoot* gamma PDF (with location `under_delay`
    and dispersion `under_disp`, and divided by the `p_u_ratio`).

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
    See ``spm_hrf.m`` in the SPM distribution.
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


def spmt(t):
    """ SPM canonical HRF, HRF values for time values `t`

    This is the canonical HRF function as used in SPM. It
    has the following defaults:
                                                defaults
                                                (seconds)
    delay of response (relative to onset)         6
    delay of undershoot (relative to onset)      16
    dispersion of response                        1
    dispersion of undershoot                      1
    ratio of response to undershoot               6
    onset (seconds)                               0
    length of kernel (seconds)                   32
    """
    return spm_hrf_compat(t, normalize=True)


def dspmt(t):
    """ SPM canonical HRF derivative, HRF derivative values for time values `t`

    This is the canonical HRF derivative function as used in SPM.

    It is the numerical difference of the HRF sampled at time `t` minus the
    values sampled at time `t` -1
    """
    t = np.asarray(t)
    return spmt(t) - spmt(t - 1)


_spm_dd_func = partial(spm_hrf_compat, normalize=True, peak_disp=1.01)


def ddspmt(t):
    """ SPM canonical HRF dispersion derivative, values for time values `t`

    This is the canonical HRF dispersion derivative function as used in SPM.

    It is the numerical difference between the HRF sampled at time `t`, and
    values at `t` for another HRF shape with a small change in the peak
    dispersion parameter (``peak_disp`` in func:`spm_hrf_compat`).
    """
    return (spmt(t) - _spm_dd_func(t)) / 0.01


def cnvlTcOld(idxPrc,
              aryBoxCarChunk,
              vecHrf,
              varNumVol,
              queOut):
    """
    Old version:
    Convolution of time courses with one canonical HRF model.
    """
    # adjust the input, if necessary, such that input is 2D, with last dim time
    tplInpShp = aryBoxCarChunk.shape
    aryBoxCarChunk = aryBoxCarChunk.reshape((-1, aryBoxCarChunk.shape[-1]))

    # Prepare an empty array for ouput
    aryConv = np.zeros(np.shape(aryBoxCarChunk))

    # Each time course is convolved with the HRF separately, because the
    # numpy convolution function can only be used on one-dimensional data.
    # Thus, we have to loop through time courses:
    for idxTc in range(0, aryConv.shape[0]):

        # Extract the current time course:
        vecTc = aryBoxCarChunk[idxTc, :]

        # In order to avoid an artefact at the end of the time series, we have
        # to concatenate an empty array to both the design matrix and the HRF
        # model before convolution.
        vecZeros = np.zeros([100, 1]).flatten()
        vecTc = np.concatenate((aryBoxCarChunk[idxTc, :], vecZeros))
        vecHrf = np.concatenate((vecHrf, vecZeros))

        # Convolve design matrix with HRF model:
        aryConv[idxTc, :] = np.convolve(vecTc,
                                        vecHrf,
                                        mode='full')[0:varNumVol]

    # Create list containing the convolved timecourses, and the process ID:
    lstOut = [idxPrc,
              aryConv.reshape(tplInpShp)]

    # Put output to queue:
    queOut.put(lstOut)
