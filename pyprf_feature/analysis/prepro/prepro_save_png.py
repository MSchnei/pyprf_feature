#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
This pre-processing script does the following:
- load temporal info (in 2D nump array)
- load spatial info (apertures stack in 3D numpy array)
- saves resulting images as pngs

Input:
- path to parent folder
- output path for png files
- path arySptExpInf.npy
- path aryTmpExpInf.npy
- downsampling factor

Output
- png files, one per TR
"""

import os
import numpy as np
from scipy.misc import imsave

# %% set parameters

# set path to parent folder
strPthPrnt = "/media/sf_D_DRIVE/MotDepPrf/Analysis/S02"
# set output path for png files
strPthOut = "/media/sf_D_DRIVE/MotDepPrf/Analysis/S02/04_motDepPrf/expInfo/pngs"

# provide path extension to *npy file with spatial info
strPthSptExpInf = "04_motDepPrf/expInfo/sptInfo/arySptExpInf.npy"
# provide path extension to *npy file with spatial info
strPthTmpExpInf = "04_motDepPrf/expInfo/tmpInfo/aryTmpExpInf_cntr.npy"

# set factors for downsampling
factorX = 1
factorY = 1

# %% load condition files and typset

# load spatial experiment info
arySptExpInf = np.load(os.path.join(strPthPrnt, strPthSptExpInf))
# load spatial conditions
aryTmpExpInf = np.load(os.path.join(strPthPrnt, strPthTmpExpInf))

# typeset arySptExpInf
arySptExpInf = arySptExpInf.astype(np.int8)
# typeset aryTmpExpInf
aryTmpExpInf = aryTmpExpInf.astype(np.float64)


# %% generate png files

# loop over unique identifiers of spatial aperture conditions
for indCnt, indCnd in enumerate(aryTmpExpInf[:, 0]):

    # retrieve image from arySptExpInf
    ima = arySptExpInf[:, :, int(indCnd)]

    # if desired, downsample
    if factorX > 1 or factorY > 1:
        ima = ima[::factorX, ::factorY]

    if indCnt > 999:
        strFlNm = ("frame" + '' + str(indCnt) + '.png')
    elif indCnt > 99:
        strFlNm = ("frame" + '0' + str(indCnt) + '.png')
    elif indCnt > 9:
        strFlNm = ("frame" + '00' + str(indCnt) + '.png')
    else:
        strFlNm = ("frame" + '000' + str(indCnt) + '.png')

    # derive output path
    strPathOutIma = os.path.join(strPthOut, strFlNm)
    # save as png
    imsave(strPathOutIma, ima)
