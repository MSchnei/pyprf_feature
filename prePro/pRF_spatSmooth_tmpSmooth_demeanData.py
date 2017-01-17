# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 14:09:45 2016

@author: marian
"""
# %% import modules

import os
import numpy as np
# import nibabel load functionality
from nibabel import load, save, Nifti1Image
from scipy import stats
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import gaussian_filter1d


# %% set paths and parameters

# set parent paths to folder containing files tah should be filtered
parentPath = '/media/sf_D_DRIVE/PacMan/Analysis/P3/UndistortedMotCor/02_highPass/'
# set path for output
outPath = '/media/sf_D_DRIVE/PacMan/Analysis/P3/UndistortedMotCor/03_SmoothSpat1SmoothTmp1Demean'
if not os.path.exists(outPath):
    os.makedirs(outPath)

# set list with file names
niiLst = ['func_07_hpf.nii.gz',
          'func_08_hpf.nii.gz',
          'func_09_hpf.nii.gz',
          'func_10_hpf.nii.gz',
          ]

fwhmSpat = 0.8
varVoxRes = 0.8
fwhmTmp = 2.832
varTr = 2.832


for idx in np.arange(len(niiLst)):
    print('---Working on Run: ' + str(idx+1))

    # %% load data with nibabel
    print('------Loading data')
    fmri_file = os.path.join(parentPath, niiLst[idx])
    fmri_data = load(fmri_file)
    data = fmri_data.get_data()
    # fix NaNs
    print('------Fix NaNs')
    data[np.isnan(data)] = 0
    # spatially smooth data
    print('------Spatially smooth')    
    for idxVol in range(0, data.shape[-1]):
        data[:, :, :, idxVol] = gaussian_filter(
            data[:, :, :, idxVol],
            np.divide(fwhmSpat, varVoxRes),
            order=0,
            mode='nearest',
            truncate=4.0)
    # temporally smooth data
            
    dataMean = np.mean(data,
                       axis=3,
                       keepdims=True)

    data = np.concatenate((dataMean,
                           data,
                           dataMean), axis=3)

    # In the input data, time goes from left to right. Therefore, we apply
    # the filter along axis=1.
    aryFuncChnk = gaussian_filter1d(data,
                                    np.divide(fwhmTmp, varTr),
                                    axis=3,
                                    order=0,
                                    mode='nearest',
                                    truncate=4.0)
    # Remove mean-intensity volumes at the beginning and at the end:
    data = data[:, :, :, 1:-1]

    # reshape
    data = data.reshape(-1, data.shape[-1])
    # %% load time series, normalize
    print('------Normalize data')
    data = np.subtract(data, np.mean(data, axis=1)[:, None])

    # %% save as nii
    print('------Saving data')
    # get new filename
    name = os.path.splitext(os.path.basename(fmri_file))[0]
    ext = os.path.splitext(os.path.basename(fmri_file))[1]
    file_name = 'zs' + str(1) + '_' + str(1) + name + ext

    # Create output nii object:
    new_image = Nifti1Image(data.reshape(fmri_data.get_shape()),
                            header=fmri_data.get_header(),
                            affine=fmri_data.get_affine()
                            )

    # Save as nii:
    save(new_image,
         os.path.join(outPath, file_name))
