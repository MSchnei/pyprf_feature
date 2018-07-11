"""Convenience module to spatially and temporally smooth functional data."""

import os
import numpy as np
from nibabel import load, save, Nifti1Image
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d

# %% set paths and parameters

# set parent paths to folder containing files that should be filtered
strPthIn = '/str/to/nii/'
# set path for output
strPthOut = '/str/to/nii/out/'
if not os.path.exists(strPthOut):
    os.makedirs(strPthOut)

# set list with file names
lstNii = ['func01.nii',
          'func02.nii',
          'func03.nii',
          'func04.nii',
          ]

# set voxel resolution of the data
varVoxRes = 0.8
# set desired smoothing kernel [in mm]
varFwhmSpat = 0.8
# set temporal resolution (TR) of the data
varTr = 2.0
# set desired sd for temporal smoothing [in s]
varSdTmp = 2.0

# %%
for idx in np.arange(len(lstNii)):
    print('---Working on Run: ' + str(idx+1))

    # load data with nibabel
    print('------Loading data')
    strPthNiiFle = os.path.join(strPthIn, lstNii[idx])
    fnFmriData = load(strPthNiiFle)
    aryData = fnFmriData.get_data()

    # fix NaNs
    print('------Fix NaNs')
    aryData[np.isnan(aryData)] = 0

    # spatially smooth the data
    if varFwhmSpat > 0.0:
        print('------Spatially smooth')
        for idxVol in range(0, aryData.shape[-1]):
            aryData[..., idxVol] = gaussian_filter(
                aryData[..., idxVol],
                np.divide(varFwhmSpat, varVoxRes),
                order=0,
                mode='nearest',
                truncate=4.0)

    # temporally smooth the data
    if varSdTmp > 0.0:
        print('------Temporally smooth')
        # reshape
        tplInpShp = aryData.shape
        aryData = aryData.reshape(-1, aryData.shape[-1])
        # For the filtering to perform well at the ends of the time series, we
        # set the method to 'nearest' and place a volume with mean intensity
        # (over time) at the beginning and at the end.
        aryDataMean = np.mean(aryData, axis=-1, keepdims=True).reshape(-1, 1)
        aryData = np.concatenate((aryDataMean, aryData, aryDataMean), axis=-1)

        # In the input data, time goes from left to right. Therefore, we apply
        # the filter along axis=1.
        aryData = gaussian_filter1d(aryData, np.divide(varSdTmp, varTr),
                                    axis=-1, order=0, mode='nearest',
                                    truncate=4.0)

        # Remove mean-intensity volumes at the beginning and at the end:
        aryData = aryData[..., 1:-1]
        # Put back to original shape:
        aryData = aryData.reshape(tplInpShp)

    # %% save as nii
    print('------Saving data')
    # get new filename
    strName = os.path.splitext(os.path.basename(strPthNiiFle))[0]
    strExt = os.path.splitext(os.path.basename(strPthNiiFle))[1]
    if varFwhmSpat > 0.0 and varSdTmp > 0.0:
        strFleName = strName + '_sptSmth_tmpSmth' + strExt
    elif varFwhmSpat > 0.0 and varSdTmp == 0.0:
        strFleName = strName + '_sptSmth' + strExt
    elif varFwhmSpat == 0.0 and varSdTmp > 0.0:
        strFleName = strName + '_tmpSmth' + strExt
    else:
        strFleName = strName + strExt

    # Create output nii object:
    objIma = Nifti1Image(aryData,
                         header=fnFmriData.header,
                         affine=fnFmriData.affine
                         )

    # Save as nii:
    save(objIma,
         os.path.join(strPthOut, strFleName))
