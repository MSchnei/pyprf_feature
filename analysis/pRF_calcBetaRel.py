# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 18:00:36 2017

@author: marian
"""

# %%
import numpy as np
import os
from scipy import stats
from scipy.stats import vonmises_line
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from nibabel import load, save, Nifti1Image

# path to found beta values
strPathBetaValues = '/media/sf_D_DRIVE/MotionLocaliser/Analysis/Pilot1_08112016/FitResults/MotionXval/MotionXval_MotLoc_aryBstBetas.npy'
# path to mask
strPathMask = '/media/sf_D_DRIVE/MotionLocaliser/Analysis/Pilot1_08112016/Struct/FuncMask_mas_man4.nii.gz'
# path to directory where results should be saved
strPathOut = '/media/sf_D_DRIVE/MotionLocaliser/Analysis/Pilot1_08112016/FitResults/MotionXval'


# *** preprocess values
# load
aryBestBetas = np.load(strPathBetaValues)
# discard responses to static for now
aryBestBetas = aryBestBetas[:, :-1]
# zscore betas
aryBestBetas = stats.zscore(aryBestBetas, axis=0, ddof=1)

## normalize 1
#vecMin = np.min(aryBestBetas, axis=1)
#vecMax = np.max(aryBestBetas, axis=1)
#aryBestBetas = np.divide(np.subtract(aryBestBetas, vecMin[:, None]),
#                         np.subtract(vecMax, vecMin)[:, None])

# normalize 2
# make sure that all values are positive
aryBestBetas = np.subtract(aryBestBetas, np.min(aryBestBetas, axis=1)[:, None])
# set minimum value to 1
aryBestBetas += 1
# closure operator (get values proportional to each other)
aryBestBetas = np.divide(aryBestBetas, np.sum(aryBestBetas, axis=1)[:, None])
# aryBestBetas = np.divide(aryBestBetas, np.max(aryBestBetas, axis=1)[:, None])

# find the maximum
vecMu = np.argmax(aryBestBetas, axis=1)

# roll so that each row has maximum at 3rd entry, to match the fitted curve
# first roll so that maximum sits on 0
rows, columns = np.ogrid[:aryBestBetas.shape[0], :aryBestBetas.shape[1]]
columns = vecMu[:, np.newaxis] - columns
aryBestBetas = aryBestBetas[rows, columns]
# then roll so that maximum sits on 2
aryBestBetas = np.roll(aryBestBetas, 2, axis=1)


x = np.linspace(-np.pi, np.pi, 4, endpoint=False)
def fitAoM(x, kappa):
    potFit = vonmises_line.pdf(x, kappa)
    potFit = np.divide(potFit, np.sum(potFit))
    return potFit

varNumVoxels = len(aryBestBetas)
vecKappas = np.empty(varNumVoxels)
vecCor = np.empty(varNumVoxels)
for ind in np.arange(varNumVoxels):
    vecKappas[ind], vecCor[ind] = curve_fit(fitAoM, x, aryBestBetas[ind, :])

# %% load mask
MskHdr = load(strPathMask)
MskData = MskHdr.get_data().astype('bool')

aryPrfRes = np.array([vecMu, vecKappas, vecCor]).T

lsNames = [
    'prefDir',
    'disp',
    'dispCor',
    ]

for ind in np.arange(aryPrfRes.shape[1]):
    dataTemp = np.zeros(MskHdr.shape)
    dataTemp[MskData] = aryPrfRes[:, ind]

    file_name = lsNames[ind] + '.nii.gz'

    # set data type of nii
#    MskHdr.set_data_dtype(np.int)

    # Create output nii object:
    new_image = Nifti1Image(dataTemp,
                            header=MskHdr.header,
                            affine=MskHdr.affine
                            )
    # Save as nii:
    save(new_image,
         os.path.join(strPathOut, file_name))

## %% plot
#plt.plot(x, aryBestBetas[ind, :], 'b+:', label='data')
#plt.plot(x, fitAoM(x, *popt), 'ro:', label='fit')
#plt.legend()
#plt.title('Fig. 3 - Fit for Kappa')
#plt.xlabel('angle')
#plt.ylabel('response')
#plt.show()

