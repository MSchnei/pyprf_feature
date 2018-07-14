# -*- coding: utf-8 -*-

""" The aim of this script will be to fit 1d gaussians or von Mises curves to
the beta parameter tuning curves. This is to get an idea of the tuning spread.

Not completed.

""" 

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

import numpy as np
import nibabel as nb
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from scipy.stats import vonmises_line, norm

# this needs to go later
import pRF_config as cfg

# %%
print('------Load best models and data')
# load the mask
niiMask = nb.load(cfg.strPathNiiMask)
aryMask = niiMask.get_data().astype('bool')

# get best models
aryRes = np.load(cfg.strPathOut + '_aryPrfRes.npy')
# mask the results array
aryRes = aryRes[aryMask, :]

# get beta weights for best models
aryBstTrainBetas = np.load(cfg.strPathOut + '_aryBstTrainBetas.npy',)
aryBstTstBetas = np.load(cfg.strPathOut + '_aryBstTestBetas.npy',)

# load manual MT mask
pathManMtMsk = '/media/sf_D_DRIVE/MotionLocaliser/Analysis/P02/Struct/P02_ManMTMsk.nii.gz'
niiManMtMsk = nb.load(pathManMtMsk)
aryManMtMsk = niiManMtMsk.get_data().astype('bool')
lgcManMtMsk = aryManMtMsk[aryMask]

# %%
varThreshR2 = 0.15
# use R2 values as a mask
vecR2 = aryRes[:, 3]
lgcR2 = [vecR2 >= varThreshR2][0]

# combine R2 and ManMtMask logical
lgc = np.logical_and(lgcR2, lgcManMtMsk)

# exclude voxels with low R2, exclude last column (since it is weight 4 static)
aryBetasTrn = aryBstTrainBetas[lgc, :-1]
aryBetasTst = aryBstTstBetas[lgc, :-1]

# %% demean the betas
aryBetasDemean = np.subtract(aryBetasTrn, np.mean(aryBetasTrn,axis=0)[None, :])
aryBetasScaled = np.divide(aryBetasTrn, np.max(aryBetasTrn,axis=1)[:, None])

aryBetasScaled = scale(aryBetasTrn, axis=1, with_mean=False, with_std=True,
                       copy=True)

# %% fit 1D Gauss
def funcGauss1D(x, mu, sig):
    """ Create 1D Gaussian. Source:
    http://mathworld.wolfram.com/GaussianFunction.html
    """
    arrOut = np.exp(-np.power((x - mu)/sig, 2.)/2)
    # normalize
    arrOut = arrOut/(np.sqrt(2.*np.pi)*sig)

    return arrOut

x = np.linspace(-np.pi, np.pi, 360)

test = funcGauss1D(x, 2, 30)
test = norm.pdf(x)
mu, std = norm.fit(y)

# %% fit von Mises

fig, ax = plt.subplots(1, 1)
# kappa = np.deg2rad(45)
kappa = 3.99
x = np.linspace(-np.pi, np.pi, 360)
test = vonmises_line.pdf(x, 0.006)

mus = np.arange(0, 360, 45)

aryCrvV1 = np.empty((len(x), len(mus)), dtype='float64')
for ind, mu in enumerate(mus-180):
    aryCrvV1[:, ind] = np.roll(vonmises_line.pdf(x, kappa), mu)

vecIndV1 = np.empty((len(mus), len(mus)))
for ind in np.arange(len(mus)):
    vecIndV1[:, ind] = aryCrvV1[mus, ind]
