# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 12:12:04 2016

@author: marian
many elements of this script are taken from
py_28_pRF_finding from Ingo Marquardt
"""
# %%
# *** Import modules
from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
import os
import numpy as np
import scipy as sp
from scipy import stats
import time
import pickle
import itertools
import multiprocessing as mp
from pRF_functions import (funcHrf, funcNrlTcTngCrv,
                           makeAoMTuneCurves, simulateGaussNoise,
                           simulateAR1, simulateAR2)
import sys

# %%
# *** Define parameters

# get some parameters from command line
varNumCmdArgs = len(sys.argv) - 1

if varNumCmdArgs == 2:
    # determine the type of aperture
    aprtType = str(sys.argv[1])
    # calculate noise again?
    lgcNoise = bool(int(sys.argv[2]))
elif varNumCmdArgs == 1:
    # determine the type of aperture
    aprtType = str(sys.argv[1])
    # calculate noise again?
    lgcNoise = False
else:
    raise ValueError('Not enough command line args provided.' +
                     'Provide aperture type information' +
                     '[mskBar, mskCircle, mskSquare]')

# Number of x-positions to model:
varNumX = 25
# Number of y-positions to model:
varNumY = 25
# Number of pRF sizes to model:
varNumPrfSizes = 22

# Extend of visual space from centre of the screen (i.e. from the fixation
# point):
varExtXmin = -12.00
varExtXmax = 12.00
varExtYmin = -12.0
varExtYmax = 12.00

# Maximum and minimum pRF model size (standard deviation of 2D Gaussian):
varPrfStdMin = 1.0
varPrfStdMax = 22.0  # max is given by np.sqrt(2*np.power(2*varExtXmax,2))

# %%
# Number of png files (meaning aperture positions) to load:
varNumPNG = 33

# Number of presented runs
varNumRuns = 6

# Volume TR of input data [s]:
varTr = 3.0

# Determine the factors by which the image should be downsampled (since the
# original array is rather big; factor 2 means:downsample from 1200 to 600):
factorX = 1
factorY = 1

# state the number of parallel processes
varPar = 8

# Output path for time course files:
strPathMdl = '/media/sf_D_DRIVE/MotionLocaliser/Simulation2p0/Apertures/pRF_model_tc/' + \
    aprtType + '/'

# %%
# provide parameters for pRF time course creation

# Base name of pickle files that contain order of stim presentat. in each run
# file should contain 1D array, column contains present. order of aperture pos,
# here file is 2D where 2nd column contains present. order of motion directions
strPathPresOrd = '/media/sf_D_DRIVE/MotionLocaliser/Simulation2p0/Conditions/Conditions_run0'

# Base name of png files representing the stimulus aperture as black and white
# the decisive information is in alpha channel (4th dimension)
strPathPNGofApertPos = '/media/sf_D_DRIVE/MotionLocaliser/Simulation2p0/Apertures/PNGs/' + \
    aprtType + '/' + 'Ima_'

# Size of png files (pixel*pixel):
tplPngSize = (128, 128)

strPathNoise = '/media/sf_D_DRIVE/MotionLocaliser/Simulation2p0/Apertures/pRF_model_tc/noise/noise_xval'

# %%
# ***  Check time
varTme01 = time.time()


# %%
# *** Load presentation order of conditions
print('------Load presentation order of apert pos and mot dir')
aryPresOrd = np.empty((0, 2))
for idx01 in range(0, varNumRuns):
    # reconstruct file name
    filename1 = (strPathPresOrd + str(idx01+1) + '.pickle')
    # load array
    with open(filename1, 'rb') as handle:
        array1 = pickle.load(handle)
    tempCond = array1["Conditions"]
    # add temp array to aryPresOrd
    aryPresOrd = np.concatenate((aryPresOrd, tempCond), axis=0)
aryPresOrd = aryPresOrd.astype(int)

# deduce the number of time points (conditions/volumes) from len aryPresOrd
varNumTP = len(aryPresOrd)

# %%
# *** Load PNGs with aperture position information
print('------Load PNGs containing aperture position')

# Create list of png files to load:
lstPathsPNGofApertPos = [None] * varNumPNG
for idx01 in range(0, varNumPNG):
    lstPathsPNGofApertPos[idx01] = (strPathPNGofApertPos + str(idx01) + '.png')

# Load png files. The png data will be saved in a numpy array of the
# following order: aryApertPos[x-pixel, y-pixel, PngNumber]. The
# first three values per pixel (RGB) that the sp.misc.imread function returns
# are not relevant here, what matters is the alpha channel (4th dimension).
# So only the 4th dimension is considered here and we discard the others.
aryApertPos = np.zeros((tplPngSize[0], tplPngSize[1], varNumPNG))
for idx01 in range(0, varNumPNG):
    aryApertPos[:, :, idx01] = sp.misc.imread(
        lstPathsPNGofApertPos[idx01])

# Convert RGB values (0 to 255) to integer ones and zeros:
aryApertPos = (aryApertPos > 0).astype(int)

# up- or downsample the image
aryApertPos = aryApertPos[0::factorX, 0::factorY, :]
tplPngSize = aryApertPos.shape[0:2]


# %%
# *** Combine information of aperture pos and presentation order
print('------Combine information of aperture pos and presentation order')
aryCond = aryApertPos[:, :, aryPresOrd[:, 0]]

# %%
# *** Create tuning curves for axes of motion
print('------create tuning curves for axes of motion')
x = np.arange(1, 8+1)
aryAoMMus = np.arange(1, 4+1, dtype='int')
# zip so that the respective 1st, 2nd, 3rd, asf. elem are combined
aryAoMMus = zip(aryAoMMus, aryAoMMus+4)
aryAoMSig = np.array([20/45, 45/45, 70/45])
iterables = [aryAoMMus, aryAoMSig]
aryAoM = list(itertools.product(*iterables))

# unpack the zipping
for ind, item in enumerate(aryAoM):
    aryAoM[ind] = list(np.hstack(item))
aryAoM = np.array(aryAoM)

aryTngCrvAoM = np.empty((len(x), len(aryAoM)))
for ind, (mu1, mu2, sig) in enumerate(aryAoM):
    aryTngCrvAoM[:, ind] = makeAoMTuneCurves(x, mu1, mu2, sig)

# %%
# testing and appending zeros
print('------Print area under the curve for axes of motion curves')
for ind in range(0, aryTngCrvAoM.shape[1]):
    print np.sum(aryTngCrvAoM[:, ind])

# add 0 at position 0 (this has to do with experim. coding: 0 means null trial)
aryTngCrvAoM = np.vstack((np.zeros(aryTngCrvAoM.shape[1]), aryTngCrvAoM))
# add 0 zero at position 9 (again experim. coding: 9 means static)
aryTngCrvAoM = np.vstack((aryTngCrvAoM, np.zeros(aryTngCrvAoM.shape[1])))

# add one more array that has equal response to all motion positions
aryTngCrvEqProb = np.hstack(([0], np.tile(np.array([1/8]), 8), [0]))
aryTngCrvEqProb = aryTngCrvEqProb.reshape((-1, 1))
# add all three arrays together to form one common array of all tuning curves
aryTngCrv = np.concatenate((aryTngCrvAoM, aryTngCrvEqProb),
                           axis=1)
varNumTngCrv = aryTngCrv.shape[1]

# Create array with boxcar weights
# first create a 2D array that depending on the tuning curve model returns
# weights, such that for every tuning curve model there is a 1D array
# that contains one weight per TR/ trial (weights are between 0 and 1)
# the resulting 1D array (per tuning curve model) can simply be multiplied with
# arrCond to yield pixel-wise boxcar functions: pwBoxC = aryCond * bcWeights[0]
bcWeights = np.empty((aryTngCrv.shape[1], len(aryPresOrd[:, 1])))
for ind01, tunCurve in enumerate(aryTngCrv.T):
    bcWeights[ind01, :] = tunCurve[aryPresOrd[:, 1]]

# %%
print('------Create array for pw box car functions')

# calculate an array that contains pixelwise box car functions for every
# tuning curve model
aryBoxCar = np.empty((tplPngSize[0], tplPngSize[1], varNumTngCrv, varNumTP),
                     dtype='float32')
for idx, bcWeight in enumerate(bcWeights):
    aryBoxCar[:, :, idx, :] = aryCond * bcWeight

# %%
# *** Prepare the creation of "neural" time courses models

# Vector with the x-indicies of the positions in the visual
# space at which to create pRF models.
vecX = np.linspace(0,
                   (tplPngSize[0] - 1),
                   varNumX,
                   endpoint=True)
# Vector with the y-indicies of the positions in the visual
# space at which to create pRF models.
vecY = np.linspace(0,
                   (tplPngSize[1] - 1),
                   varNumY,
                   endpoint=True)

# exclude x and y combination that are outside the circle (since stimuli will
# be shown in circular aperture)
iterables = [vecX, vecY]
combiXY = list(itertools.product(*iterables))
combiXY = np.asarray(combiXY)
# pass only the combinations inside the circle aperture
temp = combiXY-(tplPngSize[0]/2)
temp = np.sqrt(np.power(temp[:, 0], 2) + np.power(temp[:, 1], 2))
lgcXYcombi = np.less(temp, tplPngSize[0]/2)
combiXY = combiXY[lgcXYcombi]
# define number of combinations for x and y positions
varNumXY = len(combiXY)

# Vector with the standard deviations of the pRF models in visual space.
vecPrfSd = np.linspace(varPrfStdMin,
                       varPrfStdMax,
                       varNumPrfSizes,
                       endpoint=True)
# check whether deg to pix ratio is the same in x and y dimensions
if (tplPngSize[0]/(varExtXmax-varExtXmin) ==
        tplPngSize[1]/(varExtYmax-varExtYmin)):
    pix2degX = tplPngSize[0]/(varExtXmax-varExtXmin)
else:
    print('------ERROR. Deg: Pix ratio differs between x and y dimension. ' +
          'Check whether calculation of pRF sizes is still correct.')

# We need to account for the pixel:degree ratio by multiplying the pixel
# PrfSd in visual degree with that ratio to yield pixel PrfSd
vecPrfSd = np.multiply(vecPrfSd, pix2degX)

# Vector with indices for different tuning curve models
vecTngCrvIndcs = np.arange(varNumTngCrv, dtype='int8')

# put all possible combinations for three 2D Gauss parameters (x, y, sigma)
# and motion direction tuning curve models into tuple array
iterables = [combiXY, vecPrfSd, vecTngCrvIndcs]
aryNrlParamsPix = list(itertools.product(*iterables))
# undo the zipping
for ind, item in enumerate(aryNrlParamsPix):
    aryNrlParamsPix[ind] = list(np.hstack(item))
aryNrlParamsPix = np.array(aryNrlParamsPix)

# calculate number of models
varNumMdls = varNumXY*varNumPrfSizes*varNumTngCrv

# %%
# *** Create a dictionary that contains the ground truth (for pRF_comp)

# Vector with the modeled x-positions of the pRFs:
vecMdlXpos = np.linspace(varExtXmin,
                         varExtXmax,
                         varNumX,
                         endpoint=True)

# Vector with the modeled y-positions of the pRFs:
vecMdlYpos = np.linspace(varExtYmin,
                         varExtYmax,
                         varNumY,
                         endpoint=True)

# Vector with the modeled standard deviations of the pRFs:
vecMdlSd = np.linspace(varPrfStdMin,
                       varPrfStdMax,
                       varNumPrfSizes,
                       endpoint=True)

# exclude x and y combinations that are outside the circle mask
iterables = [vecMdlXpos, vecMdlYpos]
vecMdlXY = list(itertools.product(*iterables))
vecMdlXY = np.asarray(vecMdlXY)
vecMdlXY = vecMdlXY[lgcXYcombi]  # use the logical that we defined beforehand

# get combinations of all model parameters (this time in deg of vis angle)
iterables = [vecMdlXY, vecMdlSd, vecTngCrvIndcs]
aryNrlParamsDeg = list(itertools.product(*iterables))
for ind, item in enumerate(aryNrlParamsDeg):
    aryNrlParamsDeg[ind] = list(np.hstack(item))
aryNrlParamsDeg = np.array(aryNrlParamsDeg)

array = {'aryNrlParamsPix': aryNrlParamsPix,
         'aryNrlParamsDeg': aryNrlParamsDeg,
         'varNumXY': varNumXY,
         'varNumPrfSizes': varNumPrfSizes,
         'varNumTngCrv': varNumTngCrv,
         'varNumMdls': varNumMdls,
         }
# save dictionary to pickle
filename = os.path.join(strPathMdl, 'dicNrlParams_xval.pickle')
with open(filename, 'wb') as handle:
    pickle.dump(array, handle)


# %%
# *** Create "neural" time courses models
print('------Create "neural" time courses models')

# Empty list for results (2D gaussian models):
lstNrlMdls = [None] * varPar
# Empty list for processes:
lstPrcs = [None] * varPar
# Counter for parallel processes:
varCntPar = 0
# Counter for output of parallel processes:
varCntOut = 0
# Create a queue to put the results in:
queOut = mp.Queue()

# Number of neural models:
print('---------Number of neural models that will be created: ' +
      str(varNumMdls))

# put n =varPar number of chunks of the Gauss parameters into list for
# parallel processing
lstPrlNrlMdls = np.array_split(aryNrlParamsPix, varPar)

print('---------Creating parallel processes for neural models')

# Create processes:
for idxPrc in range(0, varPar):
    lstPrcs[idxPrc] = mp.Process(target=funcNrlTcTngCrv,
                                 args=(idxPrc,
                                       tplPngSize[0],
                                       tplPngSize[1],
                                       lstPrlNrlMdls[idxPrc],
                                       varNumTP,
                                       aryBoxCar,  # tngCrvBrds,
                                       strPathMdl,
                                       varNumMdls,
                                       varPar,
                                       queOut)
                                 )

    # Daemon (kills processes when exiting):
    lstPrcs[idxPrc].Daemon = True

# Start processes:
for idxPrc in range(0, varPar):
    lstPrcs[idxPrc].start()

# Collect results from queue:
for idxPrc in range(0, varPar):
    lstNrlMdls[idxPrc] = queOut.get(True)

# Join processes:
for idxPrc in range(0, varPar):
    lstPrcs[idxPrc].join()

# save neural models as list
print('---------Save neural models to list')
the_filename = (strPathMdl + 'lstNrlMdls_xval' + '.pickle')
with open(the_filename, 'wb') as f:
    pickle.dump(lstNrlMdls, f)

# save neural models as np array
print('---------Save neural models to numpy array')

aryNrlMdls = np.empty((0, varNumTP), dtype='float32')
# sort list by the first entry (idxPrc), prll processes finish in mixed order
lstNrlMdls = sorted(lstNrlMdls)
# Put output into correct order:
for idxRes in range(0, varPar):
    # Put fitting results into list, in correct order:
    aryNrlMdls = np.concatenate((aryNrlMdls, lstNrlMdls[idxRes][1]))
# reshaping yields and array representing the visual space, of the form
# aryNrlMdls[combi xy positions, pRF-size, tng curves, nr of time points],
# which will hold the "neural" model time courses.
aryNrlMdls = aryNrlMdls.reshape((varNumXY, varNumPrfSizes,
                                 varNumTngCrv, varNumTP))
# Save the 4D array as '*.npy' file:
print('---------Start saving neural models as nyp')
np.save(strPathMdl+'pRF_model_tc_xval',
        aryNrlMdls)
print('---------saving done')
# sdelete the list to save memory
del(lstNrlMdls)


# %%
# *** Convolve neural tc with HRF to get time courses
# before paralelisation was used for this but it runs quickly enough without
print('------Convolve neural tc with HRF to get time courses')

# Create HRF time course:
vecHrf = funcHrf(varNumTP, varTr)
# reshape the ary containing model predictors for convolution with HRF
aryNrlMdls = aryNrlMdls.reshape(varNumMdls, varNumTP)

# preprocess the data for convolution
# In order to avoid an artefact at the end of the time series, we have to
# concatenate an empty array to both the HRF model and the neural time courses
# before convolution.
vecHrf = np.concatenate((vecHrf, np.zeros([100, 1]).flatten()))
aryNrlMdls = np.hstack((aryNrlMdls, np.zeros((aryNrlMdls.shape[0], 100))))
aryMdlConv = np.zeros((varNumMdls, varNumTP))

# concolve every model time course with the HRF function
for idx in range(0, varNumMdls):
    temp = np.convolve(aryNrlMdls[idx, :], vecHrf, mode='full')[0:varNumTP]
    aryMdlConv[idx, :] = temp

# reshape data so the time courses are stored more intuitively
aryMdlConv = aryMdlConv.reshape(varNumXY, varNumPrfSizes,
                                varNumTngCrv, varNumTP)

# change data back to what it was before preprocessing
vecHrf = vecHrf[:varNumTP]
aryNrlMdls = aryNrlMdls[:, :varNumTP]
aryNrlMdls = aryNrlMdls.reshape(varNumXY, varNumPrfSizes,
                                varNumTngCrv, varNumTP)

# Save the 4D array as '*.npy' file:
print('---------Start saving convolved time courses as nyp')
np.save(strPathMdl+'pRF_model_tc_conv_xval',
        aryMdlConv)
print('---------saving done')


# %%
# *** Z-normalise the convolved time courses
print('------Z-normalise the convolved time courses')

# reshape the ary containing model predictors for convolution with HRF
aryMdlConv = aryMdlConv.reshape(varNumMdls, varNumTP)
# Zscore each time course
aryMdlConvZscore = stats.zscore(aryMdlConv, axis=1, ddof=1)

aryMdlConvZscore = aryMdlConvZscore.reshape(varNumXY, varNumPrfSizes,
                                            varNumTngCrv, varNumTP)

# Save the 4D array as '*.npy' file:
print('---------Start saving convolved and zscored time courses as nyp')
np.save(strPathMdl+'pRF_model_tc_conv_zscored_xval',
        aryMdlConvZscore)
print('---------saving done')


# %%
# *** Add AR2 noise to model time courses

print('------Add noise to model time courses')

# set the contrast to noise ratios
# Definition of CNR and tSNR follow equation 4 and 1 in Welvaert et al. (2013),
# respectively: http://dx.doi.org/10.1371/journal.pone.0077089
# If we set the CNR, the tSNR falls out automatically; it is printed below
CNR = np.array([0.1, 0.5, 1])
varNumCNR = len(CNR)

# set the noise type
noiseType = 'AR2'  # 'Gauss', 'AR1', 'AR2'

# set the correlation weights of the noise
# should be one weight for AR1, two weights for AR2, irrelevant if gaussian
aryCrrWghts = np.array([0.8, -0.25])

# reshape the array with modelled time courses
aryMdlConvZscore = aryMdlConvZscore.reshape(varNumMdls, varNumTP)

# If the SD of the signal and noise are known,
# and the signal is zero-mean (we zscored before):
# CNR = SignalSD.^2 / NoiseSD.^2 (Welvaert, 2013 equat. 4-5), or consequently
# NoiseSD = sqrt(SignalSD.^2/CNR)
# since we zscored we can assume that SignalSD.^2 = 1.^2 = 1
SDNoise = np.sqrt(np.power(1, 2)/CNR)
print ('------SDNoise was:' + str(SDNoise))
# calculate the tSNR:
# tSNR = np.mean(signal)/SDNoise
# (since we zscored the mean of the signal is 0):
tSNR = 0/SDNoise
print "tSNR was:" + str(tSNR)

if lgcNoise:
    if noiseType == 'Gauss':
        noise = simulateGaussNoise(
            0,
            SDNoise,
            varNumTP,
            varNumCNR,
            )
    elif noiseType == 'AR1':
        noise = simulateAR1(
            varNumTP,
            aryCrrWghts[0],
            SDNoise,
            0,
            1,
            varNumCNR,
            varNumTP,
            )
    elif noiseType == 'AR2':
        noise = simulateAR2(
            varNumTP,
            aryCrrWghts,
            SDNoise,
            0,
            1,
            varNumCNR,
            varNumTP,
            )
    # save noise so it can be used by other apertures
    np.save(strPathNoise, noise)
else:
    # load noise
    noise = np.load(strPathNoise + '.npy')

# response = signal + noise, here aryMdlConvZscore is the signal
# use broadcastng for efficiency
# this will yield an array (varNumMdls, varNumCNR, varNumTP)
simResp = aryMdlConvZscore[:, np.newaxis, :] + noise

# Save the 4D array as '*.npy' file:
print('---------Start saving simulated responses as nyp')
for ind in np.arange(simResp.shape[1]):
    temp = simResp[:, ind, :]
    np.save(strPathMdl+'simResp_xval_'+str(ind),
            temp)
print('---------saving done')

# %%
# *** Report time

varTme02 = time.time()
varTme03 = varTme02 - varTme01
print('-Elapsed time: ' + str(varTme03) + ' s')
print('-Done.')
# *****************************************************************************
