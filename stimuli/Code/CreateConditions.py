# -*- coding: utf-8 -*-

"""Prepare condition order, target times and noise texture."""

from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
import numpy as np
import os
import pickle
import itertools

# set paramters
varNrOfApertAxes = 4
varNrOfRuns = int((varNrOfApertAxes*(varNrOfApertAxes-1))/2)
varNrOfApertFields = 8
varNrOfApert = varNrOfApertAxes * varNrOfApertFields
varNrOfMotionDir = 8+1  # 8 motion directions + 1 static, number 9 for static

ExpectedTR = 3.0
TargetDuration = 0.3

# calculate total number of conditions
varNrOfCond = varNrOfApert*varNrOfMotionDir
# prepare vector for motion direction; define 8 directions of motion
vecMotionDir = np.linspace(1, varNrOfMotionDir, varNrOfMotionDir)

# %% prepare the aryCondStub, which will form the basis of the Condtions array

# prepare vector for aperture configurations
vecApertConfig = np.linspace(1, varNrOfApert, varNrOfApert)
# find all possible combinations
iterables = [vecApertConfig, vecMotionDir]
aryCondStub = list(itertools.product(*iterables))
aryCondStub = np.asarray(aryCondStub)

# reshape for shuffling
aryCondStub = aryCondStub.reshape((varNrOfApert, varNrOfMotionDir, 2))
# shuffle motion directions(along axis 1)
for ind, cond in enumerate(aryCondStub):
    aryCondStub[ind, :, :] = aryCondStub[ind, np.random.permutation(
        varNrOfMotionDir), :]
# transpose for convenience
aryCondStub = np.transpose(aryCondStub, (1, 0, 2))
# reshape back
aryCondStub = aryCondStub.reshape((varNrOfMotionDir*varNrOfApert, 2))

# create indices such that aperture axes will be grouped together
ind1 = []
for ind in np.arange(varNrOfApertAxes):
    ind1.append(np.arange(ind, varNrOfApert+varNrOfApertAxes,
                          varNrOfApertAxes))
ind1 = np.hstack(ind1)
ind2 = np.arange(len(aryCondStub)).reshape((varNrOfApert+varNrOfApertAxes,
                                           varNrOfApertFields))
ind2 = ind2[ind1, :].flatten()
# use the indices to group axes together
aryCondStub = aryCondStub[ind2, :]

# reshape for shuffling
aryCondStub = aryCondStub.reshape((varNrOfApertAxes*varNrOfMotionDir,
                                   varNrOfApertFields, 2))
# shuffle aperture order (along axis 1)
for ind, cond in enumerate(aryCondStub):
    aryCondStub[ind, :, :] = aryCondStub[ind, np.random.permutation(
        varNrOfApertFields), :]

# axis, subpart of axis, aperture order, [aperture index, motion direction]
aryCondStub = aryCondStub.reshape((varNrOfApertAxes, varNrOfMotionDir,
                                   varNrOfApertFields, 2))

# %%
# determine which axes will be presented in which run
aryAxisPerRun = np.array(list(
    itertools.combinations(np.arange(varNrOfApertAxes), 2)))
# shuffle along axis 0
np.random.shuffle(aryAxisPerRun)
# shuffle along axis 1
for ind, axis in enumerate(aryAxisPerRun):
    aryAxisPerRun[ind, :] = aryAxisPerRun[ind, np.random.permutation(2)]

# decide how the two axes per run should be ordered
vecAxisOrder = np.arange(2*varNrOfMotionDir)
np.random.shuffle(vecAxisOrder)

# %%
# combine the information from the aryCondStub, the aryAxisPerRun and
# vecAxisOrder to get an array that hosts aperture position and motion
# direction per run
aryCondsPerRun = np.empty((varNrOfRuns,
                           2*varNrOfMotionDir*varNrOfApertFields, 2))
for ind in np.arange(varNrOfRuns):
    aryTemp = aryCondStub[aryAxisPerRun[ind], :, :, :].reshape(
        2*varNrOfMotionDir, varNrOfApertFields, 2)
    aryCondsPerRun[ind, :, :] = aryTemp[vecAxisOrder, :, :].reshape(
        2*varNrOfMotionDir*varNrOfApertFields, 2)

# %% Insert null trials

# NullTrial = 0; Stimulus = 1
varNrNullTrialStart = 5
varNrNullTrialEnd = 5
varNrNullTrialProp = 1/8
varNrNullTrialBetw = np.round(varNrOfCond/2*varNrNullTrialProp)

# determine for every run at which position the null trials will be inserted
# avoid repetition
aryNullPos = np.empty((varNrOfRuns, varNrNullTrialBetw))
for ind in np.arange(varNrOfRuns):
    lgcRep = True
    # switch to avoid repetitions
    while lgcRep:
        aryNullPos[ind, :] = np.random.choice(np.arange(1, varNrOfCond/2),
                                              varNrNullTrialBetw,
                                              replace=False)
        # check that two null positions do not follow each other immediat.
        lgcRep = np.greater(np.sum(np.diff(np.sort(aryNullPos[ind, :])) == 1),
                            0)

# insert null trials inbetween as well as in the beginning and end
Conditions = np.empty((aryCondsPerRun.shape[0],
                       (aryCondsPerRun.shape[1] + varNrNullTrialStart +
                        varNrNullTrialEnd+varNrNullTrialBetw),
                       aryCondsPerRun.shape[2]))
for ind in np.arange(varNrOfRuns):
    conditionsTemp = np.insert(aryCondsPerRun[ind, :, :], aryNullPos[ind, :],
                               np.array([0, 0]), axis=0)
    # add null trials in beginning and end
    Conditions[ind, :, :] = np.vstack((np.zeros((varNrNullTrialStart, 2)),
                                       conditionsTemp,
                                       np.zeros((varNrNullTrialEnd, 2))))

# get number of volumes
varNrOfVols = Conditions.shape[1]

# %% Prepare target times

# prepare targets
varNrOfTargets = int(varNrOfVols/5)
Targets = np.zeros((Conditions.shape[0:2]))
TargetOnsetinSec = np.empty((varNrOfRuns, varNrOfTargets))

for ind in np.arange(varNrOfRuns):
    # prepare random target positions
    lgcRep = True
    # switch to avoid repetitions
    while lgcRep:
        TargetPos = np.random.choice(np.arange(varNrNullTrialStart,
                                     varNrOfVols-varNrNullTrialEnd),
                                     varNrOfTargets, replace=False)
        # check that two targets do not follow each other immediately
        lgcRep = np.greater(np.sum(np.diff(np.sort(TargetPos)) == 1), 0)
    Targets[ind, TargetPos] = 1
    assert varNrOfTargets == np.sum(Targets[ind])

    # prepare random target onset delay
    TargetOnsetinSec[ind, :] = np.random.uniform(0.1,
                                                 ExpectedTR-TargetDuration,
                                                 size=varNrOfTargets)

# %% Prepare array for random noise pattern
varNoiseDim = 256
aryNoiseTexture = np.ones((varNoiseDim, varNoiseDim))  # white
blackDotIdx = np.random.choice(varNoiseDim*varNoiseDim,
                               varNoiseDim*varNoiseDim/2)
aryNoiseTexture = aryNoiseTexture.flatten()
aryNoiseTexture[blackDotIdx] = -1  # black
aryNoiseTexture = aryNoiseTexture.reshape((varNoiseDim, varNoiseDim))

# %% save the results
for ind in np.arange(varNrOfRuns):
    # create dictionary for saving to pickle
    array = {'Conditions': Conditions[ind],
             'Targets': Targets[ind],
             'TargetOnsetinSec': TargetOnsetinSec[ind],
             'ExpectedTR': ExpectedTR,
             'ExpectedTargetDuration': TargetDuration,
             'NoiseTexture': aryNoiseTexture,
             }

    # save dictionary to pickle
    str_path_parent_up = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..'))
    filename = os.path.join(str_path_parent_up, 'Conditions',
                            'Conditions_run0' + str(ind+1) + '.pickle')

    with open(filename, 'wb') as handle:
        pickle.dump(array, handle)
