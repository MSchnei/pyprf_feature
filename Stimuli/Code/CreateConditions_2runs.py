# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 16:21:45 2016

@author: marian
"""

from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
import numpy as np
import os
import pickle
import itertools

# set paramters
NrOfApertures = 32
NrOfMotionDir = 8+1  # add another number for static (number 9)

ExpectedTR = 2.832
TargetDuration = 0.3

# calculate total number of conditions
NrOfCond = NrOfApertures*NrOfMotionDir
# prepare vector for motion direction; define 8 directions of motion
MotionDir = np.linspace(1, NrOfMotionDir, NrOfMotionDir)

# prepare vector for aperture configurations
ApertureConfig1 = np.linspace(1, NrOfApertures/2, NrOfApertures/2)
ApertureConfig2 = np.linspace(NrOfApertures/2 + 1,
                              NrOfApertures,
                              NrOfApertures/2)

# find all possible combinations
iterables = [ApertureConfig1, MotionDir]
Conditions1 = list(itertools.product(*iterables))
Conditions1 = np.asarray(Conditions1)
iterables = [ApertureConfig2, MotionDir]
Conditions2 = list(itertools.product(*iterables))
Conditions2 = np.asarray(Conditions2)

# then reshuffle
np.random.shuffle(Conditions1)
np.random.shuffle(Conditions2)

# prepare vector for presentation order
# NullTrial = 0; Stimulus = 1
NrNullTrialStart = 5
NrNullTrialEnd = 5
NrNullTrialProp = 1/8
NrNullTrialBetw = np.round(NrOfCond/2*NrNullTrialProp)

# determine at which position the null trials will be inserted
# avoid repetition
lgcRep = True
while lgcRep:
    NullPos = np.random.choice(np.arange(1, NrOfCond/2), NrNullTrialBetw)

    lgcRep = np.greater(np.sum(np.diff(np.sort(NullPos)) == 1), 0)

# insert null trials in between
Conditions1 = np.insert(Conditions1, NullPos, np.array([0, 0]), axis=0)
Conditions2 = np.insert(Conditions2, NullPos, np.array([0, 0]), axis=0)

# add null trials in beginning and end
Conditions1 = np.vstack((np.zeros((NrNullTrialStart, 2)),
                        Conditions1,
                        np.zeros((NrNullTrialEnd, 2))))
Conditions2 = np.vstack((np.zeros((NrNullTrialStart, 2)),
                        Conditions2,
                        np.zeros((NrNullTrialEnd, 2))))

# prepare targets
NrOfTargets = int(len(Conditions1)/5)
Targets = np.zeros(len(Conditions1))
lgcRep = True
while lgcRep:
    TargetPos = np.random.choice(np.arange(NrNullTrialStart,
                                 len(Conditions1)-NrNullTrialEnd), NrOfTargets,
                                 replace=False)
    lgcRep = np.greater(np.sum(np.diff(np.sort(TargetPos)) == 1), 0)
Targets[TargetPos] = 1
assert NrOfTargets == np.sum(Targets)

# prepare random target onset delay
TargetOnsetinSec = np.random.uniform(0.1,
                                     ExpectedTR-TargetDuration,
                                     size=NrOfTargets)

# create dictionary for saving to pickle
array_run1 = {'Conditions': Conditions1,
              'Targets': Targets,
              'TargetOnsetinSec': TargetOnsetinSec,
              'ExpectedTR': ExpectedTR,
              'ExpectedTargetDuration': TargetDuration,
              }
array_run2 = {'Conditions': Conditions2,
              'Targets': Targets,
              'TargetOnsetinSec': TargetOnsetinSec,
              'ExpectedTR': ExpectedTR,
              'ExpectedTargetDuration': TargetDuration,
              }

# save dictionary to pickle
folderpath = '/media/sf_D_DRIVE/PacMan/PsychoPyScripts/Pacman_Scripts/PacMan_Pilot3_20161220/ModBasedMotLoc/Conditions'
filename1 = os.path.join(folderpath, 'Conditions_run05.pickle')
filename2 = os.path.join(folderpath, 'Conditions_run06.pickle')

with open(filename1, 'wb') as handle:
    pickle.dump(array_run1, handle)
with open(filename2, 'wb') as handle:
    pickle.dump(array_run2, handle)
