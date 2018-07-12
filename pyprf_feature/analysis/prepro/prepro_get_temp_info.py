# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 18:51:28 2018

@author: marian
"""

import os
import numpy as np
import pickle

# %% set parameters
strPthPrnt = "/home/marian/Documents/Testing/pyprf_testing/expInfo"

# provide names of condition files in the order that they were shown
lstPickleFiles = [
    'Conditions_run01.pickle',
    'Conditions_run02.pickle',
    'Conditions_run03.pickle',
    'Conditions_run04.pickle',
    'Conditions_run05.pickle',
    'Conditions_run06.pickle',
    ]

# provide the TR in seconds
varTr = 3.0

# provide the stimulation time
varStmTm = 3.0

# %% load conditions files

# Loop through npz files in target directory:
lstCond = []
for ind, cond in enumerate(lstPickleFiles):
    inputFile = os.path.join(strPthPrnt, 'conditions', cond)

    with open(inputFile, 'rb') as handle:
        array1 = pickle.load(handle)
    aryTmp = array1["Conditions"].astype('int32')

    # append condition to list
    lstCond.append(aryTmp)

# join conditions across runs
aryCond = np.vstack(lstCond)

# create empty array
aryTmpCond = np.empty((len(aryCond), 4), dtype='float16')
# get the condition nr
aryTmpCond[:, 0] = aryCond[:, 0]
# get the onset time
aryTmpCond[:, 1] = np.cumsum(np.ones(len(aryCond))*varTr) - varTr
# get the duration
aryTmpCond[:, 2] = np.ones(len(aryCond))*varStmTm
# add the feature identifier
aryTmpCond[:, 3] = aryCond[:, 1]

strPthAry = os.path.join(strPthPrnt, 'tmpInfo',
                         'aryTmpExpInf')
np.save(strPthAry, aryTmpCond)
