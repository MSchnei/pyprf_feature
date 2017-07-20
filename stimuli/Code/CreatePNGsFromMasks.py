# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 14:03:25 2016

this script turns the pickle file create by CreateMasks.py into b and w PNGs

@author: marian
"""
#import Image
from PIL import Image
import os
import numpy as np

# factorX = 8
# factorY = 8

# value to multipy mask value (1s) with for png format
scaleValue = 255

# get parent path
str_path_parent_up = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))

# Path of mask files:
strPthMsk = (str_path_parent_up
             + os.path.sep
             + 'Masks'
             + os.path.sep)

# List of files in target directory:
lstFls = os.listdir(strPthMsk)

# Loop through npz files in target directory:
for strTmp in lstFls:
    if '.npz' in strTmp:

        # Load npz file content into list:
        with np.load((strPthMsk + strTmp)) as objMsks:
            lstMsks = objMsks.items()

        for objTmp in lstMsks:
            strMsg = 'Mask type: ' + objTmp[0]
            # The following print statement prints the name of the mask stored
            # in the npz array from which the mask shape is retrieved. Can be
            # used to check whether the correct mask has been retrieved.
            print(strMsg)
            array = objTmp[1]

        # deduce paths
        outPath = os.path.join(str_path_parent_up, "PNGs")
        if not os.path.exists(outPath):
            os.makedirs(outPath)

        # load np arrays from dictionary and save their 2D slices as png
        for index in np.arange(array.shape[2]):
            im = Image.fromarray(scaleValue
                                 * array[:, :, index].astype(np.uint8))
            filename = (objTmp[0] + '_' + str(index) + '.png')
            im.save((os.path.join(outPath, filename)))
