# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 14:03:25 2016

this script turns the pickle file create by CreateMasks.py into b and w PNGs

@author: marian
"""
import Image
import pickle
import os
import numpy as np

# get parent path
str_path_parent_up = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
# deduce paths
picklePath = os.path.join(str_path_parent_up, "Masks", 'pRFMasks.pickle')
outPath = os.path.join(str_path_parent_up, "PNGs")
if not os.path.exists(outPath):
    os.makedirs(outPath)

# load masks from pickle
print "Opening pickle.."
with open(picklePath, 'rb') as handle:
    arrays = pickle.load(handle)
print "Opening pickle done"

# load np arrays from dictionary and save their 2D slices as png
scaleValue = 255  # value to multipy mask value (1s) with for png format
for key in arrays.keys():
    array = arrays[key]
    for index in np.arange(array.shape[2]):
        im = Image.fromarray(scaleValue * array[:, :, index].astype(np.uint8))
        filename = key + '_' + str(index) + '.png'
        im.save(os.path.join(outPath, filename))
