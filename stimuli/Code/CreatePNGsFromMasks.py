# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 14:03:25 2016

this script turns the pickle file create by CreateMasks.py into b and w PNGs

@author: marian
"""
import Image
import pickle
import numpy as np
import time

# load masks from pickle
path2Pickle = '/home/marian/Documents/Testing/gif_aperture/pRFMasks.pickle'
print "Opening pickle.."
with open(path2Pickle, 'rb') as handle:
    arrays = pickle.load(handle)
print "Opening pickle done"

# load np arrays from dictionary and save their 2D slices as png
path2SavePNGs2 = '/home/marian/Documents/Testing/gif_aperture/'
scaleValue = 255  # value to multipy mask value (1s) with for png format
for key in arrays.keys():
    array = arrays[key]
    for index in np.arange(array.shape[2]):
        im = Image.fromarray(scaleValue * array[:, :, index].astype(np.uint8))
        filename = key + '_' + str(index) + '.png'
        im.save(path2SavePNGs2 + filename)