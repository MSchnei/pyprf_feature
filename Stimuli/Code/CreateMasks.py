# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 11:27:49 2016

- This script, somewhat clumsily, creates masks for pRF stimulation
  in the form of numpy arrays
- Masks are of different kind: (classical) bars, squares and circles
- All masks are placed (i.e. masked themselves) in a circle aperture
- Each mask comes in two orientations: cardinal and oblique
- The resulting masks are saved in pRFMasks.pickle
- The pickle file can be turned into b and w PNGs using CreatePNGsFromMasks.py
  (scroll to bottom of script to change the path the masks are saved to)

@author: marian
"""

from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
import numpy as np
import matplotlib.pyplot as plt
from psychopy import monitors, misc
import itertools
import pickle
import os

#%%
""" (0) set general parameters"""

varSupSmp = 1
nsplits = 8
# to work as a mask for psychopy, the matrix size must be square
xpix = 1024
ypix = 1024

#%%
"""define a circular aperture for all the following stimuli"""
minOfXY = np.minimum(xpix, ypix)
radius = minOfXY/2

X, Y = np.meshgrid(
    np.arange(xpix) - xpix/2+0.5,
    np.arange(ypix) - ypix/2+0.5
    )

R = np.sqrt(np.square(X)+np.square(Y))
aperture = np.zeros(R.shape)
aperture[R < radius] = 1

#test1 = aperture
#imgplot = plt.imshow(test1)


#%%
""" (1a) bar apertures in cardinal orientation"""

# get indices to divide along width dimension in n equally szied segments
xsplitIdx = np.split(np.arange(xpix * varSupSmp), nsplits)
# get indices to divide along height dimension in n equally szied segments
ysplitIdx = np.split(np.arange(ypix * varSupSmp), nsplits)

# create empy numpy arrays
mskV = np.empty([xpix, ypix, nsplits])
mskH = np.empty([xpix, ypix, nsplits])

for idx in np.arange(nsplits):

    msk = np.zeros((xpix, ypix))
    msk[:, xsplitIdx[idx].astype(int)] = 1
    msk = np.logical_and(msk, aperture)
    mskV[:, :, idx] = msk

    msk = np.zeros((xpix, ypix))
    msk[ysplitIdx[idx].astype(int), :] = 1
    msk = np.logical_and(msk, aperture)
    mskH[:, :, idx] = msk

#test1 = mskV[:, :, 3]
#imgplot = plt.imshow(test1)

mskBarCard = np.concatenate((mskV, mskH), axis=2)

#%%
""" (1b) bar apertures in diagonal orientation"""

# get number of diagonals
minOfXY = np.minimum(xpix, ypix)
ndiags = 2 * minOfXY - 1
lowerBound = -ndiags/2
upperBound = ndiags/2
# factor in correction for diagonal
diagCorrFactor = minOfXY/np.sqrt(2*minOfXY**2)
lowerBound = np.floor(lowerBound*diagCorrFactor)
upperBound = np.ceil(upperBound*diagCorrFactor)

# get split indices
diagIndc = np.arange(lowerBound, upperBound)
# use array_split instead of linspace (which would be
# more elegant but does not allow int output currently)
ls = np.array_split(diagIndc, nsplits)
splitIdx1 = [item[0] for item in ls]
splitIdx2 = [item[-1] for item in ls]

# create empy numpy arrays
mskObli1 = np.empty([xpix, ypix, nsplits])
mskObli2 = np.empty([xpix, ypix, nsplits])

tmpl = np.ones((xpix, ypix))

for i, (idx1, idx2) in enumerate(zip(splitIdx1, splitIdx2)):
    # zero all elements above diagonal
    msk1 = np.tril(tmpl, idx2)
    # zero all elements below diagonal
    msk2 = np.triu(tmpl, idx1)
    # combine masks
    msk = np.logical_and(msk1, msk2)
    msk = np.logical_and(msk, aperture)

    mskObli1[:, :, i] = msk
    mskObli2[:, :, i] = np.fliplr(msk)

#test1 = mskObli1[:, :, 1]
#imgplot = plt.imshow(test1)
#plt.colorbar()

mskBarObli = np.concatenate((mskObli1, mskObli2), axis=2)

mskBar = np.concatenate((mskBarCard, mskBarObli), axis=2)
# add first image, which will be zero-only image
mskBar = np.dstack((np.zeros((xpix, ypix)), mskBar))


#%%
""" (2a) square apertures in cardinal orientation"""

# get indices to divide along width dimension in n equally szied segments
xsplitIdx = np.split(np.arange(xpix * varSupSmp), nsplits/2)
# get indices to divide along height dimension in n equally szied segments
ysplitIdx = np.split(np.arange(ypix * varSupSmp), nsplits/2)

# create empty masks
mskSquareCard = np.empty([xpix, ypix, (nsplits*nsplits)/4])

# combine conditions
iterables = [np.arange(nsplits/2).astype(int),
             np.arange(nsplits/2).astype(int)]
Conditions = list(itertools.product(*iterables))

for i, (idx1, idx2) in enumerate(Conditions):

    msk1 = np.zeros((xpix, ypix))
    msk1[xsplitIdx[idx1].astype(int), :] = 1
    msk2 = np.zeros((xpix, ypix))
    msk2[:, ysplitIdx[idx2].astype(int)] = 1
    msk = np.logical_and(msk1, msk2)
    msk = np.logical_and(msk, aperture)

    mskSquareCard[:, :, i] = msk

#test1 = mskSquareCard[:, :, 5]
#imgplot = plt.imshow(test1)

#%%
""" (2b) square apertures in oblique orientation"""

# get number of diagonals
minOfXY = np.minimum(xpix, ypix)
ndiags = 2 * minOfXY - 1
lowerBound = -ndiags/2
upperBound = ndiags/2
# factor in correction for diagonal
diagCorrFactor = minOfXY/np.sqrt(2*minOfXY**2)
lowerBound = np.floor(lowerBound*diagCorrFactor)
upperBound = np.ceil(upperBound*diagCorrFactor)

# get split indices
diagIndc = np.arange(lowerBound, upperBound)
# use array_split instead of linspace (which would be
# more elegant but does not allow int output currently)
ls = np.array_split(diagIndc, nsplits/2)
splitIdx1 = [item[0] for item in ls]
splitIdx2 = [item[-1] for item in ls]

# create empy numpy arrays
mskDiag1 = np.empty([xpix, ypix, nsplits/2])
mskDiag2 = np.empty([xpix, ypix, nsplits/2])

tmpl = np.ones((xpix, ypix))

# create single diagonal masks
for i, (idx1, idx2) in enumerate(zip(splitIdx1, splitIdx2)):
    # zero all elements above diagonal
    msk1 = np.tril(tmpl, idx2)
    # zero all elements below diagonal
    msk2 = np.triu(tmpl, idx1)
    # combine masks
    msk = np.logical_and(msk1, msk2)

    mskDiag1[:, :, i] = msk
    mskDiag2[:, :, i] = np.fliplr(msk)

# create empty masks
mskSquareObli = np.empty([xpix, ypix, (nsplits*nsplits)/4])

# combine conditions
iterables = [np.arange(nsplits/2).astype(int),
             np.arange(nsplits/2).astype(int)]
Conditions = list(itertools.product(*iterables))

# cross the single diagonal masks to obtain diagonal squares
for i, (idx1, idx2) in enumerate(Conditions):
    # zero all elements above diagonal
    msk1 = mskDiag1[:, :, idx1]
    # zero all elements below diagonal
    msk2 = mskDiag2[:, :, idx2]
    # combine masks
    msk = np.logical_and(msk1, msk2)
    msk = np.logical_and(msk, aperture)

    mskSquareObli[:, :, i] = msk

#test1 = mskSquareObli[:, :, 15]
#imgplot = plt.imshow(test1)

mskSquare = np.concatenate((mskSquareCard, mskSquareObli), axis=2)
# add first image, which will be zero-only image
mskSquare = np.dstack((np.zeros((xpix, ypix)), mskSquare))


#%%
""" (3a) circle apertures in cardinal orientation"""

# 1st step: define 4 quadrant apertures
xsplitIdx = np.split(np.arange(xpix * varSupSmp), 2)
ysplitIdx = np.split(np.arange(ypix * varSupSmp), 2)
# create empy numpy arrays
mskQuadrant = np.empty([xpix, ypix, 2*2])
# combine conditions
iterables = [np.arange(2).astype(int),
             np.arange(2).astype(int)]
Conditions = list(itertools.product(*iterables))
for i, (idx1, idx2) in enumerate(Conditions):
    msk1 = np.zeros((xpix, ypix))
    msk1[xsplitIdx[idx1].astype(int), :] = 1
    msk2 = np.zeros((xpix, ypix))
    msk2[:, ysplitIdx[idx2].astype(int)] = 1
    msk = np.logical_and(msk1, msk2)
    mskQuadrant[:, :, i] = msk

# 2nd step: define ring apertures
circleSplits = 4
rmin = np.minimum(xpix, ypix)/2

X, Y = np.meshgrid(
    np.arange(xpix) - xpix/2+0.5,
    np.arange(ypix) - ypix/2+0.5
    )

R = np.sqrt(np.square(X)+np.square(Y))

splitIdx1 = np.arange(0, rmin, rmin/circleSplits)
splitIdx2 = np.arange(0, rmin, rmin/circleSplits) + rmin/circleSplits

mskRings = np.empty([xpix, ypix, circleSplits])
for i, (idx1, idx2) in enumerate(zip(splitIdx1, splitIdx2)):
    aperture = np.zeros(R.shape)
    aperture[np.logical_and(R >= idx1, R < idx2)] = 1
    mskRings[:, :, i] = aperture

# 3rd step: combine quadrant and ring apertures
mskCircleCard = np.empty([xpix, ypix, 2*2*circleSplits])

# combine conditions
iterables = [np.arange(2*2).astype(int),
             np.arange(circleSplits).astype(int)]
Conditions = list(itertools.product(*iterables))

# cross the single diagonal masks to obtain diagonal squares
for i, (idx1, idx2) in enumerate(Conditions):
    # zero all elements above diagonal
    msk1 = mskQuadrant[:, :, idx1]
    # zero all elements below diagonal
    msk2 = mskRings[:, :, idx2]
    # combine masks
    msk = np.logical_and(msk1, msk2)

    mskCircleCard[:, :, i] = msk

#test1 = mskCircleCard[:, :, 8]
#imgplot = plt.imshow(test1)


#%%
""" (3b) circle apertures in oblique orientation"""

# 1st step: define 4 quadrant apertures (this time rotated by 45 degrees)
minOfXY = np.minimum(xpix, ypix)
ndiags = 2 * minOfXY - 1
lowerBound = -ndiags/2
upperBound = ndiags/2
# factor in correction for diagonal
diagCorrFactor = minOfXY/np.sqrt(2*minOfXY**2)
lowerBound = np.floor(lowerBound*diagCorrFactor)
upperBound = np.ceil(upperBound*diagCorrFactor)

# get split indices
diagIndc = np.arange(lowerBound, upperBound)
# use array_split instead of linspace (which would be
# more elegant but does not allow int output currently)
ls = np.array_split(diagIndc, 2)
splitIdx1 = [item[0] for item in ls]
splitIdx2 = [item[-1] for item in ls]

# create empy numpy arrays
mskQuadrantObl1 = np.empty([xpix, ypix, 2])
mskQuadrantObl2 = np.empty([xpix, ypix, 2])

tmpl = np.ones((xpix, ypix))

# create single diagonal masks
for i, (idx1, idx2) in enumerate(zip(splitIdx1, splitIdx2)):
    # zero all elements above diagonal
    msk1 = np.tril(tmpl, idx2)
    # zero all elements below diagonal
    msk2 = np.triu(tmpl, idx1)
    # combine masks
    msk = np.logical_and(msk1, msk2)

    mskQuadrantObl1[:, :, i] = msk
    mskQuadrantObl2[:, :, i] = np.fliplr(msk)

# cross the 4 conditions
mskQuadrantObl = np.empty([xpix, ypix, 4])
iterables = [np.arange(2).astype(int),
             np.arange(2).astype(int)]
Conditions = list(itertools.product(*iterables))
for i, (idx1, idx2) in enumerate(Conditions):
    msk = np.logical_and(mskQuadrantObl1[:, :, idx1],
                         mskQuadrantObl2[:, :, idx2])
    mskQuadrantObl[:, :, i] = msk

# 2nd step: define ring apertures
circleSplits = 4
rmin = np.minimum(xpix, ypix)/2

X, Y = np.meshgrid(
    np.arange(xpix) - xpix/2+0.5,
    np.arange(ypix) - ypix/2+0.5
    )

R = np.sqrt(np.square(X)+np.square(Y))

splitIdx1 = np.arange(0, rmin, rmin/circleSplits)
splitIdx2 = np.arange(0, rmin, rmin/circleSplits) + rmin/circleSplits

mskRings = np.empty([xpix, ypix, circleSplits])
for i, (idx1, idx2) in enumerate(zip(splitIdx1, splitIdx2)):
    aperture = np.zeros(R.shape)
    aperture[np.logical_and(R >= idx1, R < idx2)] = 1
    mskRings[:, :, i] = aperture

# 3rd step: combine quadrant and ring apertures
mskCircleObli = np.empty([xpix, ypix, 2*2*circleSplits])

# combine conditions
iterables = [np.arange(2*2).astype(int),
             np.arange(circleSplits).astype(int)]
Conditions = list(itertools.product(*iterables))

# cross the single diagonal masks to obtain diagonal squares
for i, (idx1, idx2) in enumerate(Conditions):
    # zero all elements above diagonal
    msk1 = mskQuadrantObl[:, :, idx1]
    # zero all elements below diagonal
    msk2 = mskRings[:, :, idx2]
    # combine masks
    msk = np.logical_and(msk1, msk2)

    mskCircleObli[:, :, i] = msk

#test1 = mskCircleObli[:, :, 1]
#imgplot = plt.imshow(test1)

mskCircle = np.concatenate((mskCircleCard, mskCircleObli), axis=2)
# add first image, which will be zero-only image
mskCircle = np.dstack((np.zeros((xpix, ypix)), mskCircle))


# %%
""" (4a) circle bar apertures in cardinal orientation"""
# 1st step: define 4 hemifield apertures

mskHalf = np.empty((xpix, ypix, 4), dtype='int')

# combine conditions
Cond = [(0, 1), (1, 0)]

for i, cond in enumerate(Cond):
    m = np.empty((xpix, ypix), dtype='int')
    # set one side of array to 0
    m[:m.shape[0]/2, :m.shape[1]] = cond[0]
    # set other side of array to 1
    m[m.shape[0]/2:, :m.shape[1]] = cond[1]
    # transpose image to get vertical version
    m2 = np.transpose(m)
    # save to mskHalf
    mskHalf[:, :, 2*i] = m
    mskHalf[:, :, 2*i+1] = m2


# 2nd step: define ring apertures
circleSplits = 4
rmin = np.minimum(xpix, ypix)/2

X, Y = np.meshgrid(
    np.arange(xpix) - xpix/2+0.5,
    np.arange(ypix) - ypix/2+0.5
    )

R = np.sqrt(np.square(X)+np.square(Y))

splitIdx1 = np.arange(0, rmin, rmin/circleSplits)
splitIdx2 = np.arange(0, rmin, rmin/circleSplits) + rmin/circleSplits

mskRings = np.empty([xpix, ypix, circleSplits])
for i, (idx1, idx2) in enumerate(zip(splitIdx1, splitIdx2)):
    aperture = np.zeros(R.shape)
    aperture[np.logical_and(R >= idx1, R < idx2)] = 1
    mskRings[:, :, i] = aperture

# 3rd step: combine quadrant and ring apertures
mskCircleBarCard = np.empty([xpix, ypix, 2*2*circleSplits])

# combine conditions
iterables = [np.arange(2*2).astype(int),
             np.arange(circleSplits).astype(int)]
Conditions = list(itertools.product(*iterables))

# cross the single diagonal masks to obtain diagonal squares
for i, (idx1, idx2) in enumerate(Conditions):
    # zero all elements above diagonal
    msk1 = mskHalf[:, :, idx1]
    # zero all elements below diagonal
    msk2 = mskRings[:, :, idx2]
    # combine masks
    msk = np.logical_and(msk1, msk2)

    mskCircleBarCard[:, :, i] = msk

#test1 = mskCircleCard[:, :, 8]
#imgplot = plt.imshow(test1)


""" (4b) circle bar apertures in oblique orientation"""

# 1st step: define 4 quadrant apertures (this time rotated by 45 degrees)
minOfXY = np.minimum(xpix, ypix)
ndiags = 2 * minOfXY - 1
lowerBound = -ndiags/2
upperBound = ndiags/2
# factor in correction for diagonal
diagCorrFactor = minOfXY/np.sqrt(2*minOfXY**2)
lowerBound = np.floor(lowerBound*diagCorrFactor)
upperBound = np.ceil(upperBound*diagCorrFactor)

# get split indices
diagIndc = np.arange(lowerBound, upperBound)
# use array_split instead of linspace (which would be
# more elegant but does not allow int output currently)
ls = np.array_split(diagIndc, 2)
splitIdx1 = [item[0] for item in ls]
splitIdx2 = [item[-1] for item in ls]

# create empy numpy arrays
mskQuadrantObl1 = np.empty([xpix, ypix, 2])
mskQuadrantObl2 = np.empty([xpix, ypix, 2])

tmpl = np.ones((xpix, ypix))

# create single diagonal masks
for i, (idx1, idx2) in enumerate(zip(splitIdx1, splitIdx2)):
    # zero all elements above diagonal
    msk1 = np.tril(tmpl, idx2)
    # zero all elements below diagonal
    msk2 = np.triu(tmpl, idx1)
    # combine masks
    msk = np.logical_and(msk1, msk2)

    mskQuadrantObl1[:, :, i] = msk
    mskQuadrantObl2[:, :, i] = np.fliplr(msk)

# concatenate
mskQuadrantObl = np.concatenate((mskQuadrantObl1, mskQuadrantObl2), axis=2)

# 2nd step: define ring apertures
circleSplits = 4
rmin = np.minimum(xpix, ypix)/2

X, Y = np.meshgrid(
    np.arange(xpix) - xpix/2+0.5,
    np.arange(ypix) - ypix/2+0.5
    )

R = np.sqrt(np.square(X)+np.square(Y))

splitIdx1 = np.arange(0, rmin, rmin/circleSplits)
splitIdx2 = np.arange(0, rmin, rmin/circleSplits) + rmin/circleSplits

mskRings = np.empty([xpix, ypix, circleSplits])
for i, (idx1, idx2) in enumerate(zip(splitIdx1, splitIdx2)):
    aperture = np.zeros(R.shape)
    aperture[np.logical_and(R >= idx1, R < idx2)] = 1
    mskRings[:, :, i] = aperture

# 3rd step: combine quadrant and ring apertures
mskCircleBarObli = np.empty([xpix, ypix, 2*2*circleSplits])

# combine conditions
iterables = [np.arange(2*2).astype(int),
             np.arange(circleSplits).astype(int)]
Conditions = list(itertools.product(*iterables))

# cross the single diagonal masks to obtain diagonal squares
for i, (idx1, idx2) in enumerate(Conditions):
    # zero all elements above diagonal
    msk1 = mskQuadrantObl[:, :, idx1]
    # zero all elements below diagonal
    msk2 = mskRings[:, :, idx2]
    # combine masks
    msk = np.logical_and(msk1, msk2)

    mskCircleBarObli[:, :, i] = msk

#test1 = mskCircleBarObli[:, :, 15]
#imgplot = plt.imshow(test1)

mskCircleBar = np.concatenate((mskCircleBarCard, mskCircleBarObli), axis=2)
# add first image, which will be zero-only image
mskCircleBar = np.dstack((np.zeros((xpix, ypix)), mskCircleBar))


""" (5a) inverted circle bar apertures in cardinal orientation"""
## 1st step: define 4 hemifield apertures
#mskHalf = np.empty((xpix, ypix, 4), dtype='int')
## combine conditions
#Cond = [(0, 1), (1, 0)]
#
#for i, cond in enumerate(Cond):
#    m = np.empty((xpix, ypix), dtype='int')
#    # set one side of array to 0
#    m[:m.shape[0]/2, :m.shape[1]] = cond[0]
#    # set other side of array to 1
#    m[m.shape[0]/2:, :m.shape[1]] = cond[1]
#    # transpose image to get vertical version
#    m2 = np.transpose(m)
#    # save to mskHalf
#    mskHalf[:, :, 2*i] = m
#    mskHalf[:, :, 2*i+1] = m2
#
## 2nd step: define inverted ring apertures
#circleSplits = 4
#rmin = np.minimum(xpix, ypix)/2
#
#X, Y = np.meshgrid(
#    np.arange(xpix) - xpix/2+0.5,
#    np.arange(ypix) - ypix/2+0.5
#    )
#
#R = np.sqrt(np.square(X)+np.square(Y))
#
#splitIdx1 = np.arange(0, rmin, rmin/circleSplits)
#splitIdx2 = np.arange(0, rmin, rmin/circleSplits) + rmin/circleSplits
#
#mskRings = np.empty([xpix, ypix, circleSplits])
#for i, (idx1, idx2) in enumerate(zip(splitIdx1, splitIdx2)):
#    aperture = np.zeros(R.shape)
#    aperture[np.logical_and(R >= idx1, R < idx2)] = 1
#    mskRings[:, :, i] = aperture
#
#mskCircleBarCard = np.empty([xpix, ypix, 2*2*circleSplits])
#
## 3rd steps combine conditions (halves and rings)
#iterables = [np.arange(2*2).astype(int),
#             np.arange(circleSplits).astype(int)]
#Conditions = list(itertools.product(*iterables))
#
## cross the single diagonal masks to obtain diagonal squares
#for i, (idx1, idx2) in enumerate(Conditions):
#    # zero all elements above diagonal
#    msk1 = mskHalf[:, :, idx1]
#    # zero all elements below diagonal
#    msk2 = mskRings[:, :, idx2]
#    # combine masks
#    msk = np.logical_and(msk1, msk2)
#
#    mskCircleBarCard[:, :, i] = msk
#
## 4th step: roll back to invert
#fraction = 100
#mskInvCircleBarCard = np.empty((xpix, ypix,16))
#elem = np.hstack((np.tile([0], 4), np.tile([1], 4)))
#Conditions = np.tile(elem,2)
#for i, cond in enumerate(Conditions):
#    temp = mskCircleBarCard[:, :, i]
#    mskInvCircleBarCard[:, :, i] = np.roll(temp, 600 + fraction, axis=cond)


#%%
"""Save in a pickle"""
print "start saving arrays..."
folderpath = '/home/marian/Documents/Testing/gif_aperture'
np.save(os.path.join(folderpath, 'mskBar'), mskBar)
np.save(os.path.join(folderpath, 'mskSquare'), mskSquare)
np.save(os.path.join(folderpath, 'mskCircle'), mskCircle)
np.save(os.path.join(folderpath, 'mskCircleBar'), mskCircleBar)

print "saving arrays done"

# save all apertures in dictionary and pickle it

print "start pickling..."
pickleArray = {
    'mskBar': mskBar,
    'mskSquare': mskSquare,
    'mskCircle': mskCircle,
    'mskCircleBar': mskCircleBar,
    }

# save dictionary to pickle
picklePath = os.path.join(folderpath, 'pRFMasks.pickle')
with open(picklePath, 'wb') as handle:
    pickle.dump(pickleArray, handle)
print "pickling done"

#test1 = mskSquare[:, :, 2]
#imgplot = plt.imshow(test1)
