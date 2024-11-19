#!/usr/bin/env python3

import functions as fn
from mastu import HSV
import matplotlib.pyplot as plt
import numpy as np
# import MDSPlus as mds
from scipy.linalg import eigh, solve_banded
import sys


##################################
### loading variables and data ###
##################################


device = 'rba'
### load the dictionary
dictFile = sys.argv[1]
inputDict = fn.loadDict(dictFile)
### unzip the dictionary
shotn = inputDict['shotn']
Rzfile = inputDict['Rzfile']
I0 = inputDict['I0']
I1 = inputDict['I1']
J0 = inputDict['J0']
J1 = inputDict['J1']
tind = inputDict['tind']
dtind = inputDict['dtind']
tend = inputDict['tend']

nrMult = inputDict['nrMult']
rEnd = inputDict['rEnd']
sysErr = inputDict['sysErr']
nFisher = inputDict['nFisher']
regGuess = inputDict['regGuess']
regMin = inputDict['regMin']


#######################################
### setting up coordinates and grid ###
#######################################


### load the Rz coordinates
R0, z0 = HSV.getRz(Rzfile, I0, I1, J0, J1)

### prepping R, adding extra, and flipping
R, nR, goodChans, flipBool = fn.prepR(R0, rEnd)
### grid to interpolate onto
Rgrid, nGrid = fn.makeGrid(R, nrMult=nrMult)
### making dL
dL = fn.makeDL(Rgrid, R)


#######################################
### loading data and pre-processing ###
#######################################


### load the data from UDA
rawData, time = HSV.get(shotn)

### TODO: currently only good for testing 1 frame
dSlicePrep = (slice(I0,I1), slice(J0,J1), slice(tind-dtind,tind+dtind))
data = HSV.prepData(rawData, dSlicePrep, goodChans, flipBool, rEnd)
nT, nR = data.shape

###
dSliceBkgd = (slice(I0,I1), slice(J0,J1))
data, err = HSV.offset(data, rawData, dSliceBkgd, tend, nR, sysErr=sysErr)

scale = fn.makeScale(data)




