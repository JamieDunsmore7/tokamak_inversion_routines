#!/usr/bin/env python3

import functions as fn
from mastu import HSV
import numpy as np
import sys


# saveDir = '/common/BES_analysis/rbaResults/'
saveDir = '/home/sthoma/Documents/Results/rba/'
plot = True


###############################################################################
###                        loading variables and data                       ###
###############################################################################


### load the dictionary
dictFile = sys.argv[1]
inputDict = fn.loadDict(dictFile)

### unzip the dictionary
shotn = inputDict['shotn']
Rzfile = inputDict['Rzfile']

### the inputs to select which bits of video, and averaging
I0 = inputDict['I0']
I1 = inputDict['I1']
J0 = inputDict['J0']
J1 = inputDict['J1']
tend = inputDict['tend']
raverage = inputDict['raverage']
taverage = inputDict['taverage']
T0 = inputDict['T0']
T1 = inputDict['T1']

### for converting to photons and account for vignetting
photons = bool(inputDict['photons'])
exposureMult = inputDict['exposureMult']
vignette = bool(inputDict['vignette'])

### variables for the inversion, should be approximately constant
saveFile = inputDict['saveFile']
rEnd = inputDict['rEnd']
nrMult = inputDict['nrMult']
sysErr = inputDict['sysErr']
biasedEdges = bool(inputDict['biasedEdges'])
nFisher = inputDict['nFisher']
regGuess = inputDict['regGuess']
regMin = inputDict['regMin']


###############################################################################
###                     setting up coordinates and grid                     ###
###############################################################################


### load the Rz coordinates
R0, z0 = HSV.getRz(Rzfile, I0, I1, J0, J1)

### prepping R, adding extra, and flipping
R, goodChans, flipBool = fn.prepR(R0, rEnd)
### grid to interpolate onto
nGrid = int(len(R0) * nrMult)
Rgrid, RgridB = fn.makeGrid(R, nGrid)
### making dL
dL = fn.makeDL(Rgrid, R)


###############################################################################
###                     loading data and pre-processing                     ###
###############################################################################


### load the data from UDA
rawData, time = HSV.get(shotn)

### TODO: currently only good for testing 1 row of pixels
dSlice = (slice(I0,I1), slice(J0,J1), slice(0,len(time)))
data, err = HSV.prepData(
    rawData, dSlice, goodChans, flipBool, rEnd, tend, sysErr=sysErr
)
### average in space and time if wanted
if raverage:
    data = HSV.makeRadialAverage(data, raverage, rEnd)
    err = HSV.makeRadialAverage(err, raverage, rEnd)
if taverage:
    data = HSV.makeTimeAverage(data, taverage)
    err = HSV.makeTimeAverage(err, taverage)

### chop down to selected time range
tSlice = slice(T0, T1)
data = data[tSlice,:]
err = err[tSlice,:]
time = time[tSlice]

### convert to ph m^-2 sr^-1
if photons:
    data, err = HSV.applyCalibration(data, err, inputDict)

### convert to ph m^-2 sr^-1 s^-1
# exposureTime = HSV.getExposure(shotn, mult=exposureMult)
# data /= exposureTime
# err /= exposureTime
data, err = HSV.applyExposure(data, err, shotn, mult=exposureMult)

### apply vignette function
if vignette:
    data, err = HSV.applyVignette(
        data, err, inputDict, shotn, dSlice[:-1], flipBool, rEnd
    )

### sizes of data
nT, nR = data.shape
### scale so not working with large numbers
scale = fn.makeScale(data)


###############################################################################
###                     doing the brunt of the work here                    ###
###############################################################################


### make the regularisation band matrix
D = fn.regulMatrix(nGrid, biasedEdges=biasedEdges)
### making empty arrays for results
y = np.zeros((nT, nGrid-1))
yErr = np.zeros((nT, nGrid-1))
chi2 = np.zeros(nT)
gamma = np.zeros(nT)
backprojection = np.zeros((nT, nR))

### arrays of indices, used in each iteration
indLos = slice(0, nR)
indSpace = slice(0, nGrid-1)
### linspace, used in each iteration
Q = np.linspace(0., 1., indLos.stop-indLos.start)

### iterate over the times
for i in range(nT):
    
    y[i,indSpace], yErr[i,indSpace], backprojection[i], chi2[i], gamma[i] = \
    fn.inversion(data[i], err[i], dL, scale, nGrid, Q, D, indLos, indSpace, 
                nFisher=nFisher, regGuess=regGuess, regMin=regMin
    )

### multiply the answers by scale
y *= scale
yErr *= scale


###############################################################################
###                             saving the data                             ###
###############################################################################


if saveFile:
    
    ### create directory and change filename if needed
    npzName = fn.makeFName(saveDir, shotn, saveFile)
    np.savez(npzName, 
        R = R,
        data = data,
        err = err,
        time = time, 
        Rgrid = Rgrid,
        RgridB = RgridB,
        y = y,
        yErr = yErr,
        backprojection = backprojection,
        scale = scale,
    )
    
    ### make filename for dictionary of inputs
    dictName = npzName.replace('.npz', '.txt')
    fn.saveDict(inputDict, dictName)
    print('\n')
    print('.npz file saved:')
    print('    ', npzName)
    print('dictionary textfile saved:')
    print('    ', dictName)
    print('\n')


###############################################################################
###                            plotting the data                            ###
###############################################################################


if plot:
    fn.plotResults(R, data, err, RgridB, y, yErr, backprojection, time)