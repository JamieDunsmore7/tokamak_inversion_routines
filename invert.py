#!/usr/bin/env python3

### author: Steven Thomas
### email:  steven.thomas@ukaea.uk; sthoma@mit.edu
__version__ = '1.6.0'


import functions as fn
from mastu import HSV
import numpy as np
import sys


# saveDir = '/common/BES_analysis/rbaResults/'
saveDir = '/home/sthoma/Documents/Results/rba/'
plot = False


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

### R value pedestal fitting works up to
Rprofile0 = inputDict['Rprofile0']

### kind of interpolation to use, don't need to change
line = inputDict['line']
kindExcite = inputDict['kindExcite']
kindRecomb = inputDict['kindRecomb']
kindIonise = inputDict['kindIonise']

### using a constant percentage for profile error
percent = inputDict['percent']

### temperature to assume for neutral density from fig
figTemp = inputDict['figTemp']


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
###                      doing the main inversion here                      ###
###############################################################################


### make the regularisation band matrix
D = fn.regulMatrix(nGrid, biasedEdges=biasedEdges)
### making empty arrays for results
emissivity = np.zeros((nT, nGrid-1))
emissivityErr = np.zeros((nT, nGrid-1))
chi2 = np.zeros(nT)
gamma = np.zeros(nT)
backprojection = np.zeros((nT, nR))
### empty arrays for measurements of the emissivity profile
emMax = np.zeros(nT)
emMaxR = np.zeros(nT)
emR0 = np.zeros(nT)
emR1 = np.zeros(nT)
emFWHM = np.zeros(nT)
emFWHMerr = np.zeros(nT)

### arrays of indices, used in each iteration
indLos = slice(0, nR)
indSpace = slice(0, nGrid-1)
### linspace, used in each iteration
Q = np.linspace(0., 1., indLos.stop-indLos.start)

### iterate over the times
for i in range(nT):

    emissivity[i,indSpace], emissivityErr[i,indSpace], backprojection[i], \
        chi2[i], gamma[i] = fn.inversion(
            data[i], err[i], dL, scale, nGrid, Q, D, indLos, indSpace,
            nFisher=nFisher, regGuess=regGuess, regMin=regMin
    )

    ### multiply the answers by scale
    emissivity[i] *= scale
    emissivityErr[i] *= scale

    ### find the emissivity maximum and location
    emMax[i] = emissivity[i].max()
    emMaxR[i] = RgridB[emissivity[i].argmax()]

    ### calculate the emissivity FWHM
    emR0[i], emR1[i] = fn.findPosition(
        RgridB, emissivity[i], 0.5, kind='linear'
    )
    emFWHM[i] = emR1[i] - emR0[i]
    emFWHMerr[i] = fn.findPositionErr(
        RgridB, emissivity[i], emissivityErr[i],
        0.5, x0=emR0[i], x1=emR1[i], kind='linear'
    )


###############################################################################
###                          getting profile data                           ###
###############################################################################


### load the pedestal fitting parameters
timePed, R0Density, heightDensity, widthDensity, \
    gradDensity, bkgdDensity = HSV.getPedestal(shotn, 'n_e')
_, R0Temp, heightTemp, widthTemp, \
    gradTemp, bkgdTemp = HSV.getPedestal(shotn, 'T_e')

### make R array for profiles
Rind = fn.findNearest(RgridB, Rprofile0)
Rprofile = RgridB[Rind:]

### make ADAS data functions
### TODO: Extrapolation arguments are hardcoded
fExcite, scaleExcite, fRecomb, scaleRecomb, fIonise, scaleIonise = fn.makeADAS(
    line=line, excite=kindExcite, recomb=kindRecomb, 
    ionise=kindIonise, bounds_error=False, fill_value=None
)


###############################################################################
###                         iterate for Siz and n0                          ###
###############################################################################


### make empty arrays for Thomson profiles
profileTemp = np.zeros((nT, len(Rprofile)))
profileDensity = np.zeros((nT, len(Rprofile)))
### make empty arrays for Siz, n0, and errors
ioniseRate = np.zeros((nT, len(Rprofile)))
ioniseErr = np.zeros((nT, len(Rprofile)))
neutralDensity = np.zeros((nT, len(Rprofile)))
neutralErr = np.zeros((nT, len(Rprofile)))
### empty arrays for measurements of the ionisation profile
ioniseMax = np.zeros(nT)
ioniseMaxR = np.zeros(nT)
ioniseR0 = np.zeros(nT)
ioniseR1 = np.zeros(nT)
ioniseFWHM = np.zeros(nT)
ioniseFWHMerr = np.zeros(nT)
### empty arrays for measurements of the neutral profile
neutralMax = np.zeros(nT)
neutralMaxR = np.zeros(nT)
### here, I use exponential lengths instead of FWHM
neutralR1 = np.zeros(nT)
neutralR2 = np.zeros(nT)
neutralR3 = np.zeros(nT)
neutralWidth1 = np.zeros(nT)
neutralWidth1err = np.zeros(nT)
neutralWidth2 = np.zeros(nT)
neutralWidth2err = np.zeros(nT)
neutralWidth3 = np.zeros(nT)
neutralWidth3err = np.zeros(nT)
neutralR1A = np.zeros(nT)
neutralR2A = np.zeros(nT)
neutralR3A = np.zeros(nT)
neutralWidth1A = np.zeros(nT)
neutralWidth1Aerr = np.zeros(nT)
neutralWidth2A = np.zeros(nT)
neutralWidth2Aerr = np.zeros(nT)
neutralWidth3A = np.zeros(nT)
neutralWidth3Aerr = np.zeros(nT)
### empty arrays for psiN, fig pressure, and separatrix ratio
psiN = np.zeros((nT, len(Rprofile)))
figN0 = np.zeros(nT)
neutralRatio = np.zeros(nT)

### iterate over the times
for i in range(nT):
    
    ### make profiles for this timestep
    T = fn.findNearest(timePed, time[i])
    profileTemp[i] = HSV.mtanh(
        Rprofile, R0Temp[T], heightTemp[T],
        widthTemp[T], gradTemp[T], bkgdTemp[T]
    )
    profileDensity[i] = HSV.mtanh(
        Rprofile, R0Density[T], heightDensity[T],
        widthDensity[T], gradDensity[T], bkgdDensity[T]
    )

    ### throw it into the iteration function
    ioniseRate[i], ioniseErr[i], neutralDensity[i], neutralErr[i] = \
        fn.neutrals(
            emissivity[i,Rind:], emissivityErr[i,Rind:], profileTemp[i],
            profileDensity[i], fExcite, scaleExcite, fRecomb, scaleRecomb,
            fIonise, scaleIonise, percent=percent
    )

    ### find the emissivity maximum and location
    ioniseMax[i] = ioniseRate[i].max()
    ioniseMaxR[i] = Rprofile[ioniseRate[i].argmax()]

    ### calculate the ionisation FWHM
    ioniseR0[i], ioniseR1[i] = fn.findPosition(
        Rprofile, ioniseRate[i], 0.5, kind='linear'
    )
    ioniseFWHM[i] = ioniseR1[i] - ioniseR0[i]
    ioniseFWHMerr[i] = fn.findPositionErr(
        Rprofile, ioniseRate[i], ioniseErr[i],
        0.5, x0=ioniseR0[i], x1=ioniseR1[i], kind='linear'
    )

    ### find the neutral maximum and location
    neutralMax[i] = neutralDensity[i].max()
    neutralMaxR[i] = Rprofile[neutralDensity[i].argmax()]

    ### calculate the ionisation width, 1 e-folding length
    neutralR1[i], neutralR1A[i] = fn.findPosition(
        Rprofile, neutralDensity[i], -1., kind='exp'
    )
    neutralWidth1[i] = neutralMaxR[i] - neutralR1[i]
    neutralWidth1A[i] = neutralR1A[i] - neutralMaxR[i]
    neutralWidth1err[i], neutralWidth1Aerr[i] = fn.findPositionErr(
        Rprofile, neutralDensity[i], neutralErr[i], -1.,
        x0=neutralR1[i], x1=neutralR1A[i], kind='exp'
    )

    ### calculate the ionisation width, 2 e-folding lengths
    neutralR2[i], neutralR2A[i] = fn.findPosition(
        Rprofile, neutralDensity[i], -2., kind='exp'
    )
    neutralWidth2[i] = neutralMaxR[i] - neutralR2[i]
    neutralWidth2A[i] = neutralR2A[i] - neutralMaxR[i]
    neutralWidth2err[i], neutralWidth2Aerr[i] = fn.findPositionErr(
        Rprofile, neutralDensity[i], neutralErr[i], -2.,
        x0=neutralR2[i], x1=neutralR2A[i], kind='exp'
    )

    ### calculate the ionisation width, 3 e-folding lengths
    neutralR3[i], neutralR3A[i] = fn.findPosition(
        Rprofile, neutralDensity[i], -3., kind='exp'
    )
    neutralWidth3[i] = neutralMaxR[i] - neutralR3[i]
    neutralWidth3A[i] = neutralR3A[i] - neutralMaxR[i]
    neutralWidth3err[i], neutralWidth3Aerr[i] = fn.findPositionErr(
        Rprofile, neutralDensity[i], neutralErr[i], -3.,
        x0=neutralR3[i], x1=neutralR3A[i], kind='exp'
    )

    ### try loops to catch issues with loading HSV data
    try:
        ### get the fig density
        figPressure = HSV.calcFigPressure(shotn, time[i], fig='mid')
        figN0[i] = HSV.calcFigDensity(figPressure, T=300.)
    except:
        print('Unable to get fig pressure')
    try:
        ### get equilibrium data
        psiN[i] = HSV.getPsiN(shotn, time[i], Rprofile, 0.)[0][0,0,:]

        ### get the separatrix n0 / ne ratio
        sepInd = fn.findNearest(psiN[i], 1.)
        neutralRatio[i] = (neutralDensity[i] / profileDensity[i])[sepInd]
    except:
        print('Unable to get psiN and/or neutral Ratio')


###############################################################################
###                             saving the data                             ###
###############################################################################


### add the file version number
inputDict['__version__'] = __version__

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
        emissivity = emissivity,
        emissivityErr = emissivityErr,
        backprojection = backprojection,
        scale = scale,
        emMax = emMax,
        emMaxR = emMaxR,
        emR0 = emR0,
        emR1 = emR1,
        emFWHM = emFWHM,
        emFWHMerr = emFWHMerr,
        Rind = Rind,
        Rprofile = Rprofile,
        profileTemp = profileTemp,
        profileDensity = profileDensity,
        ioniseRate = ioniseRate,
        ioniseErr = ioniseErr,
        neutralDensity = neutralDensity,
        neutralErr = neutralErr,
        ioniseMax = ioniseMax,
        ioniseMaxR = ioniseMaxR,
        ioniseR0 = ioniseR0,
        ioniseR1 = ioniseR1,
        ioniseFWHM = ioniseFWHM,
        ioniseFWHMerr = ioniseFWHMerr,
        neutralMax = neutralMax,
        neutralMaxR = neutralMaxR,
        neutralR1 = neutralR1,
        neutralR2 = neutralR2,
        neutralR3 = neutralR3,
        neutralWidth1 = neutralWidth1,
        neutralWidth1err = neutralWidth1err,
        neutralWidth2 = neutralWidth2,
        neutralWidth2err = neutralWidth2err,
        neutralWidth3 = neutralWidth3,
        neutralWidth3err = neutralWidth3err,
        neutralR1A = neutralR1A,
        neutralR2A = neutralR2A,
        neutralR3A = neutralR3A,
        neutralWidth1A = neutralWidth1A,
        neutralWidth1Aerr = neutralWidth1Aerr,
        neutralWidth2A = neutralWidth2A,
        neutralWidth2Aerr = neutralWidth2Aerr,
        neutralWidth3A = neutralWidth3A,
        neutralWidth3Aerr = neutralWidth3Aerr,
        figN0 = figN0,
        psiN = psiN,
        neutralRatio = neutralRatio,
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
    import plotFunctions as pf
    pf.plotInversion(
        R, data, err, RgridB, emissivity, emissivityErr, backprojection, time,
        emMaxR, emMax, emR0, emR1, emFWHM, emFWHMerr,
    )
    pf.plotResults(
        Rprofile, emissivity[:,Rind:], emissivityErr[:,Rind:], emMaxR, emMax,
        emR0, emR1, emFWHM, emFWHMerr, ioniseRate, ioniseErr, ioniseMax,
        ioniseMaxR, ioniseR0, ioniseR1, ioniseFWHM, ioniseFWHMerr,
        neutralDensity, neutralErr, neutralMax, neutralMaxR, neutralR1,
        neutralR2, neutralR3, neutralWidth1, neutralWidth1err, neutralWidth2,
        neutralWidth2err, neutralWidth3, neutralWidth3err, neutralR1A,
        neutralR2A, neutralR3A, time, figN0=figN0, neutralRatio=neutralRatio,
        psiN=psiN,
    )
    from matplotlib.pyplot import show
    show()
