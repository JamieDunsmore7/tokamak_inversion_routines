#!/usr/bin/env python3

### author: Steven Thomas
### email:  steven.thomas@ukaea.uk; sthoma@mit.edu
neutral_version = '1.7.3'


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
loadFile = dictFile.replace('.txt', '.npz')

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

### use pedestal fitting or raw Thomson
kindThomson = inputDict['kindThomson']
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

### load the results from inversion
inversion = np.load(loadFile)
R = inversion['R']
data = inversion['data']
nT, nR = data.shape
err = inversion['err']
time = inversion['time']
Rgrid = inversion['Rgrid']
RgridB = inversion['RgridB']
emissivity = inversion['emissivity']
emissivityErr = inversion['emissivityErr']
backprojection = inversion['backprojection']
scale = inversion['scale']
emMax = inversion['emMax']
emMaxR = inversion['emMaxR']
emR0 = inversion['emR0']
emR1 = inversion['emR1']
emFWHM = inversion['emFWHM']
emFWHMerr = inversion['emFWHMerr']
Rind = inversion['Rind']
Rprofile = inversion['Rprofile']


###############################################################################
###                          getting profile data                           ###
###############################################################################


### using raw Thomson or Pedestal fitting
if kindThomson == 'fit':
    ### load the pedestal fitting parameters
    timeProfile, R0Density, heightDensity, widthDensity, \
        gradDensity, bkgdDensity = HSV.getPedestal(shotn, 'n_e')
    _, R0Temp, heightTemp, widthTemp, \
        gradTemp, bkgdTemp = HSV.getPedestal(shotn, 'T_e')

### always load the Thomson data
timeThomson, dataDensity, _, Rthomson = HSV.getThomson(shotn, 'n_e')
_, dataTemp, _, _ = HSV.getThomson(shotn, 'T_e')

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
### where the end of the Thomson data is
RendThomson = np.zeros(nT)
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
    T = fn.findNearest(timeThomson, time[i])
    
    ### find where the last Thomson data point is
    ### don't use HFS, LFS only
    rTh = fn.findNearest(Rthomson[T], 1.) # TODO, hardcoded 1.
    xTh = Rthomson[T,rTh:]
    ### get rid of Infs and NaNs
    booThomson = np.isfinite(dataTemp[T,rTh:]) * \
                np.isfinite(dataDensity[T,rTh:])
    ### where the last datapoint is
    RendThomson[i] = xTh[booThomson][-1]
    

    ### handle the pedestalFit or raw data differently
    if kindThomson == 'fit':

        TT = fn.findNearest(timeProfile, time[i])
        ### using pedestal fitting results
        profileTemp[i] = HSV.mtanh(
            Rprofile, R0Temp[TT], heightTemp[TT],
            widthTemp[TT], gradTemp[TT], bkgdTemp[TT]
        )
        profileDensity[i] = HSV.mtanh(
            Rprofile, R0Density[TT], heightDensity[TT],
            widthDensity[TT], gradDensity[TT], bkgdDensity[TT]
        )

    elif kindThomson == 'raw':
        ### using the raw Thomson data

        ### temperature
        yTh = dataTemp[T,rTh:][booThomson]
        left = None
        right = None
        # right = 0.200001
        profileTemp[i] = np.interp(
            Rprofile, xTh[booThomson], yTh, left=left, right=right
            )

        ### density
        yTh = dataDensity[T,rTh:][booThomson]
        left = None
        right = None
        # right = 5.000001e13
        profileDensity[i] = np.interp(
            Rprofile, xTh[booThomson], yTh, left=left, right=right
            )
        ### TODO: hardcoded extrapolation method

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
inputDict['neutral_version'] = neutral_version

if saveFile:

    ### create directory and change filename if needed
    npzName = loadFile.split('/')[-1].replace('invert', 'neutral')
    npzName = fn.makeFName(saveDir, shotn, npzName)
    
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
        RendThomson = RendThomson,
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
    pf.plotResults(
        Rprofile, emissivity[:,Rind:], emissivityErr[:,Rind:], emMaxR, emMax,
        emR0, emR1, emFWHM, emFWHMerr, ioniseRate, ioniseErr, ioniseMax,
        ioniseMaxR, ioniseR0, ioniseR1, ioniseFWHM, ioniseFWHMerr,
        neutralDensity, neutralErr, neutralMax, neutralMaxR, neutralR1,
        neutralR2, neutralR3, neutralWidth1, neutralWidth1err, neutralWidth2,
        neutralWidth2err, neutralWidth3, neutralWidth3err, neutralR1A,
        neutralR2A, neutralR3A, RendThomson, time, figN0=figN0, 
        neutralRatio=neutralRatio, psiN=psiN,
    )
    from matplotlib.pyplot import show
    show()
