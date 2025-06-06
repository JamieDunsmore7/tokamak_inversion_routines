#!/usr/bin/env python3

### author: Steven Thomas
### email:  steven.thomas@ukaea.uk; sthoma@mit.edu
neutral_version = '3.0.1'


from cpyuda import ServerException as SE
import functions as fn
from mastu import HSV
import numpy as np
from pyEquilibrium.equilibrium import equilibrium as equil
from scipy.optimize import curve_fit
import sys


# saveDir = '/common/BES_analysis/rbaResults/'
saveDir = '/home/sthoma/Documents/Results/rba/'
try:
    if sys.argv[2] == 'True':
        plot = True
    else:
        plot = False
except IndexError:
    plot = False


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
# exposureMult = inputDict['exposureMult']
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
# kindThomson = inputDict['kindThomson']
### R value pedestal fitting works up to
Rprofile0 = inputDict['Rprofile0']

### kind of interpolation to use, don't need to change
line = inputDict['line']
kindExcite = inputDict['kindExcite']
kindRecomb = inputDict['kindRecomb']
kindIonise = inputDict['kindIonise']
### for calcualting ionisation rate and neutral density
kindEmissivity = 'quadratic'
ni = 1.
N = 100

### using a constant percentage for profile error
# percent = inputDict['percent']

### temperature to assume for neutral density from fig
figTemp = inputDict['figTemp']


### load the results from inversion
inversion = np.load(loadFile)
R = inversion['R']
data = inversion['data']
nT, nR = data.shape
err = inversion['err']
mask = inversion['mask']
time = inversion['time']
Rgrid = inversion['Rgrid']
RgridB = inversion['RgridB']
emissivity = inversion['emissivity']
emissivityErr = inversion['emissivityErr']
backprojection = inversion['backprojection']
scale = inversion['scale']
emMax = inversion['emMax']
emMaxErr = inversion['emMaxErr']
emMaxR = inversion['emMaxR']
emR0 = inversion['emR0']
emR1 = inversion['emR1']
emFWHM = inversion['emFWHM']
emFWHMerr = inversion['emFWHMerr']
Rind = int(inversion['Rind'])
Rprofile = inversion['Rprofile']


###############################################################################
###                          getting profile data                           ###
###############################################################################


### load the pedestal fitting parameters
timeProfile, R0Density, heightDensity, widthDensity, \
    gradDensity, bkgdDensity = HSV.getPedestal(shotn, 'n_e')
_, R0Temp, heightTemp, widthTemp, \
    gradTemp, bkgdTemp = HSV.getPedestal(shotn, 'T_e')

### load the pedestal fitting parameters Samples
R0sDensity, heightsDensity, widthsDensity, \
    gradsDensity, bkgdsDensity = HSV.getPedestalSamples(shotn, 'n_e')
R0sTemp, heightsTemp, widthsTemp, \
    gradsTemp, bkgdsTemp = HSV.getPedestalSamples(shotn, 'T_e')

### always load the Thomson data
timeThomson, dataDensity, dataDensityErr, Rthomson = HSV.getThomson(
    shotn, 'n_e'
)
_, dataTemp, dataTempErr, _ = HSV.getThomson(shotn, 'T_e')

### make ADAS data functions
fExcite, scaleExcite, fRecomb, scaleRecomb, fIonise, scaleIonise = fn.makeADAS(
    line=line, excite=kindExcite, recomb=kindRecomb, 
    ionise=kindIonise, bounds_error=False, fill_value=None
)

# ### defining some parameters for fitting the ne and Te profiles
# R0 = 1.25


###############################################################################
###                         iterate for Siz and n0                          ###
###############################################################################


### make empty arrays for Thomson profiles
profileTemp = np.zeros((nT, len(Rprofile)))
profileDensity = np.zeros((nT, len(Rprofile)))
### which Thomson results to use
tindThomson = np.zeros(nT).astype(int)
booThomson = np.zeros((nT, Rthomson.shape[1])).astype(bool)
### where the start and end of the Thomson data is
RstartThomson = np.zeros(nT)
RendThomson = np.zeros(nT)
### make empty arrays for Siz, n0, and errors
ioniseRate = np.zeros((nT, len(Rprofile)))
ioniseErr = np.zeros((nT, len(Rprofile)))
neutralDensity = np.zeros((nT, len(Rprofile)))
neutralErr = np.zeros((nT, len(Rprofile)))
ioniseRate0 = np.zeros((nT, Rthomson.shape[1]))
ionise0Err = np.zeros((nT, Rthomson.shape[1]))
neutralDensity0 = np.zeros((nT, Rthomson.shape[1]))
neutral0Err = np.zeros((nT, Rthomson.shape[1]))
### empty arrays for measurements of the ionisation profile
ioniseMax = np.zeros(nT)
ioniseMaxErr = np.zeros(nT)
ioniseMaxR = np.zeros(nT)
ioniseMax0 = np.zeros(nT)
ioniseMax0Err = np.zeros(nT)
ioniseMax0R = np.zeros(nT)
ioniseR0 = np.zeros(nT)
ioniseR1 = np.zeros(nT)
ioniseFWHM = np.zeros(nT)
ioniseFWHMerr = np.zeros(nT)
### checking for maxima inside the thomson profiles
ioniseMaxInside = np.zeros(nT)
ioniseMaxInsideErr = np.zeros(nT)
ioniseMaxInsideR = np.zeros(nT)
ioniseBoo = np.zeros(nT).astype(bool)
### empty arrays for measurements of the neutral profile
neutralMax = np.zeros(nT)
neutralMaxErr = np.zeros(nT)
neutralMaxR = np.zeros(nT)
neutralMax0 = np.zeros(nT)
neutralMax0Err = np.zeros(nT)
neutralMax0R = np.zeros(nT)
### checking for maxima inside the thomson profiles
neutralMaxInside = np.zeros(nT)
neutralMaxInsideErr = np.zeros(nT)
neutralMaxInsideR = np.zeros(nT)
neutralBoo = np.zeros(nT).astype(bool)
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
sepNeutralDensity = np.zeros(nT)
sepNeutralDensityErr = np.zeros(nT)

### iterate over the times
for i in range(nT):
    print(f'{i}/{nT}, t={time[i]:.6f}s')
    
    ### make profiles for this timestep

    ### for Thomson
    T = fn.findNearest(timeThomson, time[i])
    tindThomson[i] = T
    ### for profiles
    TT = fn.findNearest(timeProfile, time[i])
    
    ### get rid of Infs and NaNs
    boo = np.isfinite(dataTemp[T,]) * \
                np.isfinite(dataDensity[T]) * \
                (Rthomson[T] >= Rprofile[0])
    booThomson[i] = boo
    ### where the last datapoint is
    RstartThomson[i] = Rthomson[T][boo][0]
    RendThomson[i] = Rthomson[T][boo][-1]

    
    ### using pedestal fitting results
    profileTemp[i] = HSV.mtanh(
        Rprofile, R0Temp[TT], heightTemp[TT],
        widthTemp[TT], gradTemp[TT], bkgdTemp[TT]
    )
    profileDensity[i] = HSV.mtanh(
        Rprofile, R0Density[TT], heightDensity[TT],
        widthDensity[TT], gradDensity[TT], bkgdDensity[TT]
    )

    ### make the APF sample profiles
    profilesTemp = HSV.makePedestalSamples(
        Rprofile, R0sTemp[TT], heightsTemp[TT],
        widthsTemp[TT], gradsTemp[TT], bkgdsTemp[TT]
    )
    profilesDensity = HSV.makePedestalSamples(
        Rprofile, R0sDensity[TT], heightsDensity[TT],
        widthsDensity[TT], gradsDensity[TT], bkgdsDensity[TT]
    )

    ### put it into the iteration function
    ioniseRate[i], ioniseErr[i], neutralDensity[i], neutralErr[i] = \
        fn.neutrals(
        emissivity[i,Rind:], emissivityErr[i,Rind:], profileTemp[i], 
        profileDensity[i], profilesTemp, profilesDensity, fExcite, 
        scaleExcite, fRecomb, scaleRecomb, fIonise, scaleIonise, ni=ni
    )

    ### put it into the iteration function
    ioniseRate0[i,boo], ionise0Err[i,boo], neutralDensity0[i,boo], \
    neutral0Err[i,boo] = fn.neutrals0(
        Rprofile, emissivity[i,Rind:], emissivityErr[i,Rind:], 
        Rthomson[T,boo], dataTemp[T,boo], dataDensity[T,boo], 
        dataTempErr[T,boo], dataDensityErr[T,boo], fExcite, scaleExcite, 
        fRecomb, scaleRecomb, fIonise, scaleIonise, 
        kind=kindEmissivity, ni=ni, N=N
    )

    ### maxima before the final thomson point
    insideThomson = np.where(Rprofile <= RendThomson[i])[0]
    finalInd = insideThomson[-1]

    ### find the emissivity maximum and location
    ioniseMax[i] = ioniseRate[i].max()
    ioniseMaxErr[i] = ioniseErr[i][ioniseRate[i].argmax()]
    ioniseMaxR[i] = Rprofile[ioniseRate[i].argmax()]

    ### find the emissivity maximum and location inside the Thomson
    ioniseMaxInside[i] = ioniseRate[i,insideThomson].max()
    ioniseMaxInsideErr[i] = ioniseErr[i,insideThomson][
        ioniseRate[i,insideThomson].argmax()
    ]
    ioniseMaxInsideR[i] = Rprofile[insideThomson][
        ioniseRate[i,insideThomson].argmax()
    ]
    ### 
    if ioniseRate[i,insideThomson].argmax() == finalInd:
        ioniseBoo[i] = True

    ### find the emissivity maximum and location on Thomson only
    ioniseMax0[i] = ioniseRate0[i,boo].max()
    ioniseMax0Err[i] = ionise0Err[i,boo][ioniseRate0[i,boo].argmax()]
    ioniseMax0R[i] = Rthomson[T,boo][ioniseRate0[i,boo].argmax()]

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
    neutralMaxErr[i] = neutralErr[i][neutralDensity[i].argmax()]
    neutralMaxR[i] = Rprofile[neutralDensity[i].argmax()]

    ### find the neutral maximum and location inside the Thomson
    neutralMaxInside[i] = neutralDensity[i,insideThomson].max()
    neutralMaxInsideErr[i] = neutralErr[i,insideThomson][
        neutralDensity[i,insideThomson].argmax()
    ]
    neutralMaxInsideR[i] = Rprofile[insideThomson][
        neutralDensity[i,insideThomson].argmax()
    ]
    ### 
    if neutralDensity[i,insideThomson].argmax() == finalInd:
        neutralBoo[i] = True

    ### find the neutral maximum and location on Thomson only
    neutralMax0[i] = neutralDensity0[i,boo].max()
    neutralMax0Err[i] = neutral0Err[i,boo][neutralDensity0[i,boo].argmax()]
    neutralMax0R[i] = Rthomson[T,boo][neutralDensity0[i,boo].argmax()]

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

    ### making psiN values
    eq = equil(device='MASTU', shot=shotn, time=time[i])
    psiN[i] = eq.psiN(Rprofile, 0.)[0]
    ### where is the separatrix
    sepInd = fn.findNearest(psiN[i], 1.)

    ### separatrix neutral density
    sepNeutralDensity[i] = neutralDensity[i][sepInd]
    sepNeutralDensityErr[i] = neutralErr[i][sepInd]

    ### comparing with FIGs
    figPressure = HSV.calcFigPressure(shotn, time[i], fig='mid')
    figN0[i] = HSV.calcFigDensity(figPressure, T=300.)

    ### ratio of separatrix densities
    neutralRatio[i] = (neutralDensity[i] / profileDensity[i])[sepInd]


###############################################################################
###                             saving the data                             ###
###############################################################################


### add the file version number
inputDict['neutral_version'] = neutral_version

if saveFile:

    ### create directory and change filename if needed
    npzName = loadFile.split('/')[-1].replace('invert', 'neutral')
    npzName = fn.makeFName(saveDir, shotn, npzName, kind='str')
    
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
        emMaxErr = emMaxErr,
        emMaxR = emMaxR,
        emR0 = emR0,
        emR1 = emR1,
        emFWHM = emFWHM,
        emFWHMerr = emFWHMerr,
        Rind = Rind,
        Rprofile = Rprofile,
        profileTemp = profileTemp,
        profileDensity = profileDensity,
        tindThomson = tindThomson, 
        booThomson = booThomson, 
        RstartThomson = RstartThomson,
        RendThomson = RendThomson,
        ioniseRate = ioniseRate,
        ioniseErr = ioniseErr,
        neutralDensity = neutralDensity,
        neutralErr = neutralErr,
        ioniseRate0 = ioniseRate0,
        ionise0Err = ionise0Err,
        neutralDensity0 = neutralDensity0,
        neutral0Err = neutral0Err,
        ioniseMax = ioniseMax,
        ioniseMaxErr = ioniseMaxErr,
        ioniseMaxR = ioniseMaxR,
        ioniseR0 = ioniseR0,
        ioniseR1 = ioniseR1,
        ioniseFWHM = ioniseFWHM,
        ioniseFWHMerr = ioniseFWHMerr,
        ioniseMax0 = ioniseMax0,
        ioniseMax0Err = ioniseMax0Err,
        ioniseMax0R = ioniseMax0R,
        ioniseMaxInside = ioniseMaxInside,
        ioniseMaxInsideErr = ioniseMaxInsideErr,
        ioniseMaxInsideR = ioniseMaxInsideR,
        ioniseBoo = ioniseBoo,
        neutralMax = neutralMax,
        neutralMaxErr = neutralMaxErr,
        neutralMaxR = neutralMaxR,
        neutralMax0 = neutralMax0,
        neutralMax0Err = neutralMax0Err,
        neutralMax0R = neutralMax0R,
        neutralMaxInside = neutralMaxInside,
        neutralMaxInsideErr = neutralMaxInsideErr,
        neutralMaxInsideR = neutralMaxInsideR,
        neutralBoo = neutralBoo,
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
        sepNeutralDensity = sepNeutralDensity,
        sepNeutralDensityErr = sepNeutralDensityErr,
        invert_version = inversion['invert_version'],
        neutral_version = neutral_version,
        mask = mask,
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
    fills = pf.makeFills(mask, R)
    pf.plotResults(
        Rprofile, emissivity[:,Rind:], emissivityErr[:,Rind:], emMaxR, emMax,
        emR0, emR1, emFWHM, emFWHMerr, ioniseRate, ioniseErr, ioniseMax,
        ioniseMaxR, ioniseR0, ioniseR1, ioniseFWHM, ioniseFWHMerr,
        neutralDensity, neutralErr, neutralMax, neutralMaxR, neutralR1,
        neutralR2, neutralR3, neutralWidth1, neutralWidth1err, neutralWidth2,
        neutralWidth2err, neutralWidth3, neutralWidth3err, neutralR1A,
        neutralR2A, neutralR3A, RendThomson, time, fills, figN0=figN0, 
        neutralRatio=neutralRatio, psiN=psiN,
    )
    from matplotlib.pyplot import show
    show()
