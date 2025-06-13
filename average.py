#!/usr/bin/env python3


#######################################################################
###                             imports                             ###
#######################################################################


import functions as fn
from mastu import HSV
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


#######################################################################
###                          things I define                        ###
#######################################################################


### starting path
savePath0 = '/home/sthoma/Documents/Plots/rba/'

### how many data points to average in HSV
mean = 15

############################# for plotting ############################
### for the time traces of max_emission and the radius
dx = 0.5e-2

### for brightness
N1 = 19
y1 = 1e19

### for emission
N2 = 19
y2 = 1e19

### for ionisation
N3 = 22
y3 = 1e22

### inputs when loading file
resultsFile = sys.argv[1]
trange = [float(sys.argv[2]),float(sys.argv[3])]

### the dictionary from the results
dictFile = resultsFile.replace('.npz', '.txt')
inputDict = fn.loadDict(dictFile)


#######################################################################
###                           loading data                          ###
#######################################################################


### make shot-number variable
shotn = resultsFile.split('/')[-2]

### make the folder to save the results
fileName = resultsFile.split('/')[-1][:-4]
savePath = \
    f"{savePath0}/{shotn}/{fileName}/atThomsonTime/"
if not os.path.exists(savePath):
    os.makedirs(savePath)

### the results .npz file
results = np.load(resultsFile)

### big load
R = results['R']
data = results['data']
dataErr = results['err']
time = results['time']
mask = results['mask']
Rprofile = results['Rprofile']
emissivity = results['emissivity']
emissivityErr = results['emissivityErr']
scale = float(results['scale'])
profileTemp = results['profileTemp']
profileDensity = results['profileDensity']
tindThomson = results['tindThomson']
booThomson = results['booThomson']
RstartThomson = results['RstartThomson']
RendThomson = results['RendThomson']
ioniseRate = results['ioniseRate']
ioniseErr = results['ioniseErr']
neutralDensity = results['neutralDensity']
neutralErr = results['neutralErr']
ioniseRate0 = results['ioniseRate0']
ionise0Err = results['ionise0Err']
neutralDensity0 = results['neutralDensity0']
neutral0Err = results['neutral0Err']
emMax = results['emMax']
emMaxErr = results['emMaxErr']
emMaxR = results['emMaxR']
sizMax = results['ioniseMax']
sizMaxErr = results['ioniseMaxErr']
sizMaxR = results['ioniseMaxR']
RgridB = results['RgridB']
psiN = results['psiN']
n0Sep = results['sepNeutralDensity']
n0SepErr = results['sepNeutralDensityErr']

nFisher = inputDict['nFisher']
regGuess = inputDict['regGuess']
regMin = inputDict['regMin']




#######################################################################
###                    loading suplimentary data                    ###
#######################################################################


### Thomson data
timeThomson, dataDensity, dataDensityErr, \
    RThomson = HSV.getThomson(shotn, 'n_e')
_, dataTemp, dataTempErr, _ = HSV.getThomson(shotn, 'T_e')

### pedestal fitting time points
timeProfile = HSV.getPedestal(shotn, 'n_e')[0]

### load the pedestal fitting Samples
R0sDensity, heightsDensity, widthsDensity, \
    gradsDensity, bkgdsDensity = HSV.getPedestalSamples(shotn, 'n_e')
R0sTemp, heightsTemp, widthsTemp, \
    gradsTemp, bkgdsTemp = HSV.getPedestalSamples(shotn, 'T_e')

### the rate coefficients
fExcite, scaleExcite, fRecomb, \
scaleRecomb, fIonise, scaleIonise = fn.makeADAS(
    line=inputDict['line'], excite=inputDict['kindExcite'], 
    recomb=inputDict['kindRecomb'], ionise=inputDict['kindIonise'], 
    bounds_error=False, fill_value=None
)


#######################################################################
###                   just needs calculating once                   ###
#######################################################################


###
nR = data.shape[1]
nGrid = int(
    (np.where(mask)[0][-1] - np.where(mask)[0][0]) * inputDict['nrMult']
)
Rgrid, RgridB = fn.makeGrid(R[mask], nGrid)
dL = fn.makeDL(Rgrid, R[mask])
D = fn.regulMatrix(nGrid, biasedEdges=bool(inputDict['biasedEdges']))
indLos = slice(0, nR)
indSpace = slice(0, nGrid-1)
Q = np.linspace(0., 1., np.count_nonzero(mask))

###
dt = np.diff(time).min() * mean


#######################################################################
###                    iterate over Thomson time                    ###
#######################################################################


### 
boo = (timeThomson >= trange[0]) * (timeThomson <= trange[1])

### some empty arrays
tinds = np.zeros(np.count_nonzero(boo)).astype(int)
Ts = np.zeros(np.count_nonzero(boo)).astype(int)
TTs = np.zeros(np.count_nonzero(boo)).astype(int)


### iterate
for i, twant in enumerate(timeThomson[boo]):
    print(f'{i}/{len(timeThomson[boo])}')

    ### find the indices for each time array
    tind = fn.findNearest(time, twant)
    T = fn.findNearest(timeThomson, twant)
    TT = fn.findNearest(timeProfile, twant)
    tinds[i] = tind
    Ts[i] = T
    TTs[i] = TT
    

    ### calculate the separatrix for plotting
    sepR = RgridB[fn.findNearest(psiN[tind], 1.)]


    ### average the brightnesses
    averageBrightness = np.mean(
        data[tind-mean:tind+mean], axis=0
    )
    averageBrightnessStd = np.std(
        data[tind-mean:tind+mean], axis=0
    )
    ### average over the emissivities
    averageEmissivity = np.mean(
        emissivity[tind-mean:tind+mean], axis=0
    )
    averageEmissivityStd = np.std(
        emissivity[tind-mean:tind+mean], axis=0
    )
    ### ionisation
    averageIonisation = np.mean(
        ioniseRate[tind-mean:tind+mean], axis=0
    )
    averageIonisationStd = np.std(
        ioniseRate[tind-mean:tind+mean], axis=0
    )
    ### neutral density
    averageNeutral = np.mean(
        neutralDensity[tind-mean:tind+mean], axis=0
    )
    averageNeutralStd = np.std(
        neutralDensity[tind-mean:tind+mean], axis=0
    )


    ### invert the average brightness
    emMany, emManyErr, backprojection, chi2, gamma = fn.inversion(
        averageBrightness[mask], averageBrightnessStd[mask], 
        dL, scale, nGrid, Q, D, indLos, indSpace, 
        nFisher=inputDict['nFisher'], 
        regGuess=inputDict['regGuess'], 
        regMin=inputDict['regMin']
    )

    ### the emissivity from the average of the brightnesses
    emMany *= scale
    emManyErr *= scale


    ### profile samples
    profilesTemp = HSV.makePedestalSamples(
        Rprofile, R0sTemp[TT], heightsTemp[TT],
        widthsTemp[TT], gradsTemp[TT], bkgdsTemp[TT]
    )
    profilesDensity = HSV.makePedestalSamples(
        Rprofile, R0sDensity[TT], heightsDensity[TT],
        widthsDensity[TT], gradsDensity[TT], bkgdsDensity[TT]
    )


    ### average of the brightnesses
    ioniseMany, ioniseManyErr, neutralMany, neutralManyErr = fn.neutrals(
        emMany, emManyErr, profileTemp[tind], profileDensity[tind], 
        profilesTemp, profilesDensity, fExcite, scaleExcite, fRecomb, 
        scaleRecomb, fIonise, scaleIonise, ni=1.
    )


    ### average brightness at Thomson points
    ioniseMany0, ioniseMany0Err, neutralMany0, \
    neutralMany0Err = fn.neutrals0(
        Rprofile, emMany, emManyErr, 
        RThomson[T,booThomson[tind]], dataTemp[T,booThomson[tind]], 
        dataDensity[T,booThomson[tind]], dataTempErr[T,booThomson[tind]], 
        dataDensityErr[T,booThomson[tind]], fExcite, scaleExcite, 
        fRecomb, scaleRecomb, fIonise, scaleIonise, 
        kind='quadratic', ni=1., N=100
    )


    ### ionisation0 and neutral0
    IONISATION0 = np.zeros((mean*2, len(RThomson[tindThomson[tind]][booThomson[tind]])))
    NEUTRAL0 = np.zeros((mean*2, len(RThomson[tindThomson[tind]][booThomson[tind]])))
    for j in range(tind-mean, tind+mean):
        IONISATION0[j-tind] = ioniseRate0[j][booThomson[j]]
        NEUTRAL0[j-tind] = neutralDensity0[j][booThomson[j]]
    
    ### ionisation0
    averageIonisation0 = np.mean(IONISATION0, axis=0)
    averageIonisation0Std = np.std(IONISATION0, axis=0)
    ### neutral0
    averageNeutral0 = np.mean(NEUTRAL0, axis=0)
    averageNeutral0Std = np.std(NEUTRAL0, axis=0)


    ###############################################################
    ###                         plotting                        ###
    ###############################################################


    fig, ax = plt.subplots(3, 2, figsize=(6,7.5), dpi=150)

    ###############################################################

    ax[0,0].plot(time, emMax/y1, c='k', lw=0.8, label=shotn)
    ax[0,0].fill_between(
        time, (emMax-emMaxErr)/y1, 
        (emMax+emMaxErr)/y1, color='k', alpha=0.25
    )

    ax[0,0].tick_params(
        axis="both", which='both', labelsize=9, direction='in', 
        left=True, bottom=True, right=True, top=True
    )
    ax[0,0].minorticks_on()
    ax[0,0].yaxis.get_offset_text().set_size(9)
    ax[0,0].set_xlabel('time (s)', fontsize=9)
    ax[0,0].set_ylabel(
        f'$\\epsilon_\\mathrm{{max}}$ ' + \
        f'($\\times$10$^{{{N2}}}$ ph sr$^{{-1}}$ m$^{{-3}}$ s$^{{-1}}$)', 
        fontsize=9)
    ax[0,0].set_xlim([twant-dx, twant+dx])
    y0 = emMax[tind-mean:tind+mean]/y1
    ax[0,0].set_ylim([y0.min()/3., y0.max()*2.])

    ylim = ax[0,0].get_ylim()
    ax[0,0].vlines(
        timeThomson, ylim[0], ylim[1], color='C2', lw=0.8, zorder=0
    )
    ax[0,0].vlines(
        timeThomson-dt, ylim[0], ylim[1], color='C2', lw=0.4, zorder=0
    )
    ax[0,0].vlines(
        timeThomson+dt, ylim[0], ylim[1], color='C2', lw=0.4, zorder=0
    )

    ###############################################################

    ax[1,0].plot(time, emMaxR, c='k', lw=0.8)
    ax[1,0].tick_params(
        axis="both", which='both', labelsize=9, direction='in', 
        left=True, bottom=True, right=True, top=True
    )
    ax[1,0].minorticks_on()
    ax[1,0].yaxis.get_offset_text().set_size(9)
    ax[1,0].set_xlabel('time (s)', fontsize=9)
    ax[1,0].set_ylabel('$R(\\epsilon_\\mathrm{max})$ (m)', fontsize=9)
    ax[1,0].set_xlim([twant-dx, twant+dx])
    y0 = emMaxR[tind-mean:tind+mean]
    ax[1,0].set_ylim([y0.min()/1.01, y0.max()*1.01])
    ylim = ax[1,0].get_ylim()
    ax[1,0].vlines(
        timeThomson, ylim[0], ylim[1], color='C2', lw=0.8, zorder=0
    )
    ax[1,0].vlines(
        timeThomson-dt, ylim[0], ylim[1], color='C2', lw=0.4, zorder=0
    )
    ax[1,0].vlines(
        timeThomson+dt, ylim[0], ylim[1], color='C2', lw=0.4, zorder=0
    )

    ###############################################################

    ax[2,0].plot(R[mask], data[tind][mask]/y1, c='k', lw=0.8)
    ax[2,0].fill_between(
        R[mask], (data[tind][mask]-dataErr[tind][mask])/y1, 
        (data[tind][mask]+dataErr[tind][mask])/y1, color='k', alpha=0.1
    )
    ax[2,0].plot(
        R[mask], averageBrightness[mask]/y1, c='C1', lw=0.8
    )
    ax[2,0].fill_between(
        R[mask], (averageBrightness[mask]-averageBrightnessStd[mask])/y1, 
        (averageBrightness[mask]+averageBrightnessStd[mask])/y1, 
        color='C1', alpha=0.1
    )

    ax[2,0].tick_params(
        axis="both", which='both', labelsize=9, direction='in', 
        left=True, bottom=True, right=True, top=True
    )
    ax[2,0].minorticks_on()
    ax[2,0].yaxis.get_offset_text().set_size(9)
    ax[2,0].set_xlabel('$R$ (m)', fontsize=9)
    ax[2,0].set_ylabel(
        f'Brightness ($\\times$10$^{{{N1}}}$ ' + \
        f'ph sr$^{{-1}}$ m$^{{-2}}$ s$^{{-1}}$)', 
        fontsize=9
    )
    ax[2,0].set_xlim([Rprofile[0]/1.015, Rprofile[-1]*1.015])
    ax[2,0].set_ylim([0,data[tind][mask].max()/y1*1.3])
    ax[2,0].plot(
        [sepR,sepR], ax[2,0].get_ylim(), '-', lw=0.8, c='C7', zorder=-1
    )

    ###############################################################

    ax[0,1].plot(
        Rprofile, emissivity[tind]/y2, c='k', lw=0.8, label='1 time point'
    )
    ax[0,1].fill_between(
        Rprofile, (emissivity[tind]-emissivityErr[tind])/y2, 
        (emissivity[tind]+emissivityErr[tind])/y2, color='k', alpha=0.1
    )
    ax[0,1].plot(
        Rprofile, averageEmissivity/y2, c='C0', lw=0.8, 
        label=f'Average\nafter inversion'
    )
    ax[0,1].fill_between(
        Rprofile, (averageEmissivity-averageEmissivityStd)/y2, 
        (averageEmissivity+averageEmissivityStd)/y2, color='C0', alpha=0.1
    )
    ax[0,1].plot(
        Rprofile, emMany/y2, c='C1', lw=0.8, 
        label='Average before\ninversion'
    )
    ax[0,1].fill_between(
        Rprofile, (emMany-emManyErr)/y2, 
        (emMany+emManyErr)/y2, color='C1', alpha=0.1
    )

    ax[0,1].legend(
        fancybox=1, framealpha=1, handlelength=0, 
        handletextpad=0, labelcolor='linecolor', fontsize=8
    )
    ax[0,1].tick_params(
        axis="both", which='both', labelsize=9, direction='in', 
        left=True, bottom=True, right=True, top=True
    )
    ax[0,1].minorticks_on()
    ax[0,1].yaxis.get_offset_text().set_size(9)
    ax[0,1].set_xlabel('$R$ (m)', fontsize=9)
    ax[0,1].set_ylabel(
        f'$\\epsilon$ ($\\times$10$^{{{N2}}}$ ph ' + \
        f'sr$^{{-1}}$ m$^{{-3}}$ s$^{{-1}}$)', 
        fontsize=9)
    ax[0,1].set_xlim([Rprofile[0]/1.015, Rprofile[-1]*1.015])
    ax[0,1].set_ylim([0, emissivity[tind].max()/y2*1.3])
    ax[0,1].plot(
        [sepR,sepR], ax[0,1].get_ylim(), '-', lw=0.8, c='C7', zorder=-1
    )

    ###############################################################

    ax[1,1].plot(
        Rprofile, ioniseRate[tind]/y3, c='k', lw=0.8,
    )
    ax[1,1].fill_between(
        Rprofile, (ioniseRate[tind]-ioniseErr[tind])/y3, 
        (ioniseRate[tind]+ioniseErr[tind])/y3, 
        color='k', alpha=0.1
    )
    ax[1,1].plot(
        Rprofile, averageIonisation/y3, '-', c='C0', lw=0.8,
    )
    ax[1,1].fill_between(
        Rprofile, (averageIonisation-averageIonisationStd)/y3, 
        (averageIonisation+averageIonisationStd)/y3, 
        color='C0', alpha=0.1
    )
    ax[1,1].plot(
        Rprofile, ioniseMany/y3, '-', c='C1', lw=0.8, 
    )
    ax[1,1].fill_between(
        Rprofile, (ioniseMany-ioniseManyErr)/y3, 
        (ioniseMany+ioniseManyErr)/y3, color='C1', alpha=0.1
    )

    ax[1,1].errorbar(
        RThomson[tindThomson[tind]][booThomson[tind]], 
        ioniseRate0[tind][booThomson[tind]]/y3, 
        yerr=ionise0Err[tind][booThomson[tind]]/y3, fmt='.', 
        c='k', mfc='None', elinewidth=0.8, alpha=0.8
    )
    ax[1,1].errorbar(
        RThomson[tindThomson[tind]][booThomson[tind]], 
        averageIonisation0/y3, yerr=averageIonisation0Std/y3, 
        fmt='.', c='C0', mfc='None', 
        elinewidth=0.8, alpha=0.8,
    )
    ax[1,1].errorbar(
        RThomson[tindThomson[tind]][booThomson[tind]], 
        ioniseMany0/y3, yerr=ioniseMany0Err/y3, fmt='.', c='C1', 
        mfc='None', elinewidth=0.8, alpha=0.8, 
    )

    ax[1,1].tick_params(
        axis="both", which='both', labelsize=9, direction='in', 
        left=True, bottom=True, right=True, top=True
    )
    ax[1,1].minorticks_on()
    ax[1,1].yaxis.get_offset_text().set_size(9)
    ax[1,1].set_xlabel('$R$ (m)', fontsize=9)
    ax[1,1].set_ylabel(
        f'$S_\\mathrm{{iz}}$ ' + \
        f'($\\times$10$^{{{N3}}}$ m$^{{-3}}$ s$^{{-1}}$)', 
        fontsize=9
    )
    ax[1,1].set_xlim([Rprofile[0]/1.015, Rprofile[-1]*1.015])
    ax[1,1].set_ylim(
        [0, ioniseRate0[tind][booThomson[tind]].max()/y3*1.3]
    )
    ax[1,1].plot(
        [sepR,sepR], ax[1,1].get_ylim(), '-', lw=0.8, c='C7', zorder=-1
    )

    ###############################################################

    ax[2,1].plot(
        Rprofile, neutralDensity[tind], c='k', lw=0.8, 
    )
    ax[2,1].fill_between(
        Rprofile, neutralDensity[tind]-neutralErr[tind], 
        neutralDensity[tind]+neutralErr[tind], 
        color='k', alpha=0.1
    )
    ax[2,1].plot(
        Rprofile, averageNeutral, '-', c='C0', lw=0.8,
    )
    ax[2,1].fill_between(
        Rprofile, averageNeutral-averageNeutralStd, 
        averageNeutral+averageNeutralStd, 
        color='C0', alpha=0.1
    )
    ax[2,1].plot(
        Rprofile, neutralMany, '-', c='C1', lw=0.8, 
    )
    ax[2,1].fill_between(
        Rprofile, neutralMany-neutralManyErr, 
        neutralMany+neutralManyErr, color='C1', alpha=0.1
    )

    ax[2,1].errorbar(
        RThomson[tindThomson[tind]][booThomson[tind]], 
        neutralDensity0[tind][booThomson[tind]], 
        yerr=neutral0Err[tind][booThomson[tind]], 
        fmt='.', c='k', mfc='None', 
        elinewidth=0.8, alpha=0.8, 
    )

    ax[2,1].errorbar(
        RThomson[tindThomson[tind]][booThomson[tind]], 
        averageNeutral0, yerr=averageNeutral0Std, 
        fmt='.', c='C0', mfc='None', 
        elinewidth=0.8, alpha=0.8, 
    )

    ax[2,1].errorbar(
        RThomson[tindThomson[tind]][booThomson[tind]], 
        neutralMany0, yerr=neutralMany0Err, 
        fmt='.', c='C1', mfc='None', 
        elinewidth=0.8, alpha=0.8, 
    )

    ax[2,1].tick_params(
        axis="both", which='both', labelsize=9, direction='in', 
        left=True, bottom=True, right=True, top=True
    )
    ax[2,1].minorticks_on()
    ax[2,1].yaxis.get_offset_text().set_size(9)
    ax[2,1].set_xlabel('$R$ (m)', fontsize=9)
    ax[2,1].set_ylabel('$n_0$ (m$^{-3}$)', fontsize=9)
    ax[2,1].set_yscale('log')
    ax[2,1].set_xlim([Rprofile[0]/1.015, Rprofile[-1]*1.015])
    ax[2,1].set_ylim(
        [1e14, neutralDensity0[tind][booThomson[tind]].max()*10.]
    )
    ax[2,1].plot(
        [sepR,sepR], ax[2,1].get_ylim(), '-', lw=0.8, c='C7', zorder=-1
    )

    ###############################################################

    fig.suptitle(
        f'shot # {shotn}, t={time[tind]:.6f}s, tind={tind}$\\pm${mean}', 
        fontsize=9
    )

    plt.tight_layout()

    plt.savefig(f'{savePath}time_{twant:.6f}s.png')

    plt.close()


fig, ax = plt.subplots(1, 1, figsize=(3.5,3), dpi=150)
ax.plot(time[tinds], emMax[tinds], '-', c='C0', lw=0.8)
ax.set_xlabel('$t$ (s)', fontsize=9)
ax.set_ylabel('$\\epsilon_\\mathrm{max}$ (ph sr$^{-1}$ m$^{-3}$ $s^{-1}$)', fontsize=9)
ax.tick_params(
    axis="both", which='both', labelsize=9, direction='in', 
    left=True, bottom=True, right=True, top=True
)
ax.minorticks_on()
ax.yaxis.get_offset_text().set_size(9)
plt.tight_layout()
plt.savefig(f'{savePath}01_emMax.png')


fig, ax = plt.subplots(1, 1, figsize=(3.5,3), dpi=150)
ax.plot(time[tinds], sizMax[tinds], '-', c='C1', lw=0.8)
ax.set_xlabel('$t$ (s)', fontsize=9)
ax.set_ylabel('$S_\\mathrm{iz}$ (m$^{-3}$ $s^{-1}$)', fontsize=9)
ax.tick_params(
    axis="both", which='both', labelsize=9, direction='in', 
    left=True, bottom=True, right=True, top=True
)
ax.minorticks_on()
ax.yaxis.get_offset_text().set_size(9)
plt.tight_layout()
plt.savefig(f'{savePath}02_sizMax.png')


fig, ax = plt.subplots(1, 1, figsize=(3.5,3), dpi=150)
ax.plot(time[tinds], n0Sep[tinds], '-', c='C2', lw=0.8)
ax.set_xlabel('$t$ (s)', fontsize=9)
ax.set_ylabel('$n_0$ (m$^{-3}$)', fontsize=9)
ax.tick_params(
    axis="both", which='both', labelsize=9, direction='in', 
    left=True, bottom=True, right=True, top=True
)
ax.minorticks_on()
ax.yaxis.get_offset_text().set_size(9)
plt.tight_layout()
plt.savefig(f'{savePath}03_n0Max.png')


plt.show()


print('all done :)')