#!/usr/bin/env python3


import functions as fn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mastu import HSV
import os
from scipy import constants
from scipy.optimize import curve_fit


###############################################################################
###                         specifics for the shot                          ###
###############################################################################


### shot number
shots = [
    51713,
    51765,
]
### time range
times = [
    [0.35,0.47], 
    [0.35,0.47],
    # [0.40,0.50],
]
colors = [
    'C7', 'C2',
]
fitType = 'apf'
# fitType = 'fit'


### do I want to show the plots or not
plot = True


### for thomson fitting
R0 = 1.25
### for the plots
xlim = [1.15, 1.50]
yscale = 1.1


###############################################################################
###                              load the data                              ###
###############################################################################


R = np.linspace(xlim[0], xlim[1], 100)

nep0 = (1.37, 1., 0.1, 1e4, 1.)
Tep0 = (1.37, 100., 0.1, 1e4, 10.)
pep0 = (1.37, 100., 0.1, 1e4, 10.)
neBounds = (
    [0.25, 0.01, 0., -1e5, 0.], 
    [1.7, 100., 0.5, 1e5, 10.]
)
TeBounds = (
    [0.25, 1., 0., -1e5, 0.], 
    [1.7, 500., 0.5, 1e5, 100.]
)
peBounds = (
    [0.25, 1., 0., -1e6, 0.], 
    [1.7, 50000., 0.5, 1e6, 100.]
)
neScale = 1e19

# markers = ['.', '^', 's', 'p']


###############################################################################
###                    adding the ADAS function loading                     ###
###############################################################################


# fExcite, scaleExcite, fRecomb, scaleRecomb, fIonise, scaleIonise = fn.makeADAS(
#     line='dalpha', excite='cubic', recomb='cubic', 
#     ionise='linear', bounds_error=False, fill_value=None
# )
# fExcite2, scaleExcite2, fRecomb2, scaleRecomb2, fIonise2, scaleIonise2 = \
# fn.makeADAS(
#     line='dalpha', excite='linear', recomb='linear', 
#     ionise='cubic', bounds_error=False, fill_value=None
# )


###############################################################################
###                             do the plotting                             ###
###############################################################################


### check for directory and make it if need be
savePath = f'/home/sthoma/Documents/Plots/rba/thomsonProfilesMulti/'
if not os.path.exists(savePath):
    os.makedirs(savePath)


### make the figure and axes
fig, ax = plt.subplots(1, 3, figsize=(10,3.6), dpi=150)

### iterate over the times
for i in range(0, len(times)):
    shotn = shots[i]
    trange = times[i]
    color = colors[i]

    marker = '.'

    ### pedestal fitting results
    time, neR0, neHeight, neWidth, neGrad, neBkgd = HSV.getPedestal(
        shotn, 'n_e', trange=trange
    )
    _, TeR0, TeHeight, TeWidth, TeGrad, TeBkgd = HSV.getPedestal(
        shotn, 'T_e', trange=trange
    )
    _, peR0, peHeight, peWidth, peGrad, peBkgd = HSV.getPedestal(
        shotn, 'p_e', trange=trange
    )

    time2, ne, neErr, Rraw = HSV.getThomson(shotn, 'n_e', trange=trange)
    ### check if the time range is fine between the two
    if (len(time) != len(time2)) or (
        not (np.isclose(time, time2).all() and np.isclose(time2, time).all())
        ):
        ### change trange if need be, and reload density
        trange = [time[0],time[-1]]
        _,  ne, neErr, Rraw = HSV.getThomson(shotn, 'n_e', trange=trange)
    _, Te, TeErr, _ = HSV.getThomson(shotn, 'T_e', trange=trange)
    _, pe, peErr, _ = HSV.getThomson(shotn, 'p_e', trange=trange)

    # for j in range(0, len(time2)):
    for j in range(0, len(time)):

    
        ### find the lowest x-value
        rind = HSV.findNearest(Rraw[j], xlim[0]) - 1

        ### find finite values and those above R=1.25m (by default)
        boo = np.isfinite(ne[j]) * np.isfinite(Te[j]) * (Rraw[j] > R0)


        if fitType == 'fit':
            
            try:
                nePopt, nePcov = curve_fit(
                    HSV.mtanh, Rraw[j,boo], ne[j,boo]/neScale, p0=nep0, 
                    sigma=neErr[j,boo]/neScale, absolute_sigma=True, bounds=neBounds
                )
                neFit = HSV.mtanh(R, *nePopt) * neScale
                neBool = True
            except RuntimeError:
                neBool = False

            try:
                TePopt, TePcov = curve_fit(
                    HSV.mtanh, Rraw[j,boo], Te[j,boo], p0=Tep0, 
                    sigma=TeErr[j,boo], absolute_sigma=True, bounds=TeBounds
                )
                TeFit = HSV.mtanh(R, *TePopt)
                TeBool = True
            except RuntimeError:
                TeBool = False

            try:
                pePopt, pePcov = curve_fit(
                    HSV.mtanh, Rraw[j,boo], pe[j,boo], p0=pep0, 
                    sigma=peErr[j,boo], absolute_sigma=True, bounds=peBounds
                )
                peFit = HSV.mtanh(R, *pePopt)
                peBool = True
            except RuntimeError:
                peBool = False


        elif fitType == 'apf':

            neBool = True
            TeBool = True
            peBool = True
            neFit = HSV.mtanh(
                R, neR0[j], neHeight[j], neWidth[j], neGrad[j], neBkgd[j]
            )
            TeFit = HSV.mtanh(
                R, TeR0[j], TeHeight[j], TeWidth[j], TeGrad[j], TeBkgd[j]
            )
            peFit = HSV.mtanh(
                R, peR0[j], peHeight[j], peWidth[j], peGrad[j], peBkgd[j]
            )

        
        
        ### find the maximas for the plotting
        neMax = ne[j,rind:][np.isfinite(ne[j,rind:])].max()
        TeMax = Te[j,rind:][np.isfinite(Te[j,rind:])].max()
        peMax = pe[j,rind:][np.isfinite(pe[j,rind:])].max()
        
        
        ### plot ne
        ax[0].errorbar(
            Rraw[j,rind:], ne[j,rind:], yerr=neErr[j,rind:], 
            c=color, fmt=marker, capsize=0, mfc='None', mew=0.5, 
        )
        if neBool:
            ax[0].plot(R, neFit, '-', c=color, lw=0.8)
        
        ### plot Te
        ax[1].errorbar(
            Rraw[j,rind:], Te[j,rind:], yerr=TeErr[j,rind:], 
            c=color, fmt=marker, capsize=0, mfc='None', mew=0.5, 
        )
        if TeBool:
            ax[1].plot(R, TeFit, '-', c=color, lw=0.8)
        
        ### plot pe
        if (j == 0) or (j == (len(time2) - 1)):
            ax[2].errorbar(
                Rraw[j,rind:], pe[j,rind:], yerr=peErr[j,rind:], mew=0.5, 
                c=color, fmt=marker, capsize=0, mfc='None', label=f'{shotn}, {time2[j]:.4f}s'
            )
        else:
            ax[2].errorbar(
                Rraw[j,rind:], pe[j,rind:], yerr=peErr[j,rind:], 
                c=color, fmt=marker, capsize=0, mfc='None', 
            )
        if peBool:
            ax[2].plot(R, peFit, '-', c=color, lw=0.8)
        
    savePath += f'_{shotn}_{trange[0]:.4f}_{trange[1]:.4f}s'
        
### changing axes labels and limits
ax[0].set_xlabel('$R$ (m)', fontsize=9)
ax[0].set_ylabel('$n_e$ (m$^{-3}$)', fontsize=9)
ax[0].yaxis.get_offset_text().set_fontsize(9)
# ax[0].set_ylim([0, neMax*yscale])
ax[1].set_xlabel('$R$ (m)', fontsize=9)
ax[1].set_ylabel('$T_e$ (eV)', fontsize=9)
# ax[1].set_ylim([0, TeMax*yscale])
ax[2].set_xlabel('$R$ (m)', fontsize=9)
ax[2].set_ylabel('$p_e$ (Pa)', fontsize=9)
ax[2].legend(fontsize=8)
# ax[2].set_ylim([0, peMax*yscale])

# ### do the fill between
ylim = ax[0].get_ylim()
ax[0].fill_between(
    [xlim[0], R0], [ylim[0],ylim[0]], [ylim[1],ylim[1]], 
    color='k', alpha=0.1
)
ax[0].set_ylim(ylim)
ylim = ax[1].get_ylim()
ax[1].fill_between(
    [xlim[0], R0], [ylim[0],ylim[0]], [ylim[1],ylim[1]], 
    color='k', alpha=0.1
)
ax[1].set_ylim(ylim)
ylim = ax[2].get_ylim()
ax[2].fill_between(
    [xlim[0], R0], [ylim[0],ylim[0]], [ylim[1],ylim[1]], 
    color='k', alpha=0.1
)
ax[2].set_ylim(ylim)

    
### figure wide changes
for j in range(0, 3):
    ax[j].set_xlim(xlim)
    ax[j].tick_params(
            axis="both", which='both', labelsize=9, direction='in', 
            left=True, bottom=True, right=True, top=True
        )
    ax[j].xaxis.set_minor_locator(AutoMinorLocator())
    ax[j].yaxis.set_minor_locator(AutoMinorLocator())

### adding a title 
plt.tight_layout()


plt.savefig(savePath + '.png')
    

if plot:
    plt.show()


print('done :)')
