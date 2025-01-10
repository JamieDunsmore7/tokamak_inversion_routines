#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mastu import HSV
import os


###############################################################################
###                         specifics for the shot                          ###
###############################################################################


### shot number
shotn = 50499
### time range
trange = [0.2,1.0]

### for the plots
xlim = [1.25, 1.50]
yscale = 1.1
### where to save
savePath = f'/home/sthoma/Documents/Plots/rba/{shotn}/thomsonProfiles/'
### check for directory and make it if need be
if not os.path.exists(savePath):
    os.makedirs(savePath)


###############################################################################
###                              load the data                              ###
###############################################################################


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
### R for the pedestal profiles
R = np.linspace(xlim[0], xlim[1], 100)
### raw thomson data
_, ne, neErr, Rraw = HSV.getThomson(shotn, 'n_e', trange=trange)
_, Te, TeErr, _ = HSV.getThomson(shotn, 'T_e', trange=trange)
_, pe, peErr, _ = HSV.getThomson(shotn, 'p_e', trange=trange)


###############################################################################
###                             do the plotting                             ###
###############################################################################


### iterate over the times
for i in range(0, len(time)):
    
    ### find the lowest x-value
    rind = HSV.findNearest(Rraw, xlim[0]) - 1
    ### make the profiles for the plots
    neProfile = HSV.mtanh(
        R, neR0[i], neHeight[i], neWidth[i], neGrad[i], neBkgd[i]
    )
    TeProfile = HSV.mtanh(
        R, TeR0[i], TeHeight[i], TeWidth[i], TeGrad[i], TeBkgd[i]
    )
    peProfile = HSV.mtanh(
        R, peR0[i], peHeight[i], peWidth[i], peGrad[i], peBkgd[i]
    )
    
    ### find the maximas for the plotting
    neMax = ne[i,rind:][np.isfinite(ne[i,rind:])].max()
    TeMax = Te[i,rind:][np.isfinite(Te[i,rind:])].max()
    peMax = pe[i,rind:][np.isfinite(pe[i,rind:])].max()
    
    ### make the figure and axes
    fig, ax = plt.subplots(1, 3, figsize=(8,3.6), dpi=150)
    
    ### plot ne
    ax[0].errorbar(
        Rraw[i,rind:], ne[i,rind:], yerr=neErr[i,rind:], 
        c='C0', fmt='None', capsize=4
    )
    ax[0].plot(R, neProfile, '--k')
    ax[0].text(xlim[1]-0.01, neMax*(yscale-0.05), 
        f'$R_0$: {neR0[i]:.3f}m\n' + f'$h$: {neHeight[i]:.3g}m$^{{-3}}$\n' + \
        f'$w$: {neWidth[i]:.3f}m\n' + f'd$_Rn_e$: {neGrad[i]:.3g}m$^{{-2}}$\n' + 
        f'bkgd: {neBkgd[i]:.3g}m$^{{-3}}$', ha='right', va='top', fontsize=8
    )
    
    ### plot Te
    ax[1].errorbar(
        Rraw[i,rind:], Te[i,rind:], yerr=TeErr[i,rind:], 
        c='C1', fmt='None', capsize=4
    )
    ax[1].plot(R, TeProfile, '--k')
    ax[1].text(xlim[1]-0.01, TeMax*(yscale-0.05), 
        f'$R_0$: {TeR0[i]:.3f}m\n' + f'$h$: {TeHeight[i]:.1f}eV\n' + \
        f'$w$: {TeWidth[i]:.3f}m\n' + f'd$_RT_e$: {TeGrad[i]:.0f}eV/m\n' + 
        f'bkgd: {TeBkgd[i]:.2f}eV', ha='right', va='top', fontsize=8
    )
    
    ### plot Te
    ax[2].errorbar(
        Rraw[i,rind:], pe[i,rind:], yerr=peErr[i,rind:], 
        c='C2', fmt='None', capsize=4
    )
    ax[2].plot(R, peProfile, '--k')
    ax[2].text(xlim[1]-0.01, peMax*(yscale-0.05), 
        f'$R_0$: {peR0[i]:.3f}m\n' + f'$h$: {peHeight[i]:.1f}Pa\n' + \
        f'$w$: {peWidth[i]:.3f}m\n' + f'd$_Rp_e$: {peGrad[i]:.0f}Pa/m\n' + 
        f'bkgd: {peBkgd[i]:.2f}Pa', ha='right', va='top', fontsize=8
    )
    
    ### changing axes labels and limits
    ax[0].set_xlabel('$R$ (m)')
    ax[0].set_ylabel('$n_e$ (m$^{-3}$)')
    ax[0].set_ylim([0, neMax*yscale])
    ax[1].set_xlabel('$R$ (m)')
    ax[1].set_ylabel('$T_e$ (eV)')
    ax[1].set_ylim([0, TeMax*yscale])
    ax[2].set_xlabel('$R$ (m)')
    ax[2].set_ylabel('$p_e$ (Pa)')
    ax[2].set_ylim([0, peMax*yscale])
    
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
    fig.suptitle(f'shot # {shotn}, t={time[i]:.6f}s')
    plt.tight_layout()
    plt.savefig(savePath + f't_{time[i]:.4f}s.png')
    
    # plt.show()
    plt.close()


print('done :)')
