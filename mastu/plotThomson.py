#!/usr/bin/env python3


import functions as fn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mastu import HSV
import os
from pyEquilibrium.equilibrium import equilibrium as equil
from scipy import constants
from scipy.optimize import curve_fit


###############################################################################
###                         specifics for the shot                          ###
###############################################################################


### shot number
shotn = 51706
### time range
trange = [0.2,0.7]
### do I want to show the plots or not
plot = False


### for thomson fitting
R0 = 1.25
### for the plots
xlim = [1.15, 1.50]
yscale = 1.1
### where to save
savePath = f'/home/sthoma/Documents/Plots/rba/{shotn}/thomsonProfiles/'
savePath2 = f'/home/sthoma/Documents/Plots/rba/{shotn}/ADASProfiles/'
### check for directory and make it if need be
if not os.path.exists(savePath):
    os.makedirs(savePath)
if not os.path.exists(savePath2):
    os.makedirs(savePath2)


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

### bounds for the fitting
# neBounds = (-np.inf, np.inf)
# TeBounds = (-np.inf, np.inf)
# peBounds = (-np.inf, np.inf)
neBounds = (
    [0.25, 0.01, 0., -1e4, 0.], 
    [1.7, 100., 0.2, 1e4, 10.]
)
TeBounds = (
    [0.25, 1., 0., -1e4, 0.], 
    [1.7, 500., 0.2, 1e4, 100.]
)
peBounds = (
    [0.25, 1., 0., -1e5, 0.], 
    [1.7, 5000., 0.2, 1e5, 100.]
)
neScale = 1e19


###############################################################################
###                    adding the ADAS function loading                     ###
###############################################################################


fExcite, scaleExcite, fRecomb, scaleRecomb, fIonise, scaleIonise = fn.makeADAS(
    line='dalpha', excite='cubic', recomb='cubic', 
    ionise='cubic', bounds_error=False, fill_value=None
)
fExcite2, scaleExcite2, fRecomb2, scaleRecomb2, fIonise2, scaleIonise2 = \
fn.makeADAS(
    line='dalpha', excite='linear', recomb='linear', 
    ionise='linear', bounds_error=False, fill_value=None
)


###############################################################################
###                             do the plotting                             ###
###############################################################################
RpsiN = np.arange(1., 1.6, 1e-4)

### iterate over the times
for i in range(0, len(time)):
    print(f'i={i}/{len(time)}, t={time[i]:.6f}s')

    
    ### find the lowest x-value
    rind = HSV.findNearest(Rraw[i], xlim[0]) - 1

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


    ### find finite values and those above R=1.25m (by default)
    boo = np.isfinite(ne[i]) * np.isfinite(Te[i]) * (Rraw[i] > R0)

    ### fit ne
    nep0 = (
        neR0[i], neHeight[i]/neScale, neWidth[i], 
        neGrad[i]/neScale, neBkgd[i]/neScale
    )
    try:
        nePopt, nePcov = curve_fit(
            HSV.mtanh, Rraw[i,boo], ne[i,boo]/neScale, p0=nep0, 
            sigma=neErr[i,boo]/neScale, absolute_sigma=True, bounds=neBounds
        )
        neFit = HSV.mtanh(R, *nePopt) * neScale
        neBool = True
    except RuntimeError:
        neBool = False
    except ValueError:
        neBool = False

    ### fit Te
    Tep0 = (TeR0[i], TeHeight[i], TeWidth[i], TeGrad[i], TeBkgd[i])
    try:
        TePopt, TePcov = curve_fit(
            HSV.mtanh, Rraw[i,boo], Te[i,boo], p0=Tep0, 
            sigma=TeErr[i,boo], absolute_sigma=True, bounds=TeBounds
        )
        TeFit = HSV.mtanh(R, *TePopt)
        TeBool = True
    except RuntimeError:
        TeBool = False
    except ValueError:
        TeBool = False

    ### fit pe
    pep0 = (peR0[i], peHeight[i], peWidth[i], peGrad[i], peBkgd[i])
    try:
        pePopt, pePcov = curve_fit(
            HSV.mtanh, Rraw[i,boo], pe[i,boo], p0=pep0, 
            sigma=peErr[i,boo], absolute_sigma=True, bounds=peBounds
        )
        peFit = HSV.mtanh(R, *pePopt)
        peBool = True
    except RuntimeError:
        peBool = False
    except ValueError:
        peBool = False
    
    
    ### find the maximas for the plotting
    neMax = ne[i,rind:][np.isfinite(ne[i,rind:])].max()
    TeMax = Te[i,rind:][np.isfinite(Te[i,rind:])].max()
    peMax = pe[i,rind:][np.isfinite(pe[i,rind:])].max()
    
    ###
    eq = equil(device='MASTU', shot=shotn, time=time[i])
    psiN = eq.psiN(RpsiN, 0.)[0]
    Rsep = np.interp(1., psiN, RpsiN)

    ### make the figure and axes
    fig, ax = plt.subplots(1, 3, figsize=(8,3.6), dpi=150)
    
    ### plot ne
    ax[0].errorbar(
        Rraw[i,rind:], ne[i,rind:], yerr=neErr[i,rind:], 
        c='C0', fmt='None', capsize=4
    )
    ax[0].plot(R, neProfile, '--k')
    if neBool:
        ax[0].plot(R, neFit, '--', c='C0')
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
    if TeBool:
        ax[1].plot(R, TeFit, '--', c='C1')
    ax[1].text(xlim[1]-0.01, TeMax*(yscale-0.05), 
        f'$R_0$: {TeR0[i]:.3f}m\n' + f'$h$: {TeHeight[i]:.1f}eV\n' + \
        f'$w$: {TeWidth[i]:.3f}m\n' + f'd$_RT_e$: {TeGrad[i]:.0f}eV/m\n' + 
        f'bkgd: {TeBkgd[i]:.2f}eV', ha='right', va='top', fontsize=8
    )
    
    ### plot pe
    ax[2].errorbar(
        Rraw[i,rind:], pe[i,rind:], yerr=peErr[i,rind:], 
        c='C2', fmt='None', capsize=4
    )
    ax[2].plot(R, peProfile, '--k')
    if peBool:
        ax[2].plot(R, peFit, '--', c='C2')
    ### include estimate of ne * Te for pressure
    if neBool and TeBool:
        ax[2].plot(R, neFit * TeFit * constants.e, '--', c='C5')
    ax[2].text(xlim[1]-0.01, peMax*(yscale-0.05), 
        f'$R_0$: {peR0[i]:.3f}m\n' + f'$h$: {peHeight[i]:.1f}Pa\n' + \
        f'$w$: {peWidth[i]:.3f}m\n' + f'd$_Rp_e$: {peGrad[i]:.0f}Pa/m\n' + 
        f'bkgd: {peBkgd[i]:.2f}Pa', ha='right', va='top', fontsize=8
    )
    
    ### changing axes labels and limits
    ax[0].set_xlabel('$R$ (m)')
    ax[0].set_ylabel('$n_e$ (m$^{-3}$)')
    ax[0].set_ylim([0, neMax*yscale])
    ax[0].plot([Rsep,Rsep], [0, neMax*yscale], '-', c='C7', lw=0.8, zorder=0)
    ax[1].set_xlabel('$R$ (m)')
    ax[1].set_ylabel('$T_e$ (eV)')
    ax[1].set_ylim([0, TeMax*yscale])
    ax[1].plot([Rsep,Rsep], [0, TeMax*yscale], '-', c='C7', lw=0.8, zorder=0)
    ax[2].set_xlabel('$R$ (m)')
    ax[2].set_ylabel('$p_e$ (Pa)')
    ax[2].set_ylim([0, peMax*yscale])
    ax[2].plot([Rsep,Rsep], [0, peMax*yscale], '-', c='C7', lw=0.8, zorder=0)

    ### do the fill between
    ax[0].fill_between(
        [xlim[0], R0], [0., 0.], [neMax*yscale, neMax*yscale], 
        color='C3', alpha=0.1
    )
    ax[1].fill_between(
        [xlim[0], R0], [0., 0.], [TeMax*yscale, TeMax*yscale], 
        color='C3', alpha=0.1
    )
    ax[2].fill_between(
        [xlim[0], R0], [0., 0.], [peMax*yscale, peMax*yscale], 
        color='C3', alpha=0.1
    )
    
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
    
    
    if not plot:
        plt.close()
    

    ### make the inteprolated functions
    EXC = fExcite((TeProfile, neProfile)) * scaleExcite
    REC = fRecomb((TeProfile, neProfile)) * scaleRecomb
    ION = fIonise((TeProfile, neProfile)) * scaleIonise
    EXC2 = fExcite2((TeProfile, neProfile)) * scaleExcite2
    REC2 = fRecomb2((TeProfile, neProfile)) * scaleRecomb2
    ION2 = fIonise2((TeProfile, neProfile)) * scaleIonise2


    ### make the figure
    fig, ax = plt.subplots(1, 3, figsize=(8,3.6), dpi=150)

    ### plot the rates
    ax[0].plot(R, EXC, '-', c='C0', label='cubic')
    ax[1].plot(R, REC, '-', c='C1', label='cubic')
    ax[2].plot(R, ION, '-', c='C2', label='cubic')
    ax[0].plot(R, EXC2, ':', c='k', label='linear')
    ax[1].plot(R, REC2, ':', c='k', label='linear')
    ax[2].plot(R, ION2, ':', c='k', label='linear')

    ### figure wide changes
    for j in range(0, 3):
        ax[j].set_xlim(xlim)
        ax[j].tick_params(
            axis="both", which='both', labelsize=9, direction='in', 
            left=True, bottom=True, right=True, top=True
        )
        ax[j].xaxis.set_minor_locator(AutoMinorLocator())
        ax[j].yaxis.set_minor_locator(AutoMinorLocator())
        ylim = ax[j].get_ylim()
        ax[j].set_ylim(ylim)
        ax[j].plot([Rsep,Rsep], ylim, '-', c='C7', lw=0.8, zorder=0)
        ### do the fill between
        ax[j].fill_between(
            [xlim[0], R0], [ylim[0],ylim[0]], 
            [ylim[1],ylim[1]], color='C3', alpha=0.1
        )
        ax[j].plot(xlim, [0., 0.], '-k', lw=0.8, zorder=0)
        ax[j].yaxis.get_offset_text().set_size(9)
        ax[j].set_xlabel('$R$ (m)', fontsize=9)
        ax[j].legend(
            fancybox=1, framealpha=1, fontsize=8, 
            handlelength=0, handletextpad=0, labelcolor='linecolor'
        )

    ### add ylabels
    ax[0].set_ylabel('Emission rate (ph m$^3$ s$^{-1}$)', fontsize=9)
    ax[1].set_ylabel('Emission rate (ph m$^3$ s$^{-1}$)', fontsize=9)
    ax[2].set_ylabel('Ionisation rate (m$^3$ s$^{-1}$)', fontsize=9)

    ### add titles for subplots
    ax[0].set_title('PEC$^\\mathrm{exc}_{2 \\rightarrow 1}$', fontsize=9)
    ax[1].set_title('PEC$^\\mathrm{rec}_{2 \\rightarrow 1}$', fontsize=9)
    ax[2].set_title('SCD', fontsize=9)

    ### add title to plot
    fig.suptitle(f'shot # {shotn}, t={time[i]:.6f}s')

    ### tight and save
    plt.tight_layout()
    plt.savefig(savePath2 + f't_{time[i]:.4f}s.png')


    if not plot:
        plt.close()


if plot:
    plt.show()


print('done :)')
