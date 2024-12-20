from functions import findNearest
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np


### function for plotting inversion with reconstruction and raw-data
def plotInversion(
    R, data, err, RgridB, y, yErr, backprojection, time, savePath=None
    ):
    for i in range(data.shape[0]):
        fig, ax = plt.subplots(1, 1, figsize=(3.5,3), dpi=150)
        ax.plot(R, data[i], '-', c='k', zorder=3, label='raw RBA')
        ax.fill_between(
            R, data[i]-err[i], data[i]+err[i], color='k', alpha=0.2
        )
        ax.plot(RgridB, y[i], '-', c='C0', zorder=2, label='inversion')
        ax.fill_between(
            RgridB, y[i]-yErr[i], y[i]+yErr[i], color='C0', alpha=0.2, zorder=2
        )
        ax.plot(R, backprojection[i,:], c='C2', zorder=4, label='reconst. RBA')
        xplot = [0.2,1.875]
        ax.set_xlim(xplot)
        ax.plot(xplot, [0.,0.], '-k', lw=0.8, zorder=1)
        ax.set_xlabel('R (m)', fontsize=9)
        ax.set_ylabel('Units', fontsize=9)
        ax.yaxis.get_offset_text().set_size(9)
        ax.legend(fancybox=1, framealpha=1, fontsize=8)
        ax.tick_params(axis="both", which='both', labelsize=9, direction='in', 
                    left=True, bottom=True, right=True, top=True)
        ax.set_title(f'i={i:.0f}, t={time[i]:.4f}s', fontsize=10)
        plt.tight_layout()
        if savePath:
            plt.savefig(savePath + f'{time[i]:.4f}s.png')
    # plt.show()
    return


### function for plotting emissivity, Siz, and n0
def plotResults(
    R, emiss, emissErr, Siz, SizErr, n0, n0Err, time, figN0=None, 
    neutralRatio=None, psiN=None, yMult=1.1, xlim=[1.25,1.50], savePath=None
    ):
    Rind = findNearest(R, xlim[1])
    if psiN is None:
        top = True
    else:
        top = False
    for i in range(len(time)):
        fig, ax = plt.subplots(1, 3, figsize=(6,3), dpi=150)
        
        ax[0].plot(R, emiss[i], '-', c='C0')
        ax[0].fill_between(
            R, emiss[i]-emissErr[i], 
            emiss[i]+emissErr[i], color='C0', alpha=0.3
        )
        ax[0].set_title('Emissivity', fontsize=9)
        ax[0].set_xlabel('$R$ (m)')
        ax[0].set_ylabel('$\\epsilon$ (m$^{-3}$ s$^{-1}$)')
        ax[0].yaxis.get_offset_text().set_size(9)
        ax[0].set_ylim([0., (emiss[i]+emissErr[i]).max() * yMult])
        ax[0].yaxis.set_minor_locator(AutoMinorLocator())
        
        ax[1].plot(R, Siz[i], '-', c='C1')
        ax[1].fill_between(
            R, Siz[i]-SizErr[i], Siz[i]+SizErr[i], color='C1', alpha=0.3
        )
        ax[1].set_title('Ionisation rate', fontsize=9)
        ax[1].set_xlabel('$R$ (m)')
        ax[1].set_ylabel('$S_\\mathrm{iz}$ (m$^{-3}$ s$^{-1}$)')
        ax[1].yaxis.get_offset_text().set_size(9)
        ax[1].set_ylim([0., (Siz[i]+SizErr[i]).max() * yMult])
        ax[1].yaxis.set_minor_locator(AutoMinorLocator())
        
        ax[2].plot(R, n0[i], '-', c='C2', mfc='None')
        ax[2].fill_between(
            R, n0[i]-n0Err[i], n0[i]+n0Err[i], color='C2', alpha=0.3
        )
        ax[2].set_title('Neutral density', fontsize=9)
        ax[2].set_xlabel('$R$ (m)')
        ax[2].set_ylabel('$n_0$ (m$^{-3}$)')
        ax[2].yaxis.get_offset_text().set_size(9)
        ax[2].set_ylim([
            n0[i,:Rind].min() / yMult, (n0[i] + n0Err[i]).max() * yMult
        ])
        ax[2].set_yscale('log')
        if neutralRatio:
            ax[2].text(
                xlim[0], n0[i].max(), 
                f'$n_{{0,\\mathrm{{fig}}}}$=\n{figN0[i]:.2g}m$^{{-3}}$\n' + \
                f'$(n_0/n_e)_\\mathrm{{sep}}$=\n{neutralRatio[i]:.4f}', 
                color='k', fontsize=8, va='top', ha='left'
            )
        elif figN0:
            ax[2].text(
                xlim[0], n0[i].max(), 
                f'$n_{{0,\\mathrm{{fig}}}}$=\n{figN0[i]:.2g}m$^{{-3}}$\n', 
                color='k', fontsize=8, va='top', ha='left'
            )
            
        for j in range(3):
            ax[j].set_xlim(xlim)
            ax[j].tick_params(
                axis="both", which='both', labelsize=9, direction='in', 
                left=True, bottom=True, right=True, top=top
            )
            ax[j].xaxis.set_minor_locator(AutoMinorLocator())
            if not top:
                def forward(x):
                    return np.interp(x, R, psiN[i])
                def inverse(x):
                    return np.interp(x, psiN[i], R)
                sepR = np.interp(1., psiN[i], R)
                
                secax = ax[j].secondary_xaxis(
                    'top', functions=(forward, inverse)
                )
                secax.xaxis.set_minor_locator(AutoMinorLocator())
                secax.set_xlabel('$\\Psi_\\mathrm{N}$')
                secax.tick_params(
                    axis="both", which='both', labelsize=9, direction='in', 
                            left=False, bottom=False, right=False, top=True
                )
                secax.minorticks_on()
                ylim = ax[j].get_ylim()
                ax[j].plot([sepR, sepR], ylim, '-k', lw=0.8, zorder=0)
        
        fig.suptitle(f'i={i:.0f}, t={time[i]:.4f}s', fontsize=10)
        plt.tight_layout()
        if savePath:
            plt.savefig(savePath + f'{time[i]:.4f}s.png')
    # plt.show()
    return


###############################################################################
###                          if the script is run                           ###
###############################################################################


if __name__ == "__main__":
    
    ### define where I will save the figures
    savePath0 = '/home/sthoma/Documents/Plots/rba/'
    
    
    ### do the stuff for the loading and plotting of the results
    import sys
    resultsFile = sys.argv[1]
    try:
        show = bool(sys.argv[2])
    except IndexError:
        show = False
    try:
        I = int(sys.argv[3])
    except IndexError:
        I = 0
    try:
        J = int(sys.argv[4])
        if J == 0:
            J = None
    except IndexError:
        J = None
    
    
    
    ### make folders to save plots
    import os
    savePath = \
        f"{savePath0}/{resultsFile.split('/')[-2]}/" + \
        f"{resultsFile.split('/')[-1][:-4]}/"
    savePathInversion = savePath + "inversion/"
    if not os.path.exists(savePathInversion):
        os.makedirs(savePathInversion)
    savePathResults = savePath + "emissIoniseNeutral/"
    if not os.path.exists(savePathResults):
        os.makedirs(savePathResults)
        
    
    ### load the results
    results = np.load(resultsFile)
    R = results['R']
    data = results['data']
    err = results['err']
    time = results['time']
    # Rgrid = results['Rgrid']
    RgridB = results['RgridB']
    emissivity = results['emissivity']
    emissivityErr = results['emissivityErr']
    backprojection = results['backprojection']
    # scale = results['scale']
    Rind = results['Rind']
    Rprofile = results['Rprofile']
    # profileTemp = results['profileTemp']
    # profileDensity = results['profileDensity']
    ioniseRate = results['ioniseRate']
    ioniseErr = results['ioniseErr']
    neutralDensity = results['neutralDensity']
    neutralErr = results['neutralErr']
    figN0 = results['figN0']
    psiN = results['psiN']
    neutralRatio = results['neutralRatio']
    
    
    ### call the plotting functions
    plotInversion(
        R, data[I:J], err[I:J], RgridB, emissivity[I:J], emissivityErr[I:J], 
        backprojection[I:J], time[I:J], savePath=savePathInversion, 
    )
    plotResults(
        Rprofile, emissivity[I:J,Rind:], emissivityErr[I:J,Rind:], 
        ioniseRate[I:J], ioniseErr[I:J], neutralDensity[I:J], neutralErr[I:J], 
        time[I:J], figN0=figN0[I:J], neutralRatio=neutralRatio[I:J], 
        psiN=psiN[I:J], savePath=savePathResults, 
    )
    