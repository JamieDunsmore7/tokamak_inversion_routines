from functions import findNearest
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np


### function for plotting inversion with reconstruction and raw-data
def plotInversion(
    R, data, err, RgridB, y, yErr, backprojection, time, Rmax, yMax, R0, R1, 
    fwhm, fwhmErr, savePath=None
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
        ax.plot([R0[i], R1[i]], [yMax[i]/2., yMax[i]/2.], '-k', lw=0.8)
        ax.text(
            Rmax[i]*0.99, yMax[i], f'$R_\\mathrm{{max}}$\n{Rmax[i]:.3f}m', 
            ha='right', va='center', fontsize=8
        )
        ax.text(
            R1[i]*1.01, yMax[i]/2., 
            f'FWHM\n{fwhm[i]*100:.2f}$\\pm$\n{fwhmErr[i]*100:.2f}cm', 
            ha='left', va='center', fontsize=8
        )
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
    return


### function for plotting emissivity, Siz, and n0
def plotResults(
    R, emiss, emissErr, emMaxR, emMax, emR0, emR1, emFWHM, emFWHMerr, Siz,
    SizErr, SizMax, SizMaxR, SizR0, SizR1, SizFWHM, SizFWHMerr, n0, n0Err,
    n0Max, n0MaxR, n0R1, n0R2, n0R3, n0w1, n0w1err, n0w2, n0w2err, n0w3, 
    n0w3err, n0R1A, n0R2A, n0R3A, time, figN0=None, neutralRatio=None, 
    psiN=None, yMult=1.1, xlim=[1.25,1.50], n0lim=1e14, savePath=None
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
        ax[0].plot(
            [emR0[i], emR1[i]], [emMax[i]/2., emMax[i]/2.], '-k', lw=0.8
        )
        ax[0].text(
            emMaxR[i]*0.99, emMax[i], 
            f'$R_\\mathrm{{max}}$\n{emMaxR[i]:.3f}m',
            ha='right', va='center', fontsize=8
        )
        ax[0].text(
            (emR0[i]+emR1[i])/2., 0.98*emMax[i]/2., 
            f'FWHM\n{emFWHM[i]*100:.2f}$\\pm$\n{emFWHMerr[i]*100:.2f}cm', 
            ha='center', va='top', fontsize=8
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
        ax[1].plot(
            [SizR0[i], SizR1[i]], [SizMax[i]/2., SizMax[i]/2.], '-k', lw=0.8
        )
        ax[1].text(
            SizMaxR[i]*0.99, SizMax[i], 
            f'$R_\\mathrm{{max}}$\n{SizMaxR[i]:.3f}m', 
            ha='right', va='center', fontsize=8
        )
        ax[1].text(
            (SizR0[i]+SizR1[i])/2., 0.98*SizMax[i]/2., 
            f'FWHM\n{SizFWHM[i]*100:.2f}$\\pm$\n{SizFWHMerr[i]*100:.2f}cm', 
            ha='center', va='top', fontsize=8
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
        ylim = [
            np.max((n0[i,:Rind].min() / yMult, n0lim)), 
            (n0[i] + n0Err[i]).max() * yMult
        ]
        ax[2].plot(xlim, [n0Max[i],n0Max[i]], '-', c='C0', lw=0.8)
        ax[2].plot(
            [n0MaxR[i],n0MaxR[i]], [ylim[0],n0Max[i]], '-', c='C0', lw=0.8
        )
        ax[2].text(
            n0MaxR[i], n0Max[i], 
            f'{n0MaxR[i]:.3f}m\n{n0Max[i]:.2g}m$^{{-3}}$', 
            c='C0', fontsize=8, ha='right', va='top'
        )
        yVal = n0Max[i] * np.exp(-1.)
        ax[2].plot([xlim[0],n0R1[i]], [yVal,yVal], '-', c='C1', lw=0.8)
        ax[2].plot([n0R1[i],n0R1[i]], [ylim[0],yVal], '-', c='C1', lw=0.8)
        ax[2].text(
            n0R1[i], yVal, 
            f'{n0w1[i]*100.:.1f}$\\pm${n0w1err[i]*100.:.1f}cm', 
            c='C1', fontsize=8, ha='right', va='top'
        )
        ax[2].plot([n0R1A[i],xlim[1]], [yVal,yVal], '--', c='C1', lw=0.8)
        ax[2].plot([n0R1A[i],n0R1A[i]], [ylim[0],yVal], '--', c='C1', lw=0.8)
        yVal = n0Max[i] * np.exp(-2.)
        ax[2].plot([xlim[0],n0R2[i]], [yVal,yVal], '-', c='C3', lw=0.8)
        ax[2].plot([n0R2[i],n0R2[i]], [ylim[0],yVal], '-', c='C3', lw=0.8)
        ax[2].text(
            n0R2[i], yVal, 
            f'{n0w2[i]*100.:.1f}$\\pm${n0w2err[i]*100.:.1f}cm', 
            c='C3', fontsize=8, ha='right', va='top'
        )
        ax[2].plot([n0R2A[i],xlim[1]], [yVal,yVal], '--', c='C3', lw=0.8)
        ax[2].plot([n0R2A[i],n0R2A[i]], [ylim[0],yVal], '--', c='C3', lw=0.8)
        yVal = n0Max[i] * np.exp(-3.)
        ax[2].plot([xlim[0],n0R3[i]], [yVal,yVal], '-', c='C4', lw=0.8)
        ax[2].plot([n0R3[i],n0R3[i]], [ylim[0],yVal], '-', c='C4', lw=0.8)
        ax[2].text(
            n0R3[i], yVal, 
            f'{n0w3[i]*100.:.1f}$\\pm${n0w3err[i]*100.:.1f}cm', 
            c='C4', fontsize=8, ha='right', va='top'
        )
        ax[2].plot([n0R3A[i],xlim[1]], [yVal,yVal], '--', c='C4', lw=0.8)
        ax[2].plot([n0R3A[i],n0R3A[i]], [ylim[0],yVal], '--', c='C4', lw=0.8)
        ax[2].set_title('Neutral density', fontsize=9)
        ax[2].set_xlabel('$R$ (m)')
        ax[2].set_ylabel('$n_0$ (m$^{-3}$)')
        ax[2].yaxis.get_offset_text().set_size(9)
        ax[2].set_ylim(ylim)
        ax[2].set_yscale('log')
        if neutralRatio is not None:
            ax[2].text(
                n0MaxR[i], n0Max[i], 
                f'$n_{{0,\\mathrm{{fig}}}}$=\n{figN0[i]:.2g}m$^{{-3}}$\n' + \
                f'$(n_0/n_e)_\\mathrm{{sep}}$=\n{neutralRatio[i]:.4f}', 
                color='k', fontsize=8, va='top', ha='left'
            )
        elif figN0 is not None:
            ax[2].text(
                n0MaxR[i], n0Max[i], 
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
    emMax = results['emMax']
    emMaxR = results['emMaxR']
    emR0 = results['emR0']
    emR1 = results['emR1']
    emFWHM = results['emFWHM']
    emFWHMerr = results['emFWHMerr']
    Rind = results['Rind']
    Rprofile = results['Rprofile']
    # profileTemp = results['profileTemp']
    # profileDensity = results['profileDensity']
    ioniseRate = results['ioniseRate']
    ioniseErr = results['ioniseErr']
    neutralDensity = results['neutralDensity']
    neutralErr = results['neutralErr']
    ioniseMax = results['ioniseMax']
    ioniseMaxR = results['ioniseMaxR']
    ioniseR0 = results['ioniseR0']
    ioniseR1 = results['ioniseR1']
    ioniseFWHM = results['ioniseFWHM']
    ioniseFWHMerr = results['ioniseFWHMerr']
    neutralMax = results['neutralMax']
    neutralMaxR = results['neutralMaxR']
    neutralR1 = results['neutralR1']
    neutralR2 = results['neutralR2']
    neutralR3 = results['neutralR3']
    neutralWidth1 = results['neutralWidth1']
    neutralWidth1err = results['neutralWidth1err']
    neutralWidth2 = results['neutralWidth2']
    neutralWidth2err = results['neutralWidth2err']
    neutralWidth3 = results['neutralWidth3']
    neutralWidth3err = results['neutralWidth3err']
    neutralR1A = results['neutralR1A']
    neutralR2A = results['neutralR2A']
    neutralR3A = results['neutralR3A']
    # neutralWidth1A = results['neutralWidth1A']
    # neutralWidth1Aerr = results['neutralWidth1Aerr']
    # neutralWidth2A = results['neutralWidth2A']
    # neutralWidth2Aerr = results['neutralWidth2Aerr']
    # neutralWidth3A = results['neutralWidth3A']
    # neutralWidth3Aerr = results['neutralWidth3Aerr']
    figN0 = results['figN0']
    psiN = results['psiN']
    neutralRatio = results['neutralRatio']


    ### call the plotting functions
    plotInversion(
        R, data[I:J], err[I:J], RgridB, emissivity[I:J], emissivityErr[I:J], 
        backprojection[I:J], time[I:J], emMaxR[I:J], emMax[I:J], emR0[I:J], 
        emR1[I:J], emFWHM[I:J], emFWHMerr[I:J], savePath=savePathInversion, 
    )
    plotResults(
        Rprofile, emissivity[I:J,Rind:], emissivityErr[I:J,Rind:], 
        emMaxR[I:J], emMax[I:J], emR0[I:J], emR1[I:J], emFWHM[I:J], 
        emFWHMerr[I:J], ioniseRate[I:J], ioniseErr[I:J], ioniseMax[I:J], 
        ioniseMaxR[I:J], ioniseR0[I:J], ioniseR1[I:J], ioniseFWHM[I:J], 
        ioniseFWHMerr[I:J], neutralDensity[I:J], neutralErr[I:J], 
        neutralMax[I:J], neutralMaxR[I:J], neutralR1[I:J], neutralR2[I:J], 
        neutralR3[I:J], neutralWidth1[I:J], neutralWidth1err[I:J], 
        neutralWidth2[I:J], neutralWidth2err[I:J], neutralWidth3[I:J], 
        neutralWidth3err[I:J], neutralR1A[I:J], neutralR2A[I:J], 
        neutralR3A[I:J], time[I:J], figN0=figN0[I:J], 
        neutralRatio=neutralRatio[I:J], psiN=psiN[I:J], 
        savePath=savePathResults, 
    )
    if show:
        plt.show()
    