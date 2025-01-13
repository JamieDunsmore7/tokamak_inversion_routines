### imports
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import os
import pickle
import sys

try:
    from pyEquilibrium.equilibrium import equilibrium as eq
    equilBool = True
except ModuleNotFoundError:
    try:
        sys.path.insert(0, '/home/sthoma/tools/')
        from pyEquilibrium.equilibrium import equilibrium as eq
    except ModuleNotFoundError:
        print('couldnt load equilibrium, visit')
        print('https://git.ccfe.ac.uk/SOL_Transport/pyEquilibrium/-/tree/lkogan_aeqdsk')
        equilBool = False

### loading modules above this script
parentDirectory = os.path.abspath('..')
sys.path.insert(0, parentDirectory)
import BES
import myFuncs as mf

####################################################
### change variables here for running the script ###
####################################################

# shotMag = 48115
shotMag = 50747
timeMag = 0.5
plotSW = False
# mirror = np.array([-0.736401, -1.709989, -0.221043])
mirror = np.array([ -948.631, -2227.801,     0.   ]) / 1000. # MSE
title = None
savelist = None
# savelist = ['/home/sthoma/Documents/Plots/magpi.png',
#             '/home/sthoma/Documents/Plots/magpi.eps',
#             '/home/sthoma/Documents/Plots/magpi.svg',
#             '/home/sthoma/Documents/Plots/magpi.pdf']
# plotType = 'fanShot'
# plotArgs = [48759, 48643, 48640]
# plotType = 'fanShot'
# plotArgs = [47074, 48074]
# plotType = 'fanR'
# plotArgs = [1.16, 1.25, 1.34]
# plotType = 'fanLim'
# plotArgs = [(1.10,1.22), (1.19,1.31), (1.28,1.40)]
plotType = 'LoS'
plotArgs = [1.377224207]
# plotArgs = [
    # 0.7363783717,
    # 0.720952034,
    # 0.7098255754,
    # 0.7028372884,
    # 0.6997739673,
    # 0.7060120106,
    # 0.7138326764,
    # 0.7242915034,
    # 0.7370667458,
    # 0.7518509626,
    # 0.7836748362,
#     0.8026967645,
#     0.8227551579,
#     0.8436597586,
#     0.865244031,
#     0.9137676954,
#     0.936640501,
#     0.959708035,
#     0.982894063,
#     1.006132603,
#     1.051163316,
#     1.074264884,
#     1.09724474,
#     1.120074272,
#     1.142729044,
#     1.186555743,
#     1.208590627,
#     1.230390787,
#     1.251947165,
#     1.273253202,
#     1.316490054,
#     1.336999655,
#     1.357244492,
#     1.377224207,
#     1.396938682,
#     1.438137054,
#     1.457025528,
#     1.475654006,
#     1.49402523,
#     1.512141347,
# ]

# plotArgs = [1.2]
# plotArgs = [1.16, 1.25, 1.34]

"""
Should only need to change the variables in this first section
shotMag - int: the shot number. Used for magnetic geometry.
timeMag - float, or len2 list of floats: Find magnetic geometry at
    the given time.
    Value should be in seconds.
plotSWQ - boolean: if True, will add a line where the SW beam line is
title - None or string: if None, nothing will happen, otherwise
    it will add the string to the top of the plot for a title.
savelist - None, or list of strings: if None, it won't be saved,
    otherwise it iterates over the list and saves to those filenames.
plotType - string: should be one of ['fanShot', 'fanR', 'fanLim', 'LoS']
    ### fanShot - plot fans of the view given the shot number(s), BES.
        requires a list of ints.
    ### fanR - plots fans of the view given a central R location(s), BES.
        requires a list of floats.
    ### fanLim - plots fans of the view between given pairs of R locations.
        requires a list of len2 tuples.
    ### LoS - plots individual lines-of-site given the R location(s).
        requires a list of floats.
Examples of the plotType and plotArgs are given above
"""

############ plotting argument you might want to customise ############

### size of the figure
figsize = (4.5,5)
### dpi of the figure
dpi = 150
### colour of magnetics to be plotted. C3 is red, and matches the
### separatrix colour of the Rz plot from divertor_geometry
cMag = 'C3'
### colour of machine wall and centre column
cWall = 'k'
### colour of the limiter flux surface. C7 is grey
cLim = 'C7'
### colour of the entrance pupil. C4 is purple
cMirror = 'C4'
### colourscheme for the SS beam colourbar
SScmap = plt.cm.Purples
### colours and line types used in plotting
colours = ['C0', 'C1', 'C2', 'C5', 'C6', 'C8', 'C9']
lineType = ['-', '--', ':']
Nc = len(colours) # useful later
### ticks on the SS colourbar
SSticks = np.arange(0.7, 2.09, 0.1)
### shrink for the colorbar to make it fit nicer
shrink = 0.6
### alpha for the fill between (fan only)
alpha = 0.2
### limits of plot
xlim = [-2.2, 2.2]
# ylim = [-2.2, 3.0]
ylim = [-2.2, 2.2]
### for the legend
legend = False
loc = 'upper center'
### dpi to save it as. 300dpi recommended for printing
savedpi = 300

################## probably don't need to change below ##################

### radius of vessel wall, SS beam port location, metres
radiusWall = 2.033
### radius of centre column, metres
radiusCC = 0.260841
### information for xspaceSS, the xrange the SS beam acts over
x0 = 0.543 # metres
x1 = 0.862 # metres
xn = 1001 # number points for xplot
xlen = int(((x1 - x0) * 1e3) + 2)

###### a couple of functions ######

### get the line-of-site for a given radius
def getLoSCoords(Rwant):
    ind = mf.findNearest(Rbeam, Rwant)
    Ppoint = np.array([xspaceSS[boo][ind], yspaceSS[boo][ind], sourceSS[-1]])
    b1 = Ppoint - mirror
    mLoS = b1[1] / b1[0]
    CLoS = mirror[1] - (mirror[0] * b1[1] / b1[0])
    xMax = max(mf.solveQuad(1 + mLoS**2, 2. * mLoS * CLoS, CLoS**2 - radiusWall**2))
    yMax = xMax * mLoS + CLoS
    return xMax, yMax, Ppoint

### get the LoS for the edges of a fan for given view radius
def getLoSFans(plotType, plotArg):
    if plotType == 'fanShot':
        Rwant = np.array([BES.getRz(plotArg)[0].min(),
                          BES.getRz(plotArg)[0].max()])
        label = '# {}'.format(int(plotArg))
    elif plotType == 'fanR':
        Rwant = np.array([BES.getRzCentre(plotArg)[0].min(),
                          BES.getRzCentre(plotArg)[0].max()])
        label = 'R$_\\mathrm{{mid}}$={:.2f}m'.format(plotArg)
    else:
        Rwant = np.array([plotArg[0], plotArg[1]])
        label = '{:.2f}<R<{:.2f}m'.format(plotArg[0],plotArg[1])
    ind0 = mf.findNearest(Rbeam, Rwant[0])
    Ppoint = np.array([xspaceSS[boo][ind0],
                yspaceSS[boo][ind0], sourceSS[-1]])
    b1 = Ppoint - mirror
    mLoS0 = b1[1] / b1[0]
    CLoS0 = mirror[1] - (mirror[0] * b1[1] / b1[0])
    xMax0 = max(mf.solveQuad(1 + mLoS0**2, 2. * mLoS0 * CLoS0,
                      CLoS0**2 - radiusWall**2))
    yMax0 = xMax0 * mLoS0 + CLoS0
    ind1 = mf.findNearest(Rbeam, Rwant[1])
    Ppoint = np.array([xspaceSS[boo][ind1],
                yspaceSS[boo][ind1], sourceSS[-1]])
    a1 = mirror + 0.
    b1 = Ppoint - mirror
    mLoS1 = b1[1] / b1[0]
    CLoS1 = a1[1] - (a1[0] * b1[1] / b1[0])
    xMax1 = max(mf.solveQuad(1 + mLoS1**2, 2. * mLoS1 * CLoS1,
                      CLoS1**2 - radiusWall**2))
    yMax1 = xMax1 * mLoS1 + CLoS1
    yMax2 = xMax1 * mLoS0 + CLoS0
    return xMax0, yMax0, xMax1, yMax1, yMax2, label

############################## the script ##############################

### find out the radii to add to the plot
rhoLim = mf.getRhoLimiter(shotMag, timeMag)
radiusLim = mf.rho2R(rhoLim, shotMag, timeMag)
radiusAxis = mf.getMagneticAxis(shotMag,
            [timeMag-2.5e-3,timeMag+2.5e-3])[0]
radiusSepOut, radiusSepIn = mf.getRadiusIO(shotMag, timeMag)

### find the x and y coordinates of the bits above
xplot = np.linspace(-radiusWall, radiusWall, xn)
yplot = np.sqrt(radiusWall**2 - xplot**2)
xCC = np.linspace(-radiusCC, radiusCC, xn)
yCC = np.sqrt(radiusCC**2 - xCC**2)
xSepOut = np.linspace(-radiusSepOut, radiusSepOut, xn)
ySepOut = np.sqrt(radiusSepOut**2 - xSepOut**2)
xLim = np.linspace(-radiusLim, radiusLim, xn)
yLim = np.sqrt(radiusLim**2 - xLim**2)
xSepIn = np.linspace(-radiusSepIn, radiusSepIn, xn)
ySepIn = np.sqrt(radiusSepIn**2 - xSepIn**2)
xAxis = np.linspace(-radiusAxis, radiusAxis, xn)
yAxis = np.sqrt(radiusAxis**2 - xAxis**2)

### load beam data
with open('/home/sthoma/bes/data/beam_geometry_data.pkl', 'rb') as handle:
    data =  pickle.load(handle)
### SS beam data
ss = data['ss_beam']
beamSS = ss['axis']
sourceSS = ss['src'] / 100.
### gradient of line describing SS
mSS = beamSS[1] / beamSS[0]
### intercept of line describing SS
CSS = sourceSS[1] - (sourceSS[0] *
      beamSS[1] / beamSS[0])
### y co-ordinates of SS beam
ySS = mSS * xplot + CSS

### SW beam data
sw = data['sw_beam']
beamSW = sw['axis']
sourceSW = sw['src'] / 100.
### gradient of line describing SW
mSW = beamSW[1] / beamSW[0]
### intercept of line describing SW
CSW = sourceSW[1] - (sourceSW[0] *
      beamSW[1] / beamSW[0])
### y co-ordinates of SW beam
ySW = mSW * xplot + CSW

### space SS beam acts over
xspaceSS = np.linspace(x0, x1, xlen)
yspaceSS = mSS * xspaceSS + CSS
xstep = np.diff(xspaceSS).mean()
### radius of the SS beam
RspaceSS = np.sqrt(xspaceSS**2 + yspaceSS**2)
### lower half of beam, before tangency point
boo = yspaceSS < 0.
Rbeam = np.sqrt(xspaceSS[boo]**2 + yspaceSS[boo]**2)

############################## making the plot ##############################

### create figure and axis
fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
### add magnetic and machine wall circles
ax.plot(np.append(xAxis, xAxis[::-1]), np.append(yAxis, -yAxis),
        '--', c=cMag, label='$B_\\mathrm{{axis}}$', zorder=2)
ax.plot(xplot, yplot, '-', c=cWall, zorder=5)
ax.plot(xplot, -yplot, '-', c=cWall, zorder=5)
ax.plot(xCC, yCC, '-', c=cWall, zorder=5)
ax.plot(xCC, -yCC, '-', c=cWall, zorder=5)
ax.plot(np.append(xSepOut, xSepOut[::-1]), np.append(ySepOut, -ySepOut),
        '-', c=cMag, label='Separatrix', zorder=2)
ax.plot(np.append(xSepIn, xSepIn[::-1]), np.append(ySepIn, -ySepIn),
        '-', c=cMag, zorder=2)
ax.plot(np.append(xLim, xLim[::-1]), np.append(yLim, -yLim),
        '--', c=cLim, label='$\\Psi_{N,\\mathrm{Limiter}}$', zorder=2)
ax.plot(mirror[0], mirror[1], 'X', c=cMirror,
        mec='k', label='Mirror', zorder=6)

### adding in viewing geometry to the plots
if plotType == 'LoS':
    for i in range(0, len(plotArgs)):
        print(i)
        xMax, yMax, pPoint = getLoSCoords(plotArgs[i])
        label = 'R={:.3f}m'.format(plotArgs[i])
        colour = colours[i%Nc]
        # ax.plot([mirror[0], xMax], [mirror[1], yMax], lineType[i//Nc],
        #         color=colour, label=label, zorder=4)
        ax.plot([mirror[0], xMax], [mirror[1], yMax], '-k', lw=0.8, zorder=4)
        ax.text(
            pPoint[0], pPoint[1], f'R={plotArgs[i]:.4f}m', 
            fontsize=8, ha='left', va='top'
        )
        
        
elif plotType in ['fanShot', 'fanR', 'fanLim']:
    for i in range(0, len(plotArgs)):
        xMax0, yMax0, xMax1, yMax1, yMax2, \
                    label = getLoSFans(plotType, plotArgs[i])
        colour = colours[i%Nc]
        ax.plot([mirror[0], xMax0], [mirror[1], yMax0],
                lineType[i//Nc], color=colour, zorder=4)
        ax.plot([mirror[0], xMax1], [mirror[1], yMax1],
                lineType[i//Nc], color=colour, zorder=4)
        ax.fill_between([mirror[0], xMax1, xMax0], [mirror[1], yMax2, yMax0],
                [mirror[1], yMax1, yMax0], color=colour,
                alpha=alpha, label=label, zorder=3)
else:
    i = -1

### adding beams
sc=ax.scatter(xspaceSS, yspaceSS, c=RspaceSS, marker='o',
              vmin=0.7, vmax=RspaceSS.max(),
              cmap=plt.cm.Purples, label='SS beam', zorder=2)
fig.colorbar(sc, ax=ax, label='South-south major radius (m)', ticks=SSticks, shrink=shrink)
if plotSW:
    booSW = np.sqrt(xplot**2 + ySW**2) <= radiusWall
    colour = colours[(i+1)%Nc]
    ax.plot(xplot[booSW], ySW[booSW], lineType[(i+1)//Nc],
            color=colour, label='SW beam', zorder=2)

### rest of plotting
# ax.grid(zorder=1)
if legend:
    ax.legend(fancybox=True, framealpha=1, loc=loc, ncol=2)
ax.tick_params(
    axis="both", which='both', labelsize=9, direction='in', 
    left=True, bottom=True, right=True, top=True
            )
ax.set_yticks([-2,-1,0,1,2])
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_aspect('equal')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
if title is not None:
    ax.set_title(title)
plt.tight_layout()
if savelist is not None:
    for savestring in savelist:
        plt.savefig(savestring, dpi=savedpi)

plt.show()

#
