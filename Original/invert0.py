#!/usr/bin/env python3

import functions as fn
from mastu import HSV
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh, solve_banded
import sys


##################################
### loading variables and data ###
##################################


device = 'rba'
### load the dictionary
dictFile = sys.argv[1]
inputDict = fn.loadDict(dictFile)
### unzip the dictionary
shotn = inputDict['shotn']
Rzfile = inputDict['Rzfile']
I0 = inputDict['I0']
I1 = inputDict['I1']
J0 = inputDict['J0']
J1 = inputDict['J1']
tind = inputDict['tind']
dtind = inputDict['dtind']
tend = inputDict['tend']

nrMult = inputDict['nrMult']
rEnd = inputDict['rEnd']
sysErr = inputDict['sysErr']
nFisher = inputDict['nFisher']
regGuess = inputDict['regGuess']
regMin = inputDict['regMin']
biasedEdges = bool(inputDict['biasedEdges'])


#######################################
### setting up coordinates and grid ###
#######################################


### load the Rz coordinates
R0, z0 = HSV.getRz(Rzfile, I0, I1, J0, J1)

### prepping R, adding extra, and flipping
R, goodChans, flipBool = fn.prepR(R0, rEnd)
### grid to interpolate onto
nGrid = int(len(R0) * nrMult)
Rgrid = fn.makeGrid(R, nGrid)
### making dL
dL = fn.makeDL(Rgrid, R)


#######################################
### loading data and pre-processing ###
#######################################


### load the data from UDA
rawData, time = HSV.get(shotn)

### TODO: currently only good for testing 1 frame
"""dSlicePrep = (slice(I0,I1), slice(J0,J1), slice(0,len(time)))
data = HSV.prepData(rawData, dSlicePrep, goodChans, flipBool, rEnd)
# nT, nR = data.shape

### removing the "offset" from the end, and creating an error
dSliceBkgd = (slice(I0,I1), slice(J0,J1), slice(tend,len(time)))
background = HSV.prepBackground(rawData, dSliceBkgd, goodChans, flipBool, rEnd)
print('this is the data')
print(data)
print(data.shape)
print('this is the background')
print(background)
print(background.shape)

data, err = HSV.offset(data, background, nR, sysErr=sysErr)"""
dSlice = (slice(I0,I1), slice(J0,J1), slice(0,len(time)))
data, err, nT, nR = HSV.prepData0(rawData, dSlice, goodChans, flipBool, rEnd, tend, sysErr=sysErr)
print(data, data.shape)
print(err, err.shape)

### scale so not working with large numbers
scale = fn.makeScale(data)


########################################
### doing the brunt of the work here ###
########################################


### make the regularisation band matrix
D = fn.regulMatrix(nGrid, biasedEdges=biasedEdges)
### making empty arrays for results
y = np.zeros((nT, nGrid-1))
yErr = np.zeros((nT, nGrid-1))
chi2 = np.zeros(nT)
gamma = np.zeros(nT)
backprojection = np.zeros((nT, nR))

### iterate over the times
for i in range(nT):
    errZinds = np.where(np.isclose(err[i], 0.)) 
    T = dL / err[i][:,None] * scale
    mean_d = data[i] / err[i]
    d = data[i] / err[i]
    
    ### replace infs and NaNs
    T[errZinds] = 0.
    mean_d[errZinds] = 0.
    d[errZinds] = 0.
    
    ### empty array
    W = np.ones(nGrid-1)
    ### arrays of indices
    indLos = slice(0, nR)
    indSpace = slice(0, nGrid-1)
    ### linspace
    Q = np.linspace(0., 1., indLos.stop-indLos.start)
    
    ### iterate over the nFisher asked for
    for f in range(nFisher):
        
        ### multiply tridiagonal regularisation operator 
        ### by a diagonal weight matrix W
        WD = np.copy(D)
        ### seems redundent, as W is all ones
        WD[0,1:]*=W[:-1]
        WD[1]*=W
        WD[2,:-1]*=W[1:]
        
        ### transpose the band matrix
        DTW = np.copy(WD)
        DTW[0,1:], DTW[2,:-1] = WD[2,:-1], WD[0,1:]
    
        ### solve Tikhonov regularization (optimised for speed)
        H = solve_banded(
            (1, 1), DTW, T[indLos, indSpace].T, 
            overwrite_ab=True, check_finite=False
        )
        ### fast method to calculate U, S, V = svd(H.T) of rectangular matrix
        LL = np.dot(H.T, H)
        ### also from scipy
        S2, U = eigh(LL, overwrite_a=True, check_finite=False, lower=True)
        ### singular values S can be negative due to numerical uncertainty
        S2 = np.maximum(S2, 1) 
        
        mean_p = np.dot(mean_d[indLos], U)
        
        ### guess for regularisation - estimate quantile of log(S^2)
        g0 = np.interp(regGuess, Q, np.log(S2))
        
        if f == nFisher - 1:
            ### last step - find optimal regularisation
            S = np.sqrt(S2)
            
            g0, log_fg2 = fn.FindMin(fn.GCV, g0, 1, mean_p, S, U.T) # slowest step
            ### avoid too small regularisation when min of GCV is not found
            
            gmin = np.interp(regMin, Q, np.log(S2))
            g0 = max(g0, gmin)
            
            ### filtering factor
            w = 1. / (1. + np.exp(g0) / S2)

            V = np.dot(H, U / S)
            V = solve_banded(
                (1,1), WD, V, overwrite_ab=True, 
                overwrite_b=True, check_finite=False
            )
            
        else:
            ### filtering factor
            w = 1. / (1. + np.exp(g0) / S2)
            
            ### calcualte y without evaluatin v explicitely
            Y = np.dot(H, np.dot(U / S2, w * mean_p))
            ### final inversion of mean solution , reconstruction
            Y = solve_banded(
                (1,1), WD, Y, overwrite_ab=True, 
                overwrite_b=True, check_finite=False
            )

            ### weight matrix for the next iteration
            W = 1. / np.maximum(Y, 1e-10)**.5
    
    ### 
    p = np.dot(d[indLos], U)
    Y = np.dot((w / S) * p, V.T)
    # print(fit.shape, fit)
    print(p.shape, p)
    print(w.shape, w)
    print(U.T.shape, U.T)
    
    backprojection[i,indLos] = fit = np.dot(p * w, U.T)
    chi2[i] = np.sum((d[indLos] - fit)**2) / len(fit)
    gamma[i] = np.interp(g0, np.log(S2), Q)
    
    y[i,indSpace] = Y
    yErr[i,indSpace] = np.sqrt(np.dot(V**2, (w / S)**2))
    backprojection[i] *= err[i]
    
y *= scale
yErr *= scale
RgridB = (Rgrid[1:] + Rgrid[:-1]) / 2.

print('all done')

fig, ax = plt.subplots(1, 1, figsize=(3.5,3), dpi=150)
ax.plot(R, data[0], '-', c='k', zorder=3, label='raw RBA')
ax.plot(RgridB, y[0], '-', c='C0', zorder=2, label='inversion')
ax.fill_between(RgridB, y[0]-yErr[0], y[0]+yErr[0], color='C0', alpha=0.2, zorder=2)
ax.plot(R, backprojection[0,:], c='C2', zorder=4, label='reconst. RBA')
xplot = [0.2,1.875]
ax.set_xlim(xplot)
ax.plot(xplot, [0.,0.], '-k', lw=0.8, zorder=1)
ax.set_xlabel('R (m)', fontsize=9)
ax.set_ylabel('Units', fontsize=9)
ax.legend(fancybox=1, framealpha=1, fontsize=8)
ax.tick_params(axis="both", which='both', labelsize=9, direction='in', 
              left=True, bottom=True, right=True, top=False)
# ax.set_title(title, fontsize=9)

# plt.savefig('/home/sthoma/Documents/Plots/MU04/Inversions/' + 
#             title.replace(' ', '_').replace('=', '_') + '.png')

# ax2 = ax.twinx()
# ax2.plot(R, vignette, c='C3')
# ax2.set_ylabel('Vignette fn', c='C3')
# ax2.plot(xplot, [1.,1.], '-k', lw=0.8, zorder=1)
plt.tight_layout()
plt.show()


np.savez(
    'testing0.npz', 
    R = R,
    data = data,
    Rgrid = Rgrid,
    y = y,
    yErr = yErr,
    backprojection = backprojection,
)