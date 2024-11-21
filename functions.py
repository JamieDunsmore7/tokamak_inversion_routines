#!/usr/bin/env python3

import json
import numpy as np
from scipy.linalg import eigh, solve_banded


def FindMin(F, x0, dx0, prod, S, U, tol=0.01):
    ### stupid but robust minimum searching algorithm.
    fg = F(x0, prod, S, U)
    while abs(dx0) > tol:
        fg2 = F(x0 + dx0, prod, S, U)
        if fg2 < fg:
            fg = fg2
            x0 += dx0
            continue
        else:
            dx0 /= -2.
    return x0, np.log(fg2)


def GCV(g, prod, S, U):
    ### generalized crossvalidation
    w = 1. / (1. + np.exp(g) / S**2)
    ndets = len(prod)
    return (np.sum((((w - 1) * prod))**2) + 1) / ndets / (1 - np.mean(w))**2


def inversion(data, err, dL, scale, nGrid, Q, D, indLos, 
              indSpace, nFisher=4, regGuess=0.7, regMin=0.4):
    ### one single iteration of the inversion
    ### find zeros, as we're dividing by it
    errZinds = np.where(np.isclose(err, 0.))
    T = dL / err[:,None] * scale
    mean_d = data / err
    ### replace infs and NaNs
    T[errZinds] = 0.
    mean_d[errZinds] = 0.
    
    ### empty array
    W = np.ones(nGrid-1)
    
    ### iterate over the nFisher asked for
    for f in range(nFisher):
        
        ### multiply tridiagonal regularisation operator 
        ### by a diagonal weight matrix W
        WD = np.copy(D)
        ### seems redundent, as W is all ones
        WD[0,1:] *= W[:-1]
        WD[1] *= W
        WD[2,:-1] *= W[1:]
        
        ### transpose the band matrix
        DTW = np.copy(WD)
        DTW[0,1:], DTW[2,:-1] = WD[2,:-1], WD[0,1:]
        
        ### solve Tikhonov regularization (optimised for speed)
        H = solve_banded(
            (1, 1), DTW, T[indLos,indSpace].T, 
            overwrite_ab=True, check_finite=False
        )
        ### fast method to calculate U, S, V = svd(H.T) of rectangular matrix
        LL = np.dot(H.T, H)
        ### also from scipy
        S2, U = eigh(LL, overwrite_a=True, check_finite=False, lower=True)
        ### singular values S can be negative due to numerical uncertainty
        S2 = np.maximum(S2, 1)
        ###
        mean_p = np.dot(mean_d[indLos], U)
        ### guess for regularisation - estimate quantile of log(S^2)
        g0 = np.interp(regGuess, Q, np.log(S2))
        
        if f == (nFisher-1):
            ### last step - find optimal regularisation
            S = np.sqrt(S2)
            
            g0, log_fg2 = FindMin(GCV, g0, 1, mean_p, S, U.T) # slowest step
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
    p = np.dot(mean_d[indLos], U)
    Y = np.dot((w / S) * p, V.T)
    
    backprojection = fit = np.dot(p*w, U.T) * err
    chi2 = np.sum((mean_d[indLos] - fit)**2) / len(fit)
    gamma = np.interp(g0, np.log(S2), Q)
    
    # y[i,indSpace] = Y
    yErr = np.sqrt(np.dot(V**2, (w / S)**2))
    # backprojection[i] *= err[i]
    
    return Y, yErr, backprojection, chi2, gamma


def loadDict(dictFile):
    ### load the dictionary with parameters to run the script
    with open(dictFile) as f:
        textDict = f.read()
    inputDict = json.loads(textDict)
    return inputDict


def makeDL(Rgrid, R):
    dL = 2*(np.sqrt(np.maximum((Rgrid[1:])**2-R[:,None]**2,0)) 
            -np.sqrt(np.maximum( Rgrid[:-1]**2-R[:,None]**2,0)))
    return dL


def makeGrid(R, nGrid):
    Rmin = R[0]
    Rmax = R[-1]
    Rgrid = np.linspace(Rmin, Rmax, nGrid)
    RgridB = (Rgrid[1:] + Rgrid[:-1]) / 2.
    return Rgrid, RgridB


def makeScale(data):
    i = 0
    scale = np.median(data[i:])
    while np.isclose(scale, 0.):
        i += 0
        scale = np.median(data[i:])
    return scale


def prepR(R0, rEnd):
    goodChans = np.ones(len(R0)).astype(bool)
    R = R0[goodChans]
    if np.diff(R).mean() < 0.:
        R = np.flip(R)
        flipBool = True
    else:
        flipBool = False
    if rEnd:
        R = np.insert(R, len(R), rEnd)
    return R, goodChans, flipBool


def regulMatrix(nGrid, biasedEdges=True):
    ### regularization band matrix
    bias = .1 if biasedEdges else 1e-5
    ### (3 x R_grid-1), all 1s
    D = np.ones((3, nGrid-1))
    ### make row 1 all negative 2s
    D[1, :] *= -2
    ### last in middle row is bias value
    D[1, -1] = bias
    ### middle row, first element and second last element are negative 1s
    D[1, [0, nGrid-3]] = -1
    ### last row, elements -2 and -3 are set to zero
    D[2, [-2, -3]] = 0
    return D