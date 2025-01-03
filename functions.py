#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import eigh, solve_banded


def calcSiz(
    emissivity, temperature, density, fExcite, 
    scaleExcite, fIonise, scaleIonise
    ):
    SCD = fIonise((temperature, density)) * scaleIonise
    EXC = fExcite((temperature, density)) * scaleExcite
    ionisation = 4. * np.pi * emissivity * SCD / EXC
    return ionisation


def calcErr(
    function, y, emissivity, emissivityErr, temperature, 
    density, fExcite, scaleExcite, f2, scale2, percent=0.1
    ):
    emisBounds = np.vstack(
        (emissivity - emissivityErr, emissivity + emissivityErr)
    )
    tempBounds = np.vstack(
        ((1. - percent) * temperature, (1. + percent) * temperature)
    )
    densBounds = np.vstack(
        ((1. - percent) * density, (1. + percent) * density)
    )
    err = np.zeros((8, len(emissivity)))
    I = 0
    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                err[I] = function(
                    emisBounds[i], tempBounds[j], densBounds[k], 
                    fExcite, scaleExcite, f2, scale2
                )
                I += 1
    err = np.abs(y - err).max(axis=0)
    return err


def calcN0(
    emissivity, temperature, density, fExcite, 
    scaleExcite, fRecomb, scaleRecomb
    ):
    REC = fRecomb((temperature, density)) * scaleRecomb * density * density
    EXC = fExcite((temperature, density)) * scaleExcite * density
    n0 = ((4. * np.pi * emissivity) - REC) / EXC
    return n0


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


def findNearest(arr, val):
    return np.abs(arr - val).argmin()


def findPosition(x, y, mult, kind='exp'):
    """
    kind must be either 'linear' or 'exp'
    """
    ind = y.argmax()
    yMax = y[ind]
    if kind == 'exp':
        cutoff = yMax * np.exp(mult)
    elif kind == 'linear': 
        cutoff = yMax * mult
    xVal0 = np.interp(cutoff, y[:ind+1], x[:ind+1])
    xVal1 = np.interp(cutoff, y[ind:][::-1], x[ind:][::-1])
    return xVal0, xVal1


def findPositionErr(x, y, yErr, mult, x0=None, x1=None, kind='exp'):
    xA, xB = findPosition(x, y+yErr, mult, kind=kind)
    xa, xb = findPosition(x, y-yErr, mult, kind=kind)
    if (x0 is None) or (x1 is None):
        x0, x1 = findPosition(x, y, mult, kind=kind)
    x0Err = np.max([abs(x0-xA), abs(x0-xa)])
    x1Err = np.max([abs(x1-xB), abs(x1-xB)])
    err = np.sqrt(x0Err**2 + x1Err**2)
    return err


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
    
    ###
    yErr = np.sqrt(np.dot(V**2, (w / S)**2))
    backprojection = fit = np.dot(p*w, U.T) * err
    chi2 = np.sum((mean_d[indLos] - fit)**2) / len(fit)
    gamma = np.interp(g0, np.log(S2), Q)
    
    return Y, yErr, backprojection, chi2, gamma


def loadDict(dictFile):
    ### load the dictionary with parameters to run the script
    with open(dictFile) as f:
        textDict = f.read()
    inputDict = json.loads(textDict)
    return inputDict


def loadADAS(line='dalpha'):
    dir = './ADAS/'
    if line == 'dalpha':
        excite = np.load(dir + '/EXCIT_pec12#h_pju#h0.npz')
        recomb = np.load(dir + '/RECOM_pec12#h_pju#h0.npz')
        ionise = np.load(dir + '/IONIS_scd12h.npz')
        Te = excite['Te'] # eV
        ne = excite['ne'] # m^-3
        dataExcite = excite['data'] # ph m^3 s^-1
        dataRecomb = recomb['data'] # ph m^3 s^-1
        dataIonise = ionise['data'] # m^3 s^-1
    return Te, ne, dataExcite, dataRecomb, dataIonise


def makeADAS(
    line='dalpha', excite='cubic', recomb='cubic', ionise='linear'
    ):
    Te, ne, dataExcite, dataRecomb, dataIonise = loadADAS(line=line)
    scaleExcite = 10**int(np.log10(np.median(dataExcite)))
    fExcite = RegularGridInterpolator(
        (Te, ne), dataExcite.T / scaleExcite, method=excite
        )
    scaleRecomb = 10**int(np.log10(np.median(dataRecomb)))
    fRecomb = RegularGridInterpolator(
        (Te, ne), dataRecomb.T / scaleRecomb, method=recomb
        )
    scaleIonise = 10**int(np.log10(np.median(dataIonise)))
    fIonise = RegularGridInterpolator(
        (Te, ne), dataIonise.T / scaleIonise, method=ionise
        )
    return fExcite, scaleExcite, fRecomb, scaleRecomb, fIonise, scaleIonise


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


def makeFName(saveDir, shotn, saveFile):
    ### creates directory if needed
    ### iterates filename to stop overwriting
    import os
    
    saveStr = f'{saveDir}{shotn}/'
    if not os.path.exists(saveStr):
        os.makedirs(saveStr)
    
    fname = f'{saveStr}{saveFile}'
    if os.path.isfile(fname):
        count = 1
        fname = fname.replace('.', f'_{count}.')
        while os.path.isfile(fname):
            check = f'_{count}.'
            count += 1
            replace = f'_{count}.'
            fname = fname.replace(check, replace)
    # while os.path.isfile(fname):
    #     # fname = fname.replace('.', '(1).')
    #     fname = fname.replace('.', '_1.')
    
    return fname


def makeScale(data):
    i = 0
    scale = np.median(data[i:])
    while np.isclose(scale, 0.):
        i += 0
        scale = np.median(data[i:])
    return scale


def neutrals(
    emissivity, emissivityErr, temperature, density, fExcite, 
    scaleExcite, fRecomb, scaleRecomb, fIonise, scaleIonise, percent=0.1):
    ### one iteration of the neutrals calculations
    ioniseRate = calcSiz(
        emissivity, temperature, density, 
        fExcite, scaleExcite, fIonise, scaleIonise
    )
    ioniseError = calcErr(
        calcSiz, ioniseRate, emissivity, emissivityErr, temperature, 
        density, fExcite, scaleExcite, fIonise, scaleIonise, percent=percent
    )
    neutralDensity = calcN0(
        emissivity, temperature, density, 
        fExcite, scaleExcite, fRecomb, scaleRecomb
    )
    neutralErr = calcErr(
        calcN0, neutralDensity, emissivity, emissivityErr, temperature, 
        density, fExcite, scaleExcite, fRecomb, scaleRecomb, percent=percent
    )
    return ioniseRate, ioniseError, neutralDensity, neutralErr
    

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


def saveDict(inputDict, dictFile):
    ### save the dictionary used to make the results
    with open(dictFile, 'w') as f:
        json.dump(inputDict, f)
    return