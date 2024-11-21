#!/usr/bin/env python3

import json
import numpy as np


def FindMin(F, x0, dx0, prod, S, U, tol=0.01):
    #stupid but robust minimum searching algorithm.
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
    #generalized crossvalidation
    w = 1. / (1. + np.exp(g) / S**2)
    ndets = len(prod)
    return (np.sum((((w - 1) * prod))**2) + 1) / ndets / (1 - np.mean(w))**2


def loadDict(dictFile):
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
    return Rgrid


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
    #regularization band matrix
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