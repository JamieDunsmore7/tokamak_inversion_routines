#!/usr/bin/env python3

import json
import numpy as np


def loadDict(dictFile):
    with open(dictFile) as f:
        textDict = f.read()
    inputDict = json.loads(textDict)
    return inputDict


def makeGrid(R, nrMult=2.5):
    nR = int(len(R) * nrMult)
    Rmin = R[0]
    Rmax = R[-1]
    Rgrid = np.linspace(Rmin, Rmax, nR)
    return Rgrid, nR


def makeDL(Rgrid, R):
    dL = 2*(np.sqrt(np.maximum((Rgrid[1:])**2-R[:,None]**2,0)) 
            -np.sqrt(np.maximum( Rgrid[:-1]**2-R[:,None]**2,0)))
    return dL


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
    nR = len(R)
    return R, nR, goodChans, flipBool