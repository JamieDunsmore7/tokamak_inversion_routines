### imports
import calcam
import numpy as np
import os
from os.path import dirname


###############################################################################
###                        define the filenames here                        ###
###############################################################################


### define the calibration file I want to load
calibFile = '51267_684.ccc'
### using a default result file name based off the calibFile name
resultFile = calibFile.split('.')[0] + '.npz'
### do I want to plot the stuff
plot = True


###############################################################################
###                        loading variables and data                       ###
###############################################################################


def cart2cyl(arr):
    """
    arr should be an array of length three and contain the coordinates
    [x, y, z] and returns the cylidrical coordinates [phi, R, z] where
    0 <= phi < 360 degrees
    """

    if arr.ndim == 1:
        arr = arr[np.newaxis,:]
        boo = 0
    else:
        boo = np.ones(arr.shape[0]).astype(bool)
    
    newArr = np.zeros_like(arr)
    newArr[:,0] = np.rad2deg(np.arctan2(arr[:,1], arr[:,0]))
    newArr[newArr[:,0] < 0.,0] += 360.
    newArr[:,1] = np.sqrt(arr[:,0]**2 + arr[:,1]**2)
    newArr[:,2] = arr[:,2]

    return newArr[boo]


def magnitudeFunc(arr, axis=-1):
    """
    only works on cartesian coordinates
    """

    return np.sqrt(np.sum(arr**2, axis=axis))


def makeMags(calib, cad, pupil):
    """
    calculate the length of each LoS based on the LoS and CAD model
    """
    ### creates some raydata object from calcam
    raydata = calcam.raycast_sightlines(calib, cad)
    ### use it to find the ends of the LoS
    ends = raydata.ray_end_coords
    ### magnitudes of each LoS
    mags = magnitudeFunc(ends - pupil[np.newaxis,np.newaxis,:], axis=-1)
    return mags


def raycastMethod(calib, cad, pupil, los, dl=1e-4):
    """
    calib is the calibration calcam object
    cad is the calcam CAD model of MAST-U
    pupil is the pupil location - (3)
    los is the array of lines-of-sight - (Y, X, 3)
    dl=1e-4 is the size of the steps to us in the raycast
    """

    ### create the magnitudes of each LoS
    mags = makeMags(calib, cad, pupil)

    ### arrays to put results into
    Rmin = np.zeros((los.shape[0], los.shape[1]))
    zmin = np.zeros((los.shape[0], los.shape[1]))
    phi = np.zeros((los.shape[0], los.shape[1]))
    
    ### iterate over the lines-of-sight
    for i in range(0, los.shape[0]):
        for j in range(0, los.shape[1]):
            mag = np.min((mags[i,j], 4.)) # 4 is approx diameter
            N = np.arange(0., mag + dl, dl)[:,np.newaxis]
            posns = pupil + (N * los[i,j])
            posnsCyl = cart2cyl(posns)
            Rind = posnsCyl[:,1].argmin()
            phi[i,j], Rmin[i,j], zmin[i,j] = posnsCyl[Rind]
    
    return Rmin, zmin, phi


def vectorMethod(p1, v1, p2=np.array([0.,0.,0.]), v2=np.array([0.,0.,1.])):
    """
    p1 is the pupil location - (3)
    v1 is the array of lines-of-sight - (Y, X, 3)
    p2 is some point on the z-axis - (3)
    v2 is the z-axis - (3)
    """

    ### arrays to put results into
    Rmin = np.zeros((v1.shape[0], v1.shape[1]))
    zmin = np.zeros((v1.shape[0], v1.shape[1]))
    phi = np.zeros((v1.shape[0], v1.shape[1]))

    ### iterate over all the v1 (LoS)
    for i in range(0, v1.shape[0]):
        for j in range(0, v1.shape[1]):
            cross = np.cross(v1[i,j], v2)
            Rmin[i,j] = np.abs(np.dot(p1 - p2, cross) / magnitudeFunc(cross))
            ### for the quadratic solving
            a = magnitudeFunc(v1[i,j])**2
            b = 2. * np.dot(p1, v1[i,j])
            c = magnitudeFunc(p1)**2 - Rmin[i,j]**2
            lambd = -b / (2. * a)
            posn = p1 + (lambd * v1[i,j])
            zmin[i,j] = posn[2]
            posnCyl = cart2cyl(posn)
            phi[i,j] = posnCyl[0]
    
    return Rmin, zmin, phi


###############################################################################
###                           setting up variables                          ###
###############################################################################


### get the name of the directory the file is in
filePath = os.path.dirname(__file__)

############################## for vectorMethod ###############################
### load the files I want
calib = calcam.Calibration(
    filePath.replace('Python', f'\\Outputs\\CalibrationCalcam\\{calibFile}')
)
### get the Lines of Site (LoS) of each pixel
los = calib.get_los_direction()
### get the pupil location of the camera
pupil = calib.get_pupilpos()

############################## for raycastMethod ##############################
### the MAST-U CAD model
cad = calcam.CADModel(
    filePath.replace('Python', '\\Inputs\\Models\\MAST Upgrade.ccm')
)
### enable all the features of the model
cad.set_features_enabled(True)


###############################################################################
###                           using vector algebra                          ###
###############################################################################


### define the centre column vector
p2 = np.array([0., 0., 0.])
v2 = np.array([0., 0., 1.])
### call the vectorMethod function
Rmin0, zmin0, phi0 = vectorMethod(pupil, los, p2=p2, v2=v2)


###############################################################################
###                            using the ray cast                           ###
###############################################################################


### define the step size for the raycast vectors
dl = 1e-4
### call the raycastMethod function
Rmin, zmin, phi = raycastMethod(calib, cad, pupil, los, dl=dl)


###############################################################################
###                           saving and plotting                           ###
###############################################################################


np.savez(
    filePath.replace(
        'Python', f'\\Outputs\\CalibrationCoordinates\\{resultFile}'
    ),
    Rmin0 = Rmin0,
    zmin0 = zmin0,
    phi0 = phi0,
    Rmin = Rmin,
    zmin = zmin,
    phi = phi,
)


if plot:
    import matplotlib.pyplot as plt

    ### I choose 260 because it's near midplane in one of my images
    testInd = 260
    ### set the fontsize
    fs = 9

    fig, ax = plt.subplots(1, 1, figsize=(3.35,2.8), dpi=150)
    ax.plot(Rmin0[testInd], '-', c='C0', lw=0.8, label='Vector calc')
    ax.plot(Rmin[testInd], '--', c='C1', lw=0.8, label='Brute force')
    ax.legend(fancybox=1, framealpha=1, fontsize=fs-1)
    ax.set_xlabel('pixel #', fontsize=fs)
    ax.set_ylabel('Major radius - $R$ (m)',fontsize=fs)
    ax.tick_params(axis="both", which='both', labelsize=fs, direction='in', 
                left=True, bottom=True, right=True, top=False)
    ax.set_title(f'pixel row # {testInd:.0f}', fontsize=fs)
    ax.set_xlim([-20, 500])
    ax.set_ylim([0,1.8])
    plt.tight_layout()

    fig, ax = plt.subplots(1, 1, figsize=(3.35,2.8), dpi=150)
    ax.plot(zmin0[testInd], '-', c='C0', lw=0.8, label='Vector calc A')
    ax.plot(zmin[testInd], '-', c='C3', lw=0.8, label='Brute force')
    ax.legend(fancybox=1, framealpha=1, fontsize=fs-1)
    ax.set_xlabel('pixel #', fontsize=fs)
    ax.set_ylabel('Height - $z$ (m)',fontsize=fs)
    ax.tick_params(axis="both", which='both', labelsize=fs, direction='in', 
                left=True, bottom=True, right=True, top=False)
    ax.set_title(f'pixel row # {testInd:.0f}', fontsize=fs)
    plt.tight_layout()

    fig, ax = plt.subplots(1, 1, figsize=(3.35,2.8), dpi=150)
    ax.plot(phi0[testInd], '-', c='C0', lw=0.8, label='Vector calc A')
    ax.plot(phi[testInd], '-', c='C3', lw=0.8, label='Brute force')
    ax.legend(fancybox=1, framealpha=1, fontsize=fs-1)
    ax.set_xlabel('pixel #', fontsize=fs)
    ax.set_ylabel('Toroidal angle - $\\phi$ ($\\degree$)',fontsize=fs)
    ax.tick_params(axis="both", which='both', labelsize=fs, direction='in', 
                left=True, bottom=True, right=True, top=False)
    ax.set_title(f'pixel row # {testInd:.0f}', fontsize=fs)
    plt.tight_layout()

    plt.show()