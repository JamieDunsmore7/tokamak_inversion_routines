import calcam
import cv2
import numpy as np
import pyuda
client = pyuda.Client()
from scipy.special import ndtr


def findNearest(arr, val):
    return np.abs(arr - val).argmin()


def get(shotn, trange=[-1.,-1.], tind=None):
    if tind is not None:
        return getSingle(shotn, tind)
    
    data = client.get_images('rba', shotn)
    # windowJ = slice(data.top, data.bottom + 1)
    # windowI = slice(data.left, data.right + 1)
    I = data.right + 1 - data.left
    J = data.bottom + 1 - data.top
    dtype = data.frames[0].k.dtype
    time = data.frame_times
    
    if (trange[0] == -1.) and (trange[1] == -1.):
        t0 = 0
        T = len(data.frames)
    else:
        t0 = findNearest(time, trange[0])
        t1 = findNearest(time, trange[1])
        T = t1 - t0
        time = time[t0:t1]
        
    frames = np.zeros((I,J,T)).astype(dtype)
    for i in range(0, T):
        # frames[windowI,windowJ,i] = data.frames[i+t0].k.T
        frames[...,i] = data.frames[i+t0].k.T
        
    return frames[:,::-1,:], time


def getSingle(shotn, tind):
    data = client.get_images('rba', shotn, frame_number=tind)
    # windowJ = slice(data.top, data.bottom + 1)
    # windowI = slice(data.left, data.right + 1)
    # frame = np.zeros((I,J)).astype(data.frames[0].k.dtype)
    
    # frame[windowI,windowJ] = data.frames[0].k.T
    frame = data.frames[0].k.T
    time = data.frame_times[0]
    
    return frame[:,::-1], time


def getRz(Rzfile, I0, I1, J0, J1):
    
    data = np.load(Rzfile)
    R = data['Rmin'][I0,J0:J1]
    z = data['zmin'][I0,J0:J1]
    
    return R, z


def getVectors(calibFile):
    calib = calcam.Calibration(calibFile)
    
    los = calib.get_los_direction()
    pupil = calib.get_pupilpos()
    
    return los, pupil


def makeImage(shotn, tind, savePath='/home/sthoma/calcam/images/'):
    frame, _ = getSingle(shotn, tind)
    
    file = 'image_{}_{}.png'.format(shotn, tind)
    cv2.imwrite(savePath + file, frame)
    return


def vignetteFunction(xy, x0, y0, wx, wy, shape, amp, offset):
    x, y = xy
    z = np.sqrt(((x - x0) / wx) ** 2 + ((y - y0) / wy) ** 2)
    ### this is a divide by 1000, not for changing units but to remove a 
    ### useless fit parameter. See the testing calibration jupyter notebooks
    g = np.exp(-0.5 * (z * 1e-3)**2)
    f = ndtr((z - 1.) / shape)
    output = amp * g * (1. - f) + offset
    return output.ravel()


def vignette(x0=0, x1=1023, dx=1, y0=0, y1=1023, dy=1, xc=519.195, yc=500.687, 
             wx=140.377, wy=140.257, shape=0.19405, amp=152.067, offset=121.7288):
    x = np.arange(x0, x1+dx, dx)
    y = np.arange(y0, y1+dy, dy)
    xx, yy = np.meshgrid(x, y)
    vignFn = vignetteFunction((xx, yy), xc, yc, wx, wy, shape, amp, offset)
    return vignFn.reshape(xx.shape)


def getWindow(shotn):
    rba = client.get_images('rba', shotn, frame_number=0)
    I0 = rba.top
    I1 = rba.bottom
    J0 = rba.left
    J1 = rba.right
    return I0, I1, J0, J1


def transform(vignFn, shotn):
    I0, I1, J0, J1 = getWindow(shotn)
    window = (slice(I0, I1+1), slice(J0, J1+1))
    vignFn = vignFn[window]
    return vignFn.T[:,::-1]


def prepData(raw, dSlice, goodChans, flipBool, rEnd):
    ### raw should be in the shape (nz, nR, nt)
    data = np.mean(raw[dSlice], axis=-1, keepdims=True)[0]
    data = data[goodChans,:]
    if flipBool:
        data = np.flip(data, axis=0)
    if rEnd:
        data = np.insert(data, data.shape[0], 0., axis=0)
    return data.T


def offset(data, raw, dSlice, tend, nR, sysErr=5.):
    background = np.mean(
        raw[dSlice][:,:,tend:], axis=-1, keepdims=True
    )[0].T
    dataLow = data - background
    errLow = np.std(dataLow, axis=0, keepdims=True) / 3.
    ind1 = np.r_[1,0:nR-1]
    ind2 = np.r_[1:nR,nR-2]
    errLow += np.std(
        np.diff(
            dataLow - (dataLow[:,ind1] + dataLow[:,ind2]) / 2., axis=0
        ), axis=0
    ) / np.sqrt(2.)
    dataLow -= dataLow[:,[-1]]
    errLow = np.maximum(errLow, -dataLow)
    
    calf, calfErr = makeCal(nR, sysErr=sysErr)
    data = dataLow * calf
    err = np.sqrt(
        (errLow * calf)**2 + (dataLow * calfErr)**2
    )
    return data, err


def makeCal(nR, sysErr=5.):
    # TODO: needs updating with proper calibration
    calf = np.ones(nR)
    calfErr = np.ones(nR) * sysErr / 100.
    return calf, calfErr







#