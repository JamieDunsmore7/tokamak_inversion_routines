import calcam
import cv2
import numpy as np
import pyuda
client = pyuda.Client()


def findNearest(arr, val):
    return np.abs(arr-val).argmin()


def get(shotn, trange=[-1.,-1.], tind=None, I=1024, J=1024):
    if tind is not None:
        return getSingle(shotn, tind, I=I, J=J)
    
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


def getSingle(shotn, tind, I=1024, J=1024):
    data = client.get_images('rba', shotn, frame_number=tind)
    # windowJ = slice(data.top, data.bottom + 1)
    # windowI = slice(data.left, data.right + 1)
    # frame = np.zeros((I,J)).astype(data.frames[0].k.dtype)
    
    # frame[windowI,windowJ] = data.frames[0].k.T
    frame = data.frames[0].k.T
    time = data.frame_times[0]
    
    return frame[:,::-1], time


def getVectors(calibFile):
    calib = calcam.Calibration(calibFile)
    
    los = calib.get_los_direction()
    pupil = calib.get_pupilpos()
    
    return los, pupil
    

def makeImage(shotn, tind, savePath='/home/sthoma/calcam/images/', I=1024, J=1024):
    frame, _ = getSingle(shotn, tind, I=I, J=J)
    
    file = 'image_{}_{}.png'.format(shotn, tind)
    cv2.imwrite(savePath + file, frame)
    return

#