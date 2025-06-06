import cv2
from functions import findNearest
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import pyuda
client = pyuda.Client()
from scipy.special import ndtr


def get(shotn, trange=[-1.,-1.], tind=None, end=False):
    if tind is not None:
        return getSingle(shotn, tind)
    
    _data = client.get_images('rba', shotn, frame_number=0)
    I = _data.right + 1 - _data.left
    J = _data.bottom + 1 - _data.top
    dtype = _data.frames[0].k.dtype
    time = _data.frame_times
    
    if (trange[0] == -1.) and (trange[1] == -1.):
        T = len(time)
        data = client.get_images('rba', shotn)
    else:
        t0 = findNearest(time, trange[0])
        t1 = findNearest(time, trange[1])
        T = t1 - t0
        if end:
            time = time[t0:t1+1]
            data = client.get_images(
                'rba', shotn, first_frame=t0, last_frame=t1+1
            )
            T += 1
        else:
            time = time[t0:t1]
            data = client.get_images(
                'rba', shotn, first_frame=t0, last_frame=t1
            )
        
    frames = np.zeros((I,J,T)).astype(dtype)
    for i in range(0, T):
        frames[...,i] = data.frames[i].k.T
        
    return frames[:,::-1,:], time


def getSingle(shotn, tind):
    data = client.get_images('rba', shotn, frame_number=tind)
    # windowJ = slice(data.top, data.bottom + 1)
    # windowI = slice(data.left, data.right + 1)
    # frame = np.zeros((I,J)).astype(data.frames[0].k.dtype)
    
    # frame[windowI,windowJ] = data.frames[0].k.T
    frame = data.frames[0].k.T
    time = data.frame_times[tind]
    
    return frame[:,::-1], time


def getExposure(shotn, mult=1e-6):
    # TODO: need to remove the mult kwargs
    data = client.get_images('rba', shotn, frame_number=0)
    multText = data.camera.split(sep=',')[-1]
    if multText == '':
        mult = 1e-6
    elif multText == 'ns':
        mult = 1e-9
    else:
        print(f'ERROR: unable to obtain exposure multiplier')
        print(f'for shot # {shotn}.')
        print(f'Exiting function.')
        from sys import exit
        exit(1)
    return data.exposure * mult


def getRz(Rzfile, I0, I1, J0, J1, kind=0):
    
    data = np.load(Rzfile)
    if kind == 0:
        R = data['Rmin0'][I0:I1,J0:J1].mean(axis=0)
        z = data['zmin0'][I0:I1,J0:J1].mean(axis=0)
        
    else:
        R = data['Rmin'][I0:I1,J0:J1].mean(axis=0)
        z = data['zmin'][I0:I1,J0:J1].mean(axis=0)
    
    return R, z


def getVectors(calibFile):
    import calcam
    calib = calcam.Calibration(calibFile)
    
    los = calib.get_los_direction()
    pupil = calib.get_pupilpos()
    
    return los, pupil


def makeImage(shotn, tind, savePath='/home/sthoma/calcam/Work/Inputs/Images/'):
    frame, _ = getSingle(shotn, tind)
    
    file = 'image_{}_{}.png'.format(shotn, tind)
    cv2.imwrite(savePath + file, frame)
    return


def loadVignette(inputDict):
    x0 = inputDict['x0']
    x1 = inputDict['x1']
    dx = inputDict['dx']
    y0 = inputDict['y0']
    y1 = inputDict['y1']
    dy = inputDict['dy']
    xc = inputDict['xc']
    yc = inputDict['yc']
    wx = inputDict['wx']
    wy = inputDict['wy']
    shape = inputDict['shape']
    amp = inputDict['amp']
    offset = inputDict['offset']
    vignetteArgs = (
        x0, x1, dx, y0, y1, dy, xc, yc, wx, wy, shape, amp, offset
    )
    return vignetteArgs


def getFig(shotn, fig='mid'):
    if fig == 'mid':
        sigName = '/aga/hm12'
    elif fig == 'low':
        sigName = '/aga/hl11'
    elif fig == 'high':
        sigName = '/aga/hu08'
    else:
        print('variable fig must be a string of "mid", "low" or "high"')
        print('defaulting to "mid"')
    data = client.get(sigName, shotn)
    return data.data, data.time.data


def calcFigPressure(shotn, twant, fig='mid'):
    data, time = getFig(shotn, fig=fig)
    tind = findNearest(time, twant)
    return data[tind]


def calcFigDensity(pressure, T=300.):
    kB = 1.380649e-23 # boltzmann hardcoded, same as scipy.constants
    n0 = pressure / (kB * T)
    return n0


def vignetteFunction(xy, x0, y0, wx, wy, shape, amp, offset):
    x, y = xy
    z = np.sqrt(((x - x0) / wx) ** 2 + ((y - y0) / wy) ** 2)
    ### this is a divide by 1000, not for changing units but to remove a 
    ### useless fit parameter. See the testing calibration jupyter notebooks
    g = np.exp(-0.5 * (z * 1e-3)**2)
    f = ndtr((z - 1.) / shape)
    output = amp * g * (1. - f) + offset
    return output.ravel()


def vignette(
        x0=0, x1=1023, dx=1, y0=0, y1=1023, dy=1, xc=519.195, yc=500.687,
        wx=140.377, wy=140.257, shape=0.19405, amp=152.067, offset=121.7288
    ):
    x = np.arange(x0, x1+dx, dx)
    y = np.arange(y0, y1+dy, dy)
    xx, yy = np.meshgrid(x, y)
    vignFn = vignetteFunction((xx, yy), xc, yc, wx, wy, shape, amp, offset)
    return vignFn.reshape(xx.shape)


def getWindow(shotn, minus=True):
    ### the indexing of top, etc., starts at 1 not 0 like in python
    if minus:
        number = -1
    else:
        number = 0
    rba = client.get_images('rba', shotn, frame_number=0)
    I0 = rba.top - number
    I1 = rba.bottom - number
    J0 = rba.left - number
    J1 = rba.right - number
    return I0, I1, J0, J1


def transform(vignFn, shotn):
    I0, I1, J0, J1 = getWindow(shotn)
    window = (slice(I0, I1+1), slice(J0, J1+1))
    vignFn = vignFn[window]
    return vignFn.T[:,::-1]


def applyVignette(data, err, inputDict, shotn, dSlice, flipBool, rEnd):
    vignArgs = loadVignette(inputDict)
    vignFn = vignette(*vignArgs)
    vignFn = transform(vignFn, shotn)
    ### last two vignArgs are amp(litude) and offset, respectively
    # vignFn /= vignFn.max()
    # vignFn -= vignArgs[-1]
    vignFn /= (vignArgs[-2] + vignArgs[-1])
    # vignFn = vignFn[dSlice][0]
    vignFn = vignFn[dSlice].mean(axis=0)
    
    if flipBool:
        vignFn = np.flip(vignFn)
    if rEnd:
        rSlice = slice(0, data.shape[1]-1)
    else:
        rSlice = slice(0, data.shape[1])
    
    data[:,rSlice] /= vignFn[np.newaxis,:]
    err[:,rSlice] /= vignFn[np.newaxis,:]
    return data, err


def makeTimeRange(shotn, T0, T1):
    
    time = client.get_images('rba', shotn, frame_number=0).frame_times
    try:
        time0 = time[T0]
    except IndexError:
        print(f'Index error, T0={T0}, len(time)={len(time)}')
        print('Setting T0=-1')
    try:
        time1 = time[T1]
    except IndexError:
        print(f'Index error, T1={T1}, len(time)={len(time)}')
        print('Setting T1=-1')
    
    timeRange = [time0,time1]
    
    return timeRange


def makeTRange(shotn, time0, time1):

    time = client.get_images('rba', shotn, frame_number=0).frame_times
    
    T0 = findNearest(time, time0)
    if T0 == 0:
        print(f'WARNING: index found is first element.')

    T1 = findNearest(time, time1)
    if T1 == (len(time)-1):
        print(f'WARNING: index found is last element.')

    return T0, T1


# def makeBackground(shotn, tend, dSlice, goodChans, flipBool, rEnd):
    
#     backgroundBool = True
#     ### prepping the data
#     if (tend != -1):
#         trange = makeTimeRange(shotn, tend, -1)

#         ### check if there is data at the end
#         if not np.isclose(trange[0], trange[1]):
#             ### get the data and average over z-drection if need be
#             data = get(
#                 shotn, trange=trange, end=True
#             )[0][dSlice[:-1]].mean(axis=0)
#             ### rotate
#             data = data[goodChans,:].T

#             if flipBool:
#                 data = np.flip(data, axis=1)

#             if rEnd:
#                 data = np.insert(data, data.shape[1], 0., axis=1)

#             background = np.mean(data, axis=0, keepdims=True)
        
#         else:
#             backgroundBool = False

#     else:
#         backgroundBool = False
    
#     if not backgroundBool:
#         print('No background subtraction done')
#         if rEnd:
#             background = np.zeros((1,len(goodChans)+1))
#         else:
#             background = np.zeros((1,len(goodChans)))
    
#     # dataLowEnd = data - background
#     # errLow = np.std(dataLowEnd, axis=0, keepdims=True) / 3.
#     errLow = np.std(data, axis=0, keepdims=True)
    
#     return background, errLow


def makeBackground(shotn, tend, dSlice, goodChans, flipBool, rEnd):
    
    ### prepping the data
    if (tend != -1):
        trange = makeTimeRange(shotn, tend, -1)

        ### check if there is data at the end
        if not np.isclose(trange[0], trange[1]):
            ### get the data and average over z-drection if need be
            data = get(
                shotn, trange=trange, end=True
            )[0][dSlice[:-1]].mean(axis=0)
            ### rotate
            data = data[goodChans,:].T

            if flipBool:
                data = np.flip(data, axis=1)

            if rEnd:
                data = np.insert(data, data.shape[1], 0., axis=1)

            errLow = np.std(data, axis=0, keepdims=True)
        
        else:
            
            if rEnd:
                errLow = np.zeros((1,len(goodChange)+1))
                
            else:
                errLow = np.zeros((1,len(goodChange)))

    else:
        
        if rEnd:
            errLow = np.zeros((1,len(goodChange)+1))
            
        else:
            errLow = np.zeros((1,len(goodChange)))
    
    return errLow


def loadMask(maskFile, flipBool, rEnd):
    
    mask = np.load(maskFile)
    if flipBool:
        mask = np.flip(mask)
    if rEnd:
        mask = np.insert(mask, len(mask), True)
    
    return mask


def prepData(
    shotn, rawData, dSlice, goodChans, flipBool, 
    rEnd, tend, sysErr=5., photon=False
    ):
    ### preparing data for the inversion routine
    ### raw should be in the shape (nz, nR, nt)
    data = rawData[dSlice].mean(axis=0)
    data = data[goodChans,:].T
    if flipBool:
        data = np.flip(data, axis=1)
    if rEnd:
        data = np.insert(data, data.shape[1], 0., axis=1)
    
    nR = data.shape[1]
    errLow = makeBackground(shotn, tend, dSlice, goodChans, flipBool, rEnd)
    
    # dataLow = data - background
    # dataLow -= dataLow[:,[-1]]
    
    calf, calfErr = makeCal(nR, sysErr=sysErr)
    
    if photon:
        photonNoise = np.sqrt(data)
        # photonNoise[np.isclose(photonNoise, 0.)] = np.sqrt(0.5)
        errLow = np.sqrt((data * calfErr)**2 + errLow**2 + photonNoise**2)
    else:
        errLow = np.sqrt((data * calfErr)**2 + errLow**2)
    
    errLow = np.maximum(errLow, -data)
    # errLow[np.isclose(errLow, 0.)] = np.inf
    
    data *= calf
    
    return data, errLow


# def prepData(
#     shotn, rawData, dSlice, goodChans, flipBool, rEnd, tend, sysErr=5.
#     ):
#     ### preparing data for the inversion routine
#     ### raw should be in the shape (nz, nR, nt)
#     data = rawData[dSlice].mean(axis=0)
#     data = data[goodChans,:].T
#     if flipBool:
#         data = np.flip(data, axis=1)
#     if rEnd:
#         data = np.insert(data, data.shape[1], 0., axis=1)
    
#     nR = data.shape[1]
#     ### amendment 8thJan2025
#     ### takes into account long plasmas
#     ### amendment 28Feb2025
#     ### changes to allow it to load less data
#     # if (tend != -1) and (tend < (data.shape[0]-1)):
#     #     background = np.mean(data[tend:,:], axis=0, keepdims=True)
#     # else:
#     #     print('No background subtraction done')
#     #     background = np.zeros((1,data.shape[1]))
#     ### background is the mean of the no-plasma signal
#     ### errLow is the std-dev of the no-plasma signal
#     background, errLow = makeBackground(
#         shotn, tend, dSlice, goodChans, flipBool, rEnd
#     )
#     ### end of amendment
#     ### end of amendment 2
#     # errLow = np.std(dataLow[tend:,:], axis=0, keepdims=True) / 3.
    
#     dataLow = data - background
#     dataLow -= dataLow[:,[-1]]
    
#     # ind1 = np.r_[1,0:nR-1]
#     # ind2 = np.r_[1:nR,nR-2]
#     # errLow += np.std(
#     #     np.diff(
#     #         dataLow - (dataLow[:,ind1] + dataLow[:,ind2]) / 2., axis=0
#     #     ), axis=0
#     # ) / np.sqrt(2.)
    
#     calf, calfErr = makeCal(nR, sysErr=sysErr)
    
#     errLow = np.sqrt((dataLow * calfErr)**2 + errLow**2)
#     errLow = np.maximum(errLow, -dataLow)
#     errLow[np.isclose(errLow, 0.)] = np.inf
    
#     data = dataLow * calf
    
#     return data, errLow


def makeCal(nR, sysErr=5.):
    # TODO: needs updating with proper calibration
    calf = np.ones(nR)
    calfErr = np.ones(nR) * sysErr / 100.
    return calf, calfErr


def makeRadialAverage(data, raverage, rEnd):
    ### assumes data is 2D with shape (nT, nR)
    data2 = np.zeros_like(data).astype(float)
    kernel = np.ones(raverage).astype(float) / raverage
    
    if rEnd:
        rSlice = slice(0, data.shape[1]-1)
    else:
        rSlice = slice(0, data.shape[1])
    
    for i in range(0, data.shape[0]):
        data2[i,rSlice] = np.convolve(data[i,rSlice], kernel, mode='same')
    return data2
        

def makeTimeAverage(data, taverage):
    ### assumes data is 2D with shape (nT, nR)
    data2 = np.zeros_like(data).astype(float)
    kernel = np.ones(taverage).astype(float) / taverage
    
    for i in range(0, data.shape[1]):
        data2[:,i] = np.convolve(data[:,i], kernel, mode='same')
    return data2


def applyCalibration(data, err, inputDict):
    m = inputDict['mPhotons']
    mErr = inputDict['mPhotonsErr']
    data, _ = count2photon(data, m=m, mErr=mErr)
    err, _ = count2photon(err, m=m, mErr=mErr)
    return data, err
    

def count2photon(data, m=8.7051e12, mErr=6.4e9):
    photons = data * m #* 4. * np.pi
    photonsErr = data * mErr #* 4. * np.pi
    return photons, photonsErr


def applyExposure(data, err, shotn, mult=1e-6):
    # TODO: need to remove the mult kwargs
    exposureTime = getExposure(shotn, mult=mult)
    data /= exposureTime
    err /= exposureTime
    return data, err


def mtanh(R, R0, height, width, grad, bkgd):
    ### modified tanh function
    z = -4. * (R - R0) / width
    L = 1. / (1. + np.exp(-z))
    profile = (L * (height - bkgd + ((z * width * grad) / 4.))) + bkgd
    return profile


def mtanhGradient(R, R0, height, width, grad, bkgd):
    sigma = 0.25 * width
    z = -4. * (R - R0) / width
    L = 1. / (1. + np.exp(-z))
    c = 4. * (height - bkgd) / width
    return -1. * L * ((1. - L) * (c + grad * z) + grad)


def getPedestal(shotn, kind, prefix='/apf/core/mtanh/lfs/', trange=[-1.,-1.]):
    time = client.get(prefix+'time', shotn).data
    R0 = client.get(prefix+kind+'/pedestal_location', shotn).data
    height = client.get(prefix+kind+'/pedestal_height', shotn).data
    width = client.get(prefix+kind+'/pedestal_width', shotn).data
    grad = client.get(prefix+kind+'/pedestal_top_gradient', shotn).data
    bkgd = client.get(prefix+kind+'/background_level', shotn).data
    if (trange[0] == -1.) and (trange[1] == -1.):
        boo = np.ones_like(time).astype(bool)
    else:
        boo = (time >= trange[0]) * (time <= trange[1])
    return time[boo], R0[boo], height[boo], width[boo], grad[boo], bkgd[boo]


def getPedestalSamples(
    shotn, kind, prefix='/apf/core/mtanh/lfs/', trange=[-1.,-1.]
    ):
    time = client.get(prefix+'time', shotn).data
    R0 = client.get(prefix+kind+'/pedestal_location_samples', shotn).data
    height = client.get(prefix+kind+'/pedestal_height_samples', shotn).data
    width = client.get(prefix+kind+'/pedestal_width_samples', shotn).data
    grad = client.get(prefix+kind+'/pedestal_top_gradient_samples', shotn).data
    bkgd = client.get(prefix+kind+'/background_level_samples', shotn).data
    if (trange[0] == -1.) and (trange[1] == -1.):
        boo = np.ones_like(time).astype(bool)
    else:
        boo = (time >= trange[0]) * (time <= trange[1])
    return R0[boo], height[boo], width[boo], grad[boo], bkgd[boo]


def makePedestalSamples(Rprofile, R0, height, width, grad, bkgd):
    profiles = np.zeros((len(R0), len(Rprofile)))
    for i in range(0, len(R0)):
        profiles[i] = mtanh(
            Rprofile, R0[i], height[i], width[i], grad[i], bkgd[i]
        )
    return profiles


def getThomson(shotn, kind, prefix='/ayc/', trange=[-1.,-1.]):
    time = client.get(prefix+'time', shotn).data
    data = client.get(prefix+kind, shotn).data
    err = client.get(prefix+'d'+kind, shotn).data
    R = client.get(prefix+'R', shotn).data
    if (trange[0] == -1.) and (trange[1] == -1.):
        boo = np.ones_like(time).astype(bool)
    else:
        boo = (time >= trange[0]) * (time <= trange[1])
    return time[boo], data[boo], err[boo], R[boo]


def getPsiN(shotn, t, R, z):
    """
    Find normalised flux surfaces in MAST/MAST-U for a specified
    shot. shotn should be an integer, t, R, and z can be floats
    or array-like.
    psiN is returned with shape according to (nz, nR, nt)
    """
    from pyEquilibrium.equilibrium import equilibrium as equil
    if shotn >= 45177:          # decide on which device needed
        device = 'MASTU'
    else:
        device = 'MAST'
    # make t, R and z the right types and shapes
    if isinstance(t ,float):
        t = np.array([t])
    Nt = len(t)
    # if Nt > 1:
    #     if np.diff(t).mean() < dt:
    #         t = t[::int(dt / roundSF(np.diff(t).mean(), 2))]
    #         Nt = len(t)
    if isinstance(R, float):
        R = np.array([[R]])
    if isinstance(z, float):
        z = np.array([[z]])
    if R.ndim == 2 and z.ndim == 2 and R.shape == z.shape:
        N1 = R.shape[0]
        N2 = R.shape[1]
    else:
        R, z = np.meshgrid(R, z)
        N1 = R.shape[0]
        N2 = R.shape[1]
    psiN = np.zeros((Nt, N1, N2)) # empty psiN, then find it
    for i in range(0, Nt):
        equi = equil(device=device, shot=shotn, time=t[i])
        for j in range(0, N1):
            for k in range(0, N2):
                psiN[i,j,k] = equi.psiN(R[j,k], z[j,k])
    return psiN, t


def getMinorRadius(
    shotn, time, Rwant, zwant, Rmin=0.1, Rmax=1.5, dR=5e-4,
    device='MASTU', mastu_prefix='epm', efitpp_file=None
    ):
    ### mastu_prefix = 'epm' or 'epq'
    from pyEquilibrium.equilibrium import equilibrium as equil
    from scipy.interpolate import interp1d
    
    ### make the Rwant a list if only a single value is provided
    if not isinstance(Rwant, (list, np.ndarray)):
        Rwant = [Rwant]
    if not isinstance(zwant, (list, np.ndarray)):
        zwant = [zwant]

    N = min(len(Rwant), len(zwant))
    ### make the equilibrium object
    if efitpp_file:
        eq = equil(device='MASTU', time=time, efitpp_file=efitpp_file)
    else:
        eq = equil(
            device='MASTU', shot=shotn, time=time, 
            mastu_prefix=mastu_prefix, efitpp_file=efitpp_file
        )
    ### find the magnetic axis
    Raxis, Zaxis = eq.axis
    ### make the Rarray and find psiN for it
    R = np.arange(Rmin, Rmax + (dR*0.5), dR)
    psiN = eq.psiN(R, Zaxis)[0]
    ### the index closest to the magnetic axis
    ind = psiN.argmin()
    ### the index where psiN turns in the centre column
    ind0 = psiN[:ind].argmax()
    ### chop the arrays down from this point
    R = R[ind0:]
    psiN = psiN[ind0:]
    ind -= ind0
    ### make the inteprolation function
    hfsFunc = interp1d(psiN[:ind+1], R[:ind+1])
    lfsFunc = interp1d(psiN[ind:], R[ind:])

    ### empty array for the results
    minorRadius = np.zeros(N)
    for i in range(0, N):
        ### find psiN of the position I want
        psiWant = eq.psiN(Rwant[i], zwant[i])[0][0]
        try:
            ### inteprolate to find majorRadius
            hfsR = hfsFunc(psiWant)
            lfsR = lfsFunc(psiWant)
            ### now find the difference
            minorRadius[i] = (lfsR - hfsR) * 0.5
        ### assumes it failed because it couldn't find the inner psiN
        except ValueError:
            ###
            minorRadius[i:] = Rwant[i:] - Rwant[i-1] + minorRadius[i-1]
            break
            
    return minorRadius


def getDalpha(shotn, sigName='/XIM/DA/HM10/T', trange=[-1.,-1.]):
    
    data = client.get(sigName, shotn)
    time, sig = data.time.data, data.data
    
    if (trange[0] == -1.) and (trange[1] == -1.):
        boo = np.ones_like(time).astype(bool)
    
    else:
        boo = (time >= trange[0]) * (time <= trange[1])
    
    return time[boo], sig[boo]


def processDalpha(
    shotn, sigName='/XIM/DA/HM10/T', trange=[0.1,0.95], moving_av_length=1e-3, 
    ):

    ### return whole signal, chop down after processing
    time, sig = getDalpha(shotn, sigName=sigName, trange=[-1.,-1.])
    
    n = int(moving_av_length / (time[1] - time[0]))
    ret = np.cumsum(sig, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret = ret[n - 1:] / n
    sig[n-1:] -= ret

    if (trange[0] == -1.) and (trange[1] == -1.):
        boo = np.ones_like(time).astype(bool)
    
    else:
        boo = (time >= trange[0]) * (time <= trange[1])
    
    return time[boo], sig[boo]


def identify_bursts(series, thresh):#, analyse=False):
    
    inds_above_thresh = np.where(series>thresh)
    inds_below_thresh = np.where(series<=thresh)
    windows = [] 
    leng = len(inds_above_thresh[0])+0.0
    for i, ind in enumerate(inds_above_thresh[0]):
        try:
            #Extract highest possible index that occurs before a filement,
            ind_low = np.extract(inds_below_thresh < ind,inds_below_thresh)[-1]
            #Extract lowest possible index that occurs after a filemant
            ind_up = np.extract(inds_below_thresh > ind,inds_below_thresh)[0]
            if (ind_low,ind_up) not in windows:
                #Make sure that there is no double counting
                windows.append((ind_low,ind_up))
        except:
            pass
    
#     if analyse:
#         N_bursts = len(windows)
#         burst_ratio = len(list(series))/N_bursts
#         av_window = np.mean([y - x for x,y in windows])
#         return windows, N_bursts, burst_ratio, av_window
#     else:
#         return windows

    return windows


def detectELMs(
    shotn, sigName='/XIM/DA/HM10/T', trange=[0.1,0.95], moving_av_length=1e-3, 
    minDuration=5e-5, maxDuration=5e-2, minSeperation=2e-3, threshold=0.011, 
    interval=2e-2, full=True, 
    ):
    
    dtime, dalpha = processDalpha(
        shotn, sigName=sigName, trange=trange, 
        moving_av_length=moving_av_length, 
    )
    
    orig_windows = identify_bursts(dalpha, threshold)#, analyse=False)


    windows = orig_windows[:]
    too_short_windows = []
    too_long_windows = []
    num_too_short = 0
    num_too_long = 0
    
    for i in range(len(windows)):
        if (dtime[windows[i][1]] - dtime[windows[i][0]]) < minDuration:
            too_short_windows.append(windows[i])
            windows[i] = (0,0)
            num_too_short+=1
        if (dtime[windows[i][1]] - dtime[windows[i][0]]) > maxDuration:
            too_long_windows.append(windows[i])
            windows[i] = (0,0)
            num_too_long += 1
            
    windows = [x for x in windows if x!=(0,0)]
    
    too_short_inds = [
        np.argmax(
            dalpha[too_short_windows[x][0]:too_short_windows[x][1]]
        ) + too_short_windows[x][0] for x in range(len(too_short_windows))
    ]
    too_long_inds = [
        np.argmax(
            dalpha[too_long_windows[x][0]:too_long_windows[x][1]]
        ) + too_long_windows[x][0] for x in range(len(too_long_windows))
    ]
    
    too_short = dalpha[too_short_inds]
    too_short_t = dtime[too_short_inds]
    
    too_long = dalpha[too_long_inds]
    too_long_t = dtime[too_long_inds]
    
    elm_inds = [
        np.argmax(
            dalpha[windows[x][0]:windows[x][1]]
        ) + windows[x][0] for x in range(len(windows))
    ]
    elm_heights = dalpha[elm_inds]
    elm_times = dtime[elm_inds]
    
    too_recurrent = 0
    freq_filtered_inds = np.zeros(0, dtype=int)
    for i in range(0, len(elm_inds)-1):
        if (elm_heights[i] != 0) and (i not in freq_filtered_inds):
            close_inds = np.where(
                elm_times[i+1:] - elm_times[i] < minSeperation
            )[0] + i + 1
            freq_filtered_inds = np.concatenate((freq_filtered_inds, close_inds))
    
    freq_filtered_inds = np.unique(freq_filtered_inds)
    too_recurrent = len(freq_filtered_inds)
    freq_filtered_t = elm_times[freq_filtered_inds]
    freq_filtered = elm_heights[freq_filtered_inds]
    
    elm_times[freq_filtered_inds] = 0
    elm_heights[freq_filtered_inds] = 0
    
    elm_times = elm_times[np.where(elm_heights != 0)]
    elm_heights = elm_heights[np.where(elm_heights != 0)]

    num_elms = len(elm_heights)
    
    freq_t = np.linspace(
        min(elm_times), max(elm_times), 
        int((max(elm_times) - min(elm_times)) / interval)
    )
    freq = np.zeros(len(freq_t))
    
    for index, time in enumerate(freq_t):
        num = len(
            np.where(
                (elm_times > time - interval / 2) & 
                (elm_times < time + interval / 2)
            )[0]
        )
        freq[index] = num / interval
    
    # dtime(alpha) - time (and signal) of the filtered d-alpha trace
    # freq(_t) - time smoothed frequency
    # freq_filtered(_t) - too close to another ELM
    # too_short(_t) - too short a duration
    # too_long(_t) - too long a duration
    # elm_times(heights) - the time (and height) of the ELMs that made it through
    
    if full:
        return dtime, dalpha, freq_t, freq, freq_filtered_t, freq_filtered, \
        too_short_t, too_short, too_long_t, too_long, elm_times, elm_heights
    
    else:
        return dtime, dalpha, elm_times, elm_heights

#
    """
    
    function abel_inversion, r,g

;computes the inverse Abel transform of radial profiles. Can convert a
;chord-integrated tangential brightness measurement into local emissivity
;If converting brightness (Power/area/ster) to emissivity (Power/vol),
;then multiply result by 4pi 

nr=n_elements(r)
result=fltarr(nr)
dr=r-shift(r,1)
dr(0)=dr(1)
dg=g-shift(g,1)
dg(0)=0.

for i=nr-1,0,-1 do begin
   for j=i,nr-1 do begin
      integrand=-1/!pi*dg(j)/sqrt(r(j)^2.-r(i)^2.)
      if finite(integrand) then result(i)=result(i)+integrand
   endfor   
endfor

return,result

end
    """