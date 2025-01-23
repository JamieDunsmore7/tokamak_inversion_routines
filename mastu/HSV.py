import calcam
import cv2
from functions import findNearest
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import pyuda
client = pyuda.Client()
from scipy.special import ndtr


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
    time = data.frame_times[tind]
    
    return frame[:,::-1], time


def getExposure(shotn, mult=1e-6):
    data = client.get_images('rba', shotn, frame_number=0)
    return data.exposure * mult


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
    vignFn = vignFn[dSlice][0]
    
    if flipBool:
        vignFn = np.flip(vignFn)
    if rEnd:
        rSlice = slice(0, data.shape[1]-1)
    else:
        rSlice = slice(0, data.shape[1])
    
    data[:,rSlice] /= vignFn[np.newaxis,:]
    err[:,rSlice] /= vignFn[np.newaxis,:]
    return data, err


def prepData(rawData, dSlice, goodChans, flipBool, rEnd, tend, sysErr=5.):
    ### preparing data for the inversion routine
    ### raw should be in the shape (nz, nR, nt)
    data = rawData[dSlice][0] # remove z axis so it's 2D
    data = data[goodChans,:].T
    if flipBool:
        data = np.flip(data, axis=1)
    if rEnd:
        data = np.insert(data, data.shape[1], 0., axis=1)
    
    nR = data.shape[1]
    ### amendment 8thJan2025
    ### takes into account long plasmas
    if (tend != -1) and (tend < (data.shape[0]-1)):
        background = np.mean(data[tend:,:], axis=0, keepdims=True)
    else:
        print('No background subtraction done')
        background = np.zeros((1,data.shape[1]))
    ### end of amendment
    dataLow = data - background
    errLow = np.std(dataLow[tend:,:], axis=0, keepdims=True) / 3.
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


def getMinorRadius(shotn, time, Rwant, zwant, Rmin=0.1, Rmax=1.5, dR=5e-4, device='MASTU', mastu_prefix='epm'):
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
    eq = equil(device='MASTU', shot=shotn, time=time, mastu_prefix=mastu_prefix)
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