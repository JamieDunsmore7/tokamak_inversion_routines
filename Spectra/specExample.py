import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import pyuda
from scipy.optimize import curve_fit


### function for loading all the spectrometer channels, 5 in total
def getSpec(shotn, wl='linear'):
    if wl == 662:
        coeff = np.array([-8.003857638849921e-08, 0.008115909538641598, 656.594265617137])
    elif wl == 'linear':
        coeff = np.array([8.07716749e-3, 6.52544930e2])
    else:
        # wl=658
        coeff = np.array([-7.962860863439673e-08, 0.008153179395386338, 652.5690918688852])
    px = np.polyval(coeff, np.arange(1, 1341))
    client = pyuda.Client()
    time = client.get('/XSM/MSE/SPEC/CH00', shotn).time.data
    data = np.zeros((5,len(time),len(px)))
    for i in range(0, 5):
        data[i] = client.get('/XSM/MSE/SPEC/CH'+str(i).zfill(2), shotn).data
    return time, data, px


### define a bunch of functions that we can fit to the data depending on the
### number of peaks
### define the one gaussian
def gaussian(x, A, x0, sigma):
    return A * np.exp(-(x - x0)**2 / (2. * sigma**2))
def func1(x, A1, x01, s1, C):
    y = gaussian(x, A1, x01, s1) + C
    return y
def func2(x, A1, x01, s1, A2, x02, s2, C):
    y = gaussian(x, A1, x01, s1) + gaussian(x, A2, x02, s2) + C
    return y
def func3(x, A1, x01, s1, A2, x02, s2, A3, x03, s3, C):
    y = gaussian(x, A1, x01, s1) + gaussian(x, A2, x02, s2) + \
        gaussian(x, A3, x03, s3) + C
    return y
def func4(x, A1, x01, s1, A2, x02, s2, A3, x03, s3, A4, x04, s4, C):
    y = gaussian(x, A1, x01, s1) + gaussian(x, A2, x02, s2) + \
        gaussian(x, A3, x03, s3) + gaussian(x, A4, x04, s4) + C
    return y
def func5(x, A1, x01, s1, A2, x02, s2, A3, x03, s3, A4, x04, s4, A5, x05, s5, C):
    y = gaussian(x, A1, x01, s1) + gaussian(x, A2, x02, s2) + \
        gaussian(x, A3, x03, s3) + gaussian(x, A4, x04, s4) + \
        gaussian(x, A5, x05, s5) + C
    return y
def func6(x, A1, x01, s1, A2, x02, s2, A3, x03, s3, A4, x04, s4, A5, x05, s5, A6, x06, s6, C):
    y = gaussian(x, A1, x01, s1) + gaussian(x, A2, x02, s2) + \
        gaussian(x, A3, x03, s3) + gaussian(x, A4, x04, s4) + \
        gaussian(x, A5, x05, s5) + gaussian(x, A6, x06, s6) + C
    return y
def func7(x, A1, x01, s1, A2, x02, s2, A3, x03, s3, A4, x04, s4, A5, x05, s5, A6, x06, s6, A7, x07, s7, C):
    y = gaussian(x, A1, x01, s1) + gaussian(x, A2, x02, s2) + \
        gaussian(x, A3, x03, s3) + gaussian(x, A4, x04, s4) + \
        gaussian(x, A5, x05, s5) + gaussian(x, A6, x06, s6) + \
        gaussian(x, A7, x07, s7) + C
    return y


def findNearest(arr, val):
    return np.abs(arr - val).argmin()


### load the transmission curve of the two filters as provided by andover.
### if you're not on freia and don't have access to these files, let me know
filter = np.load('/home/sthoma/tokamak_inversion_routines/mastu/filterHSV.npz')
filter_15A = np.load('/home/sthoma/tokamak_inversion_routines/mastu/filterHSV_1.5A.npz')
filter_15B = np.load('/home/sthoma/tokamak_inversion_routines/mastu/filterHSV_1.5B.npz')
filter_andover = np.load('/home/sthoma/tokamak_inversion_routines/mastu/filterHSV_andover.npz')


### example shot number
shotn = 50738
### wl='linear' is the correct calibration
spt, spd, wl = getSpec(shotn, wl='linear')


###########################################
#### the bulk of the work is done here ####
###########################################


### select which to fit to
ch = 4 # which channel
tind = 8 # which time
x = wl[15:]
y = spd[ch,tind,15:]

### fit to dalpha peaks
I0 = findNearest(wl, 655.)
J0 = findNearest(wl, 657.)
x0 = wl[I0:J0]
y0 = spd[ch,tind,I0:J0]
p0 = (6e5, 656.1012, 0.80, 1e5, 656.2819, 0.05, 600.)
low0 = (0, 656., 0.01, 0., 656., 0.01, 500.)
high0 = (7e5, 656.3, 8., 2e5, 656.4, .1, 1000.)
popt0, pcov0 = curve_fit(func2, x0, y0, p0=p0, bounds=(low0,high0))
pcov0 = np.sqrt(np.diag(pcov0))
print(popt0)
# print(pcov0)


I1 = findNearest(wl, 657.4)
J1 = findNearest(wl, 658.8)
x1 = wl[I1:J1]
y1 = spd[ch,tind,I1:J1]
# p1 = (3000., 657.8, 0.20, 1600., 658.3, 0.20, 1600., 658.0, 0.20, 1600., 658.4, 0.20, 600.)
# low1 = (1000., 657.6, 0.05, 1000., 658.1, 0.05, 1000., 657.8, 0.05, 1000., 658.2, 0.05, 400.)
# high1 = (3400., 658.0, 0.40, 2000., 658.5, 0.40, 2000., 658.2, 0.40, 2000., 658.6, 0.40, 800.)
# popt1, pcov1 = curve_fit(func4, x1, y1, p0=p1, bounds=(low1,high1))
p1 = (3000., 657.8, 0.20, 1600., 658.3, 0.20, 600.)
low1 = (400., 657.6, 0.005, 400., 658.1, 0.005, 400.)
high1 = (5000., 658.0, 0.40, 5000., 658.5, 0.40, 800.)
popt1, pcov1 = curve_fit(func2, x1, y1, p0=p1, bounds=(low1,high1))
pcov1 = np.sqrt(np.diag(pcov1))
print(popt1)
# print(pcov1)


# x1 = wl[J0:]
# y1 = spd[ch,tind,J0:]

### some options for plotting
filtercolour = 'k'
filtercolour2 = 'C7'
filterlw = 1.
ymin = 0.
ymax = y0.max() * 1.05
ymax2 = 100.
ymin2 = ymax2/ymax*ymin
carbonlines = [657.80481,658.2876]

### plot it
fig, ax = plt.subplots(1, 1, figsize=(4,3), dpi=150)
ax2 = ax.twinx()
ax.vlines(carbonlines, ymin, ymax, colors='C4', linewidth=0.95, zorder=3)
ax2.plot(filter['wavelength'], filter['transmission'], '-', c='C7', lw=filterlw, zorder=2, label='Current')
ax2.plot(filter_15A['wavelength'], filter_15A['transmission'], '-', c='C6', lw=filterlw, zorder=2, label='1.5A')
ax2.plot(filter_15B['wavelength'], filter_15B['transmission'], '--', c='C8', lw=filterlw, zorder=2, label='1.5B')
ax2.plot(filter_andover['wavelength'], filter_andover['transmission'], '-', c='C9', lw=filterlw, zorder=2, label='andover')
ax2.legend(fancybox=1, framealpha=1, loc='upper right', fontsize=8)

ax.plot(x, y, '-k', zorder=3) # data
### plot the fits indivudually
ax.plot(x, gaussian(x, popt0[0],popt0[1],popt0[2])+popt0[-1], '--', c='C0', zorder=4)
ax.plot(x, gaussian(x, popt0[3],popt0[4],popt0[5])+popt0[-1], '--', c='C1', zorder=4)
ax.plot(x, gaussian(x, popt1[0],popt1[1],popt1[2])+popt1[-1], '--', c='C2', zorder=4)
ax.plot(x, gaussian(x, popt1[3],popt1[4],popt1[5])+popt1[-1], '--', c='C3', zorder=4)
# ax.plot(x, gaussian(x, popt1[6],popt1[7],popt1[8])+popt1[-1], '--', c='C4', zorder=4)
# ax.plot(x, gaussian(x, popt1[9],popt1[10],popt1[11])+popt1[-1], '--', c='C5', zorder=4)
### labels and title
ax.set_xlabel('Wavelength (nm)', fontsize=9)
ax.set_ylabel('Spec (arb)', fontsize=9)
ax2.set_ylabel('Filter transmission (%)', c=filtercolour2, fontsize=9)
ax.set_title('# {}, t={:.6f}s'.format(shotn,spt[tind]), fontsize=10)
### set the limits
# ax.set_xlim(xlim)
# ax.set_xticks(xticks)
ax.set_ylim([ymin, ymax])
ax2.set_ylim([ymin2,ymax2])
### tighten it up and save it
ax.tick_params(
                axis="both", which='both', labelsize=9, direction='in', 
                left=True, bottom=True, right=False, top=True
            )
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax2.tick_params(
                axis="both", which='both', labelsize=9, direction='in', 
                left=False, bottom=False, right=True, top=False
            )
ax2.yaxis.set_minor_locator(AutoMinorLocator())
ax.set_xlim([651.5,661.5])

# ampRatio = popt0[0] / (popt1[0] + popt1[6])
# areaRatio = (popt0[0] * popt0[2]) / ((popt1[0] * popt1[2]) + (popt1[6] * popt1[8]))
ampRatio = popt0[0] / (popt1[0] + popt1[3])
areaRatio = (popt0[0] * popt0[2]) / ((popt1[0] * popt1[2]) + (popt1[3] * popt1[5]))

ax.text(652., y.max()*0.5, 
        f'Ratio of amplitudes\n= {ampRatio:.1f}\nRatio of areas\n= {areaRatio:.1f}', 
        fontsize=9, ha='left', va='center')

plt.tight_layout()
# plt.savefig(plotpath + 'shot_{}_ch_{}_tind_{}.png'.format(shotn,ch,tind))
plt.show()














#
