import HSV
import numpy as np
import MDSplus as mds
from  scipy.linalg import eigh, solve_banded

class LLAMA_tomography():

    def __init__(self, shot, system, regularisation='GCV', time_avg= 0.0005):
        self.shot = shot
        # self.regularisation = regularisation
        ### STh what is regularisation? not needed?
        # self.time_avg = time_avg
        ### STh system seems like it's picking a Ly-a, so perhaps changing this keyword
        ### could be a way of telling it to use HSV?
        # self.system = system # which Ly-a array brightness data is taken from
        self.I0 = 233
        self.I1 = 234
        self.J0 = 50
        self.J1 = 180
        self.tind = 501

    def getRz(self):
        data = np.load('/home/sthoma/inversion-tools/rba_viewing.npz')
        self.R = data['Rmin'][I0:I1,J0:J1]
        self.z = data['zmin'][I0:I1,J0:J1]
        # self.R = np.array([ 1.75388838, 1.68691841, 1.60095721, 1.54012735, 1.49778808,
        #                     1.46703683, 1.44362267, 1.4250236 , 1.40970481, 1.39669763,
        #                     1.38536834, 1.375288  , 1.36615804, 1.35776508, 1.34995296,
        #                     1.34260508, 1.33563268, 1.32896745, 1.32255534, 1.31635381,
        #                     1.31032834, 1.30445085, 1.29869817, 1.29305113, 1.28749363,
        #                     1.28201201, 1.27659465, 1.27123144, 1.26591378, 1.26063418,
        #                     1.25538589, 1.25016313, 1.24496077, 1.23977411, 1.23459892,
        #                     1.22943157, 1.22426866, 1.21910708, 1.21394404, 1.20877694,
        #                     1.20360341, 1.19842126, 1.19322837, 1.18802289, 1.18280309,
        #                     1.17756735, 1.1723141 , 1.16704168, 1.16174894, 1.15643459,
        #                     1.1510972 , 1.14573573, 1.14034915, 1.13493611, 1.12949588,
        #                     1.12402738, 1.11852962, 1.1130019 , 1.10744305, 1.10185263,
        #                     1.09622945, 1.09057314, 1.0848826 , 1.07915747, 1.07339671,
        #                     1.06760005, 1.06176643, 1.05589564, 1.04998676, 1.04403943,
        #                     1.03805303, 1.03202694, 1.02596087, 1.01985396, 1.01370605,
        #                     1.00751659, 1.00128496, 0.99501099, 0.98869411, 0.98233385,
        #                     0.97593001, 0.96948217, 0.96298976, 0.95645269, 0.94987063,
        #                     0.94324312, 0.93656988, 0.92985073, 0.92308542, 0.91627368,
        #                     0.90941508, 0.90250958, 0.89555697, 0.88855707, 0.88150969,
        #                     0.87441467, 0.86727185, 0.86008111, 0.85284227, 0.84555529,
        #                     0.83822011, 0.83083665, 0.82340487, 0.81592473, 0.80839616,
        #                     0.80081919, 0.79319384, 0.78552017, 0.77779823, 0.7700281 ,
        #                     0.76220987, 0.75434348, 0.74642912, 0.73846705, 0.73045745,
        #                     0.72240008, 0.71429556, 0.70614403, 0.69794539, 0.68970036,
        #                     0.68140867, 0.67307111, 0.66468744, 0.65625853, 0.6477841 ,
        #                     0.63926496, 0.63070142, 0.62209344, 0.61344184, 0.60474698])
        # self.z = np.array([ -0.00381096, -0.0062133 , -0.00689943, -0.00699856, -0.00692772,
        #                     -0.00680384, -0.00667332, -0.00654492, -0.00642565, -0.00631666,
        #                     -0.00621481, -0.00611793, -0.00603054, -0.00594687, -0.00586964,
        #                     -0.00579415, -0.00572498, -0.0056563 , -0.00559255, -0.00553005,
        #                     -0.00546998, -0.00541063, -0.00535449, -0.00529862, -0.00524418,
        #                     -0.00519097, -0.00513885, -0.0050866 , -0.00503628, -0.00498571,
        #                     -0.00493677, -0.0048875 , -0.00483878, -0.00479134, -0.0047435 ,
        #                     -0.00469604, -0.00464892, -0.00460208, -0.00455551, -0.00450915,
        #                     -0.00446299, -0.00441699, -0.00437068, -0.00432497, -0.00427936,
        #                     -0.00423382, -0.00418805, -0.00414266, -0.00409729, -0.00405194,
        #                     -0.00400645, -0.00396114, -0.00391582, -0.00387042, -0.00382508,
        #                     -0.00377972, -0.00373434, -0.00368899, -0.00364354, -0.00359803,
        #                     -0.00355264, -0.00350704, -0.00346162, -0.00341592, -0.00337046,
        #                     -0.00332499, -0.00327914, -0.00323362, -0.00318765, -0.00314208,
        #                     -0.003096  , -0.00305038, -0.00300475, -0.00295852, -0.00291285,
        #                     -0.00286652, -0.00282081, -0.0027751 , -0.00272865, -0.00268291,
        #                     -0.00263719, -0.00259065, -0.00254492, -0.00249921, -0.00245353,
        #                     -0.00240692, -0.00236126, -0.00231564, -0.00227006, -0.00222452,
        #                     -0.00217795, -0.00213248, -0.00208708, -0.00204174, -0.00199648,
        #                     -0.00195129, -0.00190618, -0.00186116, -0.00181494, -0.00177009,
        #                     -0.00172535, -0.00168073, -0.00163622, -0.00159185, -0.00154905,
        #                     -0.00150498, -0.00146105, -0.00141728, -0.00137367, -0.00133024,
        #                     -0.00128698, -0.00124552, -0.00120267, -0.00116001, -0.00111757,
        #                     -0.00107705, -0.00103507, -0.00099507, -0.00095359, -0.00091235,
        #                     -0.00087319, -0.0008325 , -0.00079394, -0.0007557 , -0.00071589,
        #                     -0.0006783 , -0.00063912, -0.00060221, -0.00056566, -0.00052948])

    def load_geometry(self,r_end=False,sys_err=5):

#        node = OMFITmdsValue(server='CMOD',shot=self.shot,treename='SPECTROSCOPY',
#            TDI='\\SPECTROSCOPY::TOP.BOLOMETER.RESULTS.DIODE.'+\
#            '{:s}:BRIGHT'.format(self.system))

        ### STh I think this is specific for C-Mod data?
        # node = mds.Tree('spectroscopy', self.shot)
        # node = node.getNode('\\spectroscopy::top.bolometer.results.diode.'+\
        #     '{:s}:BRIGHT'.format(self.system))

        # check which channels are empty

        ### STh hard to know what bright is gonna look like if 
        ### I can't load the file types
        # bright = node.data()
        # bright = np.loadtxt('/home/sthoma/inversion-tools/Example/shot_1070511010_brightness_data.txt')
        frame, time0 = HSV.getSingle(self.shot, self.tind)
        bright = frame[self.I0:self.I1,self.J0:self.J1] # goes at approx midplane up to 180
        time = np.array([time0])
        
        ### STh so this is just the brightness I think?
        # self.good_chans = np.where(bright[0] != 0)[0]

        ### STh could be the LCFS on LFS?
        ### or it's the radial locations? Pretty sure it's this one.
#        lfs_r = node.dim_of(0)[self.good_chans]
        # R = np.loadtxt('/home/sthoma/inversion-tools/Example/shot_1070511010_R_values.txt')
        # lfs_r = node.dim_of(0).data()[self.good_chans] # have to do it this way if no OMFITmdsValue
        ### use self.R instead
        self.getRz()
        # lfs_r = R[self.good_chans] # have to do it this way if no OMFITmdsValue
        lfs_r = self.R
        lfs_r = np.flip(lfs_r) # stored as decreasing in CMOD tree
        ### flip makes it now increasing

        
        if r_end:
            lfs_r = np.insert(lfs_r,len(lfs_r),r_end) # want to insert a 0 at r_end
            ### STh inserts it at the end

        # LFS_weights = np.ones(len(lfs_r)) # constant weights

        self.nch_lfs = len(lfs_r) # number of channels

        ### ignore claibration for now

        self.calf = np.ones(lfs_r.shape)
        self.calfErr = np.ones(lfs_r.shape)*sys_err/100

        ## create response matrix

        R_tg_virtual = lfs_r

        #center of mass of the LOS
        self.R_tg = R_tg_virtual
        self.Z_tg = np.zeros_like(self.R_tg) # assume at midplane (z = 0)

        self.lfs_min = self.R_tg[0] # first r value
        self.lfs_max = self.R_tg[-1] # last r value

        # self.nr = 50 # gives resolution of emissivity grid
        self.nr = 325 # still 2.5 times
        #self.R_grid = np.linspace(self.lfs_min-.01,self.lfs_max+.01,self.nr)
        self.R_grid = np.linspace(self.lfs_min,self.lfs_max,self.nr)

        dL = 2*(np.sqrt(np.maximum((self.R_grid[1:])**2-R_tg_virtual[:,None]**2,0))
               -np.sqrt(np.maximum( self.R_grid[:-1]**2-R_tg_virtual[:,None]**2,0)))
        self.dL = dL # no need to sum over spot size


    #Simplest data load
    #smooths data for an entire shot
    def load_data(self,r_end=False):

#        node = OMFITmdsValue(server='CMOD',shot=self.shot,treename='SPECTROSCOPY',
#            TDI='\\SPECTROSCOPY::TOP.BOLOMETER.RESULTS.DIODE.'+\
#            '{:s}:BRIGHT'.format(self.system))

        # node = mds.Tree('spectroscopy', self.shot)
        # node = node.getNode('\\spectroscopy::top.bolometer.results.diode.'+\
        #     '{:s}:BRIGHT'.format(self.system))

        # raw_data = node.data()
        # raw_data = np.loadtxt('/home/sthoma/inversion-tools/Example/shot_1070511010_brightness_data.txt')
        # raw_data = raw_data[:,self.good_chans]
        # raw_data = np.flip(raw_data, axis=1)
        #raw_data = np.flip(raw_data)
        frame, time0 = HSV.getSingle(self.shot, self.tind)
        raw_data = np.flip(frame[self.I0:self.I1,self.J0:self.J1], axis=1) # goes at approx midplane up to 180

        # add a zero at desired r_end value (set in load_geometry)
        if r_end:
            _zeros = np.zeros(len(raw_data[:,0]))[:,None]
            raw_data = np.concatenate((raw_data,_zeros),axis=1)

        n_los = len(raw_data[0])

#        tvec = node.dim_of(1)
        # tvec = node.dim_of(1).data()
        # tvec = np.loadtxt('/home/sthoma/inversion-tools/Example/shot_1070511010_t_values.txt')
        tvec = np.array([time0])

        last_good_ind = 0
        # option to offset data if it reads negative values
        for tt in range(len(tvec)):

            min_data = raw_data[tt,-2] # because there's a 0 by default at the end

            ### option 1 ###
            '''
            if min_data < 0:
                raw_data[tt,:-1] -= min_data
            '''
            if min_data >= 0:
                last_good_ind = tt
            elif min_data < 0:
                raw_data[tt,:-1] += np.max(raw_data[last_good_ind])

        offset = slice(0, tvec.searchsorted(0))
        dt = (tvec[-1]-tvec[0])/(len(tvec)-1)

        n_smooth = 1 # no time smoothing - already done for brightness data

        nt,nch = raw_data.shape

        data_low = raw_data
        tvec_low = tvec

        nt = nt//n_smooth*n_smooth

        tvec_low = tvec[:nt].reshape(-1,n_smooth).mean(1)

        data_low = raw_data[:nt].reshape(-1,n_smooth, nch).mean(1)-raw_data[offset].mean(0)

        # estimate noise from the signal before the plasma
        error_low1 = np.zeros_like(data_low)
        error_low2 = np.std(data_low[tvec_low<0],0)
        error_low21 = np.std(data_low[tvec_low<0],0)[None,:]
        error_low = np.zeros_like(data_low)+np.std(data_low[tvec_low<0],0)[None,:]/3


        # guess errorbarss from the variation between neighboring channels
        ind1 = np.r_[1,0:n_los-1]
        ind2 = np.r_[  1:n_los  ,n_los-2]

        # the ind1 and ind2, basically shift data_low up and down by one to find the average
        # value of the neighboring channels. That is then subtracting from the original array
        # to get the average difference from neighboring hcannels

        # the difference between the followin neighbor is calculated and then the standard error
        # is calculated for each channel
        error_low += np.std(np.diff(data_low-(data_low[:,ind1]+data_low[:,ind2])/2,axis=0),axis=0)/np.sqrt(2)

        # remove offset estimated from the edge most detector
        offset_time = data_low[:,[-1]]
        data_low -= offset_time
        # make sure that zero value is within errorbars when data are negative
        error_low = np.maximum(error_low, -data_low)

        self.data = data_low *self.calf#[ph/m^2s]
        self.err  = np.sqrt(\
                    (error_low*self.calf)**2+\
                    (data_low*self.calfErr)**2
                   ) #[ph/m^2s] # laggnerf
        self.tvec = tvec_low #[s]
        self.scale = np.median(self.data) #just a normalisation to aviod calculation with so huge exponents

        self.nt = len(self.tvec)


    def regul_matrix(self, biased_edges = True):
        #regularization band matrix

        bias = .1 if biased_edges else 1e-5
        D = np.ones((3,self.nr-1))
        D[1,:] *= -2
        D[1,-1] = bias
        D[1,[0,self.nr-3]] = -1
        D[2,[-2,-3]] = 0

        #D = inv(solve_banded((1,1),D, eye( self.nr-1)))
        #imshow(D, interpolation='nearest',   cmap='seismic');show()
        return D

    def PRESS(self,g, prod,S,U):
        #predictive sum of squares
        w = 1./(1.+np.exp(g)/S**2)
        ndets = len(prod)
        return np.sum((np.dot(U, (1-w)*prod)/np.einsum('ij,ij,j->i', U,U, 1-w))**2)/ndets


    def GCV(self,g, prod,S,U):
        #generalized crossvalidation
        w = 1./(1.+np.exp(g)/S**2)
        ndets = len(prod)
        return (np.sum((((w-1)*prod))**2)+1)/ndets/(1-np.mean(w))**2


    def FindMin(self,F, x0,dx0,prod,S,U,tol=0.01):
        #stupid but robust minimum searching algorithm.

        fg = F(x0, prod, S,U)
        while abs(dx0) > tol:
            fg2 = F(x0+dx0, prod,S,U)

            if fg2 < fg:
                fg = fg2
                x0 += dx0
                continue
            else:
                dx0/=-2.

        return x0, np.log(fg2)


    def calc_tomo(self, n_blocks = 10):
        #calculate tomography of data splitted in n_blocks using optimised minimum fisher regularisation
        #Odstrcil, T., et al. "Optimized tomography methods for plasma
        #emissivity reconstruction at the ASDEX  Upgrade tokamak.
        #" Review of Scientific Instruments 87.12 (2016): 123505.

        #defined independently for LFS and HFS
        reg_level_guess = .7, .6
        reg_level_min = .4, .4

        nfisher = 4

        #prepare regularisation operator
        D = self.regul_matrix(biased_edges=True)


        self.y = np.zeros((self.nt, self.nr-1))
        self.y_err = np.zeros((self.nt, self.nr-1))
        self.chi2lfs = np.zeros(self.nt)
        self.gamma_lfs = np.zeros(self.nt)
        self.backprojection = np.zeros_like(self.data)

        itime = np.arange(self.nt)
        tinds = np.array_split(itime, n_blocks)

        for ib, tind in enumerate(tinds):

            ## cmod mod: see where the error is 0 and replace to avoid dividing by 0
            mean_err_zero_inds = np.where(self.err[tind].mean(0) == 0)
            err_zero_inds = np.where(self.err[tind] == 0)

            T = self.dL/self.err[tind].mean(0)[:,None]*self.scale
            mean_d = self.data[tind].mean(0)/self.err[tind].mean(0)
            d = self.data[tind]/self.err[tind]

            ## replace infinities and nans with 0
            T[mean_err_zero_inds] = 0
            mean_d[mean_err_zero_inds] = 0
            d[err_zero_inds] = 0

            iside = 0
            side = 'LFS'

            W = np.ones(self.nr-1)

            ind_los = slice(0,self.nch_lfs)
            ind_space = slice(0,self.nr-1)
            lfs_contribution = [0]

            Q = np.linspace(0,1,ind_los.stop-ind_los.start)

            for ifisher in range(nfisher):
                #multiply tridiagonal regularisation operator by a diagonal weight matrix W
                WD = np.copy(D)

                WD[0,1:]*=W[:-1]
                WD[1]*=W
                WD[2,:-1]*=W[1:]

                #transpose the band matrix
                DTW = np.copy(WD)
                DTW[0,1:],DTW[2,:-1] = WD[2,:-1],WD[0,1:]

                #####    solve Tikhonov regularization (optimised for speed)
                H = solve_banded((1,1),DTW,T[ind_los,ind_space].T, overwrite_ab=True,check_finite=False)
                #fast method to calculate U,S,V = svd(H.T) of rectangular matrix
                LL = np.dot(H.T, H)
                S2,U = eigh(LL,overwrite_a=True, check_finite=False,lower=True)
                S2 = np.maximum(S2,1) #singular values S can be negative due to numerical uncertainty

                mean_p = np.dot(mean_d[ind_los]-np.mean(lfs_contribution,0),U)

                #guess for regularisation - estimate quantile of log(S^2)
                g0 = np.interp(reg_level_guess[iside], Q, np.log(S2))

                if ifisher == nfisher -1:
                    #last step - find optimal regularisation
                    S = np.sqrt(S2)

                    g0, log_fg2 = self.FindMin(self.GCV, g0 ,1,mean_p,S,U.T) #slowest step
                    #avoid too small regularisation when min of GCV is not found

                    gmin = np.interp(reg_level_min[iside], Q, np.log(S2))
                    g0 = max(g0, gmin)

                    #filtering factor
                    w = 1./(1.+np.exp(g0)/S2)

                    V = np.dot(H,U/S)
                    V = solve_banded((1,1),WD,V, overwrite_ab=True,overwrite_b=True,check_finite=False)
                else:
                    #filtering factor
                    w = 1./(1.+np.exp(g0)/S2)

                    #calculate y without evaluating V explicitly
                    y = np.dot(H,np.dot(U/S2,w*mean_p))
                    #final inversion of mean solution , reconstruction
                    y = solve_banded((1,1),WD,y, overwrite_ab=True,overwrite_b=True,check_finite=False)

                    #plt.plot(y)
                    #weight matrix for the next iteration
                    W = 1/np.maximum(y,1e-10)**.5

            p = np.dot(d[:,ind_los]-lfs_contribution,U)
            y = np.dot((w/S)*p,V.T)

            self.backprojection[tind,ind_los] = fit = np.dot(p*w,U.T)+lfs_contribution
            chi2 = np.sum((d[:,ind_los]-fit)**2,1)/np.size(fit,1)
            gamma = np.interp(g0,np.log(S2),Q)

            self.chi2lfs[tind] = chi2
            self.gamma_lfs[tind] = gamma

            self.y[tind,ind_space] = y
            #correction for under/over estimated data uncertainty
            self.y_err[tind,ind_space] = np.sqrt(np.dot(V**2,(w/S)**2))#*chi2[:,None])

        self.backprojection[tind] *= self.err[tind].mean(0)

        self.y *= self.scale
        self.y_err *= self.scale

        self.R_grid_b = (self.R_grid[1:]+ self.R_grid[:-1])/2

        return self.R_grid_b, self.y,self.y_err, self.backprojection


# def tomoCMOD(shot,system,tWindow=False,r_end=0.93,sys_err=5,n_blocks=None):
def tomoCMOD(shot,system,tWindow=False,r_end=1.80,sys_err=5,n_blocks=None):

    print('Inversion has zero at '+str(r_end) + ' m')
    tomo = LLAMA_tomography(shot,system,time_avg=0) # cmod brightness data already smoothed in time
    tomo.load_geometry(r_end=r_end,sys_err=sys_err)
    tomo.load_data(r_end=r_end)
    if n_blocks == None:
        n_blocks = len(tomo.tvec)
    tomo.calc_tomo(n_blocks=n_blocks)

    return tomo.tvec,tomo.R_grid_b,tomo.y,tomo.y_err,tomo.backprojection
