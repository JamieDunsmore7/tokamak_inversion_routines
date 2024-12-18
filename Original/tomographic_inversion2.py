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
        print('end of init')


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
        bright = np.loadtxt('/home/sthoma/inversion-tools/Example/shot_1070511010_brightness_data.txt')
        ### STh so this is just the brightness I think?
        self.good_chans = np.where(bright[0] != 0)[0]

        ### STh could be the LCFS on LFS?
        ### or it's the radial locations? Pretty sure it's this one.
#        lfs_r = node.dim_of(0)[self.good_chans]
        R = np.loadtxt('/home/sthoma/inversion-tools/Example/shot_1070511010_R_values.txt')
        # lfs_r = node.dim_of(0).data()[self.good_chans] # have to do it this way if no OMFITmdsValue
        lfs_r = R[self.good_chans] # have to do it this way if no OMFITmdsValue
        lfs_r = np.flip(lfs_r) # stored as decreasing in CMOD tree

        
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

        self.nr = 50 # gives resolution of emissivity grid
        #self.R_grid = np.linspace(self.lfs_min-.01,self.lfs_max+.01,self.nr)
        self.R_grid = np.linspace(self.lfs_min,self.lfs_max,self.nr)

        dL = 2*(np.sqrt(np.maximum((self.R_grid[1:])**2-R_tg_virtual[:,None]**2,0))
               -np.sqrt(np.maximum( self.R_grid[:-1]**2-R_tg_virtual[:,None]**2,0)))
        self.dL = dL # no need to sum over spot size
        print('end of load_geometry')


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
        raw_data = np.loadtxt('/home/sthoma/inversion-tools/Example/shot_1070511010_brightness_data.txt')
        raw_data = raw_data[:,self.good_chans]
        raw_data = np.flip(raw_data, axis=1)
        #raw_data = np.flip(raw_data)

        # add a zero at desired r_end value (set in load_geometry)
        if r_end:
            _zeros = np.zeros(len(raw_data[:,0]))[:,None]
            raw_data = np.concatenate((raw_data,_zeros),axis=1)

        n_los = len(raw_data[0])

#        tvec = node.dim_of(1)
        # tvec = node.dim_of(1).data()
        tvec = np.loadtxt('/home/sthoma/inversion-tools/Example/shot_1070511010_t_values.txt')

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
        print('end of load_data')


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


def tomoCMOD(shot,system,tWindow=False,r_end=0.93,sys_err=5,n_blocks=None):

    print('Inversion has zero at '+str(r_end) + ' m')
    tomo = LLAMA_tomography(shot,system,time_avg=0) # cmod brightness data already smoothed in time
    tomo.load_geometry(r_end=r_end,sys_err=sys_err)
    tomo.load_data(r_end=r_end)
    if n_blocks == None:
        n_blocks = len(tomo.tvec)
    tomo.calc_tomo(n_blocks=n_blocks)

    return tomo.tvec,tomo.R_grid_b,tomo.y,tomo.y_err,tomo.backprojection
