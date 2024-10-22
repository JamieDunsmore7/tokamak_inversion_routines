import numpy as np
import matplotlib.pylab as plt
from IPython import embed
import MDSplus as mds
from scipy.signal import savgol_filter

from  scipy.linalg import eigh, solve_banded

#from omfit_classes.omfit_mds import OMFITmdsValue


def update_fill_between(fill,x,y_low,y_up,min,max ):
    paths, = fill.get_paths()
    nx = len(x)
    
    y_low = np.maximum(y_low, min)
    y_low[y_low==max] = min
    y_up = np.minimum(y_up,max)
    
    vertices = paths.vertices.T
    vertices[:,1:nx+1] = x,y_up
    vertices[:,nx+1] =  x[-1],y_up[-1]
    vertices[:,nx+2:-1] = x[::-1],y_low[::-1]
    vertices[:, 0] = x[0],y_up[0]
    vertices[:,-1] = x[0],y_up[0]
    
def update_errorbar(err_plot, x,y,yerr):
    
    plotline, caplines, barlinecols = err_plot

    # Replot the data first
    plotline.set_data(x,y)

    # Find the ending points of the errorbars
    error_positions = (x,y-yerr), (x,y+yerr)

    # Update the caplines
    if len(caplines) > 0:
        for j,pos in enumerate(error_positions):
            caplines[j].set_data(pos)

    # Update the error bars
    barlinecols[0].set_segments(list(zip(list(zip(x,y-yerr)), list(zip(x,y+yerr))))) 


class LLAMA_tomography():

    def __init__(self, shot, system, regularisation='GCV'):
        self.shot = shot
        self.regularisation = regularisation
        self.system = system # which Ly-a array brightness data is taken from

        
    def load_geometry(self,r_end=None):
        
#        node = OMFITmdsValue(server='CMOD',shot=self.shot,treename='SPECTROSCOPY',
#            TDI='\\SPECTROSCOPY::TOP.BOLOMETER.RESULTS.DIODE.'+\
#            '{:s}:BRIGHT'.format(self.system))
    
        node = mds.Tree('spectroscopy', self.shot)
        node = node.getNode(f'\\spectroscopy::top.bolometer.results.diode.{self.system}:BRIGHT')
        bright = node.data()
        lfs_r = node.dim_of(0).data()
  
            
        # check which channels are empty

        self.good_chans = bright[0] != 0

        lfs_r = lfs_r[self.good_chans] # have to do it this way if no OMFITmdsValue
        lfs_r = lfs_r[::-1] # stored as decreasing in CMOD tree
  
        self.nch_lfs = len(lfs_r)


        ## create response matrix
        #center of mass of the LOS
        self.R_tg = lfs_r
        self.Z_tg = np.zeros_like(self.R_tg) # assume at midplane (z = 0)

        self.lfs_min = self.R_tg[0] # first r value
        self.lfs_max = self.R_tg[-1] # last r value
        
        if r_end is not None:
            self.lfs_max = r_end
       
        self.nr = 100 # gives resolution of emissivity grid
        self.R_grid = np.linspace(self.lfs_min,self.lfs_max,self.nr)
        self.R_grid_b = (self.R_grid[1:]+ self.R_grid[:-1])/2


        # self.R_grid is an evenly spaced grid for the inversion to be performed on, starting at the minimum value of R_tg and ending at the specified inversion zero, r_end.
        # R_tg are the input tangency radii. R_grid and R_tg are both 1D arrays.
        # dL gives the distance between all R_grid points ALONG THE LINE-OF-SIGHT for each chord.
        self.dL = 2*(np.sqrt(np.maximum((self.R_grid[1:])**2-self.R_tg[:,None]**2,0))       
                    -np.sqrt(np.maximum( self.R_grid[:-1]**2-self.R_tg[:,None]**2,0)))
                
        #evaluate back projection for the every point of the grid
        self.dL_grid = 2*(np.sqrt(np.maximum((self.R_grid[1:])**2-self.R_grid_b[:,None]**2,0))       
                         -np.sqrt(np.maximum( self.R_grid[:-1]**2-self.R_grid_b[:,None]**2,0)))


    def calibration(self, sys_err=5):
        ### ignore calibration for now
        self.calf = np.ones(self.nch_lfs) #it is stored in W/m^2 in MDS+??
        #5% relative calibration error
        self.calfErr = np.ones(self.nch_lfs)*sys_err/100
 
    #smooths data for an entire shot
    def load_data(self, calib_factor=1, apply_offsets=False, t_window = None, smooth_brightness_in_time = False):

#        node = OMFITmdsValue(server='CMOD',shot=self.shot,treename='SPECTROSCOPY',
#            TDI='\\SPECTROSCOPY::TOP.BOLOMETER.RESULTS.DIODE.'+\
#            '{:s}:BRIGHT'.format(self.system))
        
        node = mds.Tree('spectroscopy', self.shot)
        node = node.getNode(f'\\spectroscopy::top.bolometer.results.diode.{self.system}:BRIGHT')
    
        raw_data = node.data()
        tvec = node.dim_of(1).data()

        if apply_offsets == True:
            if self.system == 'WB4LY':
                # NOTE: check actual signals to confirm the expected behaviour for channel 7.
                offset_channel_values = raw_data[:,7] #this channel is dead on WB4LY in 2009 (only shows dark trace) we can subtract its reading from the other channels to remove noise.
                raw_data = raw_data - offset_channel_values[:, np.newaxis]

            elif self.system == 'LYMID':
                # NOTE: this should be used WITH EXTREME CAUTION. Don't know which shots this should apply to

                #assume that the ratio of channel 3 to channel 1 should be fixed throughout the shot
                #essentially we are trading the noise in channels 3,4,5 etc. for the noise in channel 1 (which is much reduced compared to the others.)
                offset_channel_values = raw_data[:,0]
                average_delta = np.mean(raw_data[:,3]) / np.mean(raw_data[:,1]) 
                channel_offset = raw_data[:,3] - (raw_data[:,1] * average_delta) #resulting offset
                array_2d = np.repeat(channel_offset[:, np.newaxis], len(raw_data[0]), axis=1)
                exclude_columns = [0, 1, 2, 8, 10, 15, 19]  # don't apply to dead channels and to the channels with different noise (1 and 10)

                array_2d[:,exclude_columns] = 0
                raw_data = raw_data - array_2d
            else:
                print('Offsets not applied')
                
       
        raw_data = raw_data[:,self.good_chans]
        raw_data = raw_data[:,::-1] # is this the same as np.flip(raw_data, axis=1)?
        raw_data = raw_data * calib_factor #apply calibration factor to adjust for diode degradation, if known.






        offset = tvec.searchsorted(0)
        dt = (tvec[-1]-tvec[0])/(len(tvec)-1)

        n_smooth = 1 # no time smoothing - already done for brightness data

        nt,nch = raw_data.shape

        data_low = raw_data
        tvec_low = tvec

        nt = nt//n_smooth*n_smooth
        
        tvec_low = tvec[:nt].reshape(-1,n_smooth).mean(1)
        data_low = raw_data[:nt].reshape(-1,n_smooth, nch).mean(1)
        
        #remove offset before t = 0
        data_low -= data_low[:offset].mean(0)

        # estimate noise from the signal before the plasma+ calibration error
        error_low = np.hypot(self.calfErr * data_low, np.std(data_low[:offset],0)) 


        #add offset to all channels such that the outermost is above zero
        # last_good_ind = 0        
        # option to offset data if it reads negative values
        # for tt in range(len(tvec)):
        #     if data_low[tt,-1] >= 0:
        #         last_good_ind = tt
        #     else:
        #         #NOTE why is there max??
        #         data_low[tt] += np.max(data_low[last_good_ind])

        
        
        # make sure that zero value is within errorbars when data are negative
        error_low = np.maximum(error_low, -data_low)

        ## cmod mod: see where the error is 0 and replace to avoid dividing by 0
        error_low[error_low == 0] = np.inf
        
        self.data = data_low * self.calf#[ph/m^2s]
        #TODO add calibration error later!
        self.err = error_low*self.calf #[ph/m^2s]
        self.tvec = tvec_low #[s]

        # only invert the data within the time window
        if t_window is not None:
            t_mask = (self.tvec > t_window[0]) & (self.tvec < t_window[1])
            self.tvec = self.tvec[t_mask]
            self.data = self.data[t_mask]
            self.err = self.err[t_mask]
            raw_data = raw_data[t_mask]


        # cycle through every radial location and smooth in time with a savgol filter.
        if smooth_brightness_in_time == True:
            for idx in range(len(self.data[0])-1):
                self.data[:,idx] = savgol_filter(self.data[:,idx], 19, 1) #Default is linear smoothing with window length of 19 elements.

        self.scale = np.median(self.data) #just a normalisation to aviod calculation with so huge exponents
        self.nt = nt


    def regul_matrix(self, biased_edges = True):
        #regularization band matrix

 
        def triband2dense(A):
            Afull = np.diag(A[0,1:],k=-1)+np.diag(A[1],k=0)+np.diag(A[2,:-1],k=1)
            return Afull

#         
        bias = .1 if biased_edges else 1e-5
        D = np.ones((3,self.nr-1))
        D[1,:] *= -2
        D[1,-1] =  bias
        D[1,[0,self.nr-3]] = -1
        D[2,[-2,-3]] = 0
   
 
        # plt.imshow(triband2dense(D).T, interpolation='nearest',   cmap='seismic')
        # plt.show()
 
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


    def calc_tomo(self, n_blocks = 20, reg_value = 0):
        #calculate tomography of data splitted in n_blocks using optimised minimum fisher regularisation
        #Odstrcil, T., et al. "Optimized tomography methods for plasma 
        #emissivity reconstruction at the ASDEX  Upgrade tokamak.
        #" Review of Scientific Instruments 87.12 (2016): 123505.
        
        self.n_blocks = n_blocks
     
        reg_level_guess = .7
        reg_level_min = .5
        nfisher = 4
        
        #prepare regularisation operator, bias right side of grid to zero
        D = self.regul_matrix(biased_edges=True)
        
        
        self.y = np.zeros((self.nt, self.nr-1))
        self.y_err = np.zeros((self.nt, self.nr-1))
        self.chi2 = np.zeros(self.nt)
        self.gamma = np.zeros(self.nt) 
        self.backprojection = np.zeros_like(self.data)

        itime = np.arange(self.nt)
        tinds = np.array_split(itime, n_blocks)

        for ib, tind in enumerate(tinds):

            T = self.dL/self.err[tind].mean(0)[:,None]*self.scale
            mean_d = self.data[tind].mean(0)/self.err[tind].mean(0)
            d = self.data[tind]/self.err[tind]
  

            W = np.ones(self.nr-1)
 
            Q = np.linspace(0,1,self.nch_lfs)
    
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
                H = solve_banded((1,1),DTW,T.T, overwrite_ab=True,check_finite=False)
                #fast method to calculate U,S,V = svd(H.T) of rectangular matrix 
                LL = np.dot(H.T, H)
                S2,U = eigh(LL,overwrite_a=True, check_finite=False,lower=True)  
                S2 = np.maximum(S2,1) #singular values S can be negative due to numerical uncertainty                 

                mean_p = np.dot(mean_d,U)
                
                #guess for regularisation - estimate quantile of log(S^2)
                g0 = np.interp(reg_level_guess, Q, np.log(S2))

                if ifisher == nfisher -1:
                    #last step - find optimal regularisation
                    S = np.sqrt(S2)
                    
                    if reg_value == 0:
                        g0, log_fg2 = self.FindMin(self.GCV, g0 ,1,mean_p,S,U.T) #slowest step
                        #avoid too small regularisation when min of GCV is not found
                        
                        gmin = np.interp(reg_level_min, Q, np.log(S2))
                        g0 = max(g0, gmin)
                    else:
                        g0 = np.interp(reg_value, Q, np.log(S2))
                        
                    
                    #filtering factor
                    w = 1./(1.+np.exp(g0)/S2)
                    
                else:
                    #filtering factor
                    w = 1./(1.+np.exp(g0)/S2)
                    
                    #calculate y without evaluating V explicitly
                    y = np.dot(H,np.dot(U/S2,w*mean_p))
                    #final inversion of mean solution , reconstruction
                    y = solve_banded((1,1),WD,y, overwrite_ab=True,overwrite_b=True,check_finite=False) 
   
                    #weight matrix for the next iteration
                    W = 1/np.maximum(y,1e-10)**.5


            V = np.dot(H,U/S)  
            V = solve_banded((1,1),WD,V, overwrite_ab=True,overwrite_b=True,check_finite=False)   
            
            p = np.dot(d,U)
            y = np.dot((w/S)*p,V.T)
    
            self.backprojection[tind] = np.dot(p*w,U.T)
            chi2 = np.mean((d-self.backprojection[tind])**2,1)
            gamma = np.interp(g0,np.log(S2),Q)

            self.chi2[tind] = chi2
            self.gamma[tind] = gamma
 
            self.y[tind] = y
            #correction for under/over estimated data uncertainty
            #TODO check this!!!!
            # embed()
            self.y_err[tind] = np.sqrt(np.dot(V**2,(w/S)**2))#*chi2[:,None])
            
           
            #here was the error
            self.backprojection[tind] *= self.err[tind].mean(0)
            
        #backprojection evaluated on self.R_grid
        self.y *= self.scale
        self.y_err *= self.scale
        self.backprojection_grid = np.dot(self.y,self.dL_grid.T)


    
    def show_reconstruction(self):
        from matplotlib.widgets import Slider, MultiCursor,Button
                
        f,ax = plt.subplots(1,2, sharex='col', figsize=(9,6))
        ax_time = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor='y')
        slide_time = Slider(ax_time, 'Time:', max(self.tvec[0],0), self.tvec[-1], valinit=0, valstep=0.001, valfmt='%1.4fs')

        ax_reg = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='y')        
        slide_reg = Slider(ax_reg, 'Regularisation:', 0, 1, valinit=0, valstep=0.001, valfmt='%1.3f')
                


        #f.subplots_adjust(bottom=.2)
        f.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.25, hspace=0.3, wspace=0.4)


        r = self.R_grid_b

        confidence = ax[1].fill_between(r, r*0, r*0, alpha=.5, facecolor='b', edgecolor='None')
        tomo_mean, = ax[1].plot([],[], lw=2)
        

        errorbar = ax[0].errorbar(0,np.nan,0,  capsize = 4
                                        ,c='g',marker='o',fillstyle='none',ls='none')
        retro , = ax[0].plot([],[],'b-')
        retro2, = ax[0].plot([],[],'bx')

        
        ax[1].axvline(self.lfs_min,c='k',ls='--')
        ax[1].axvline(self.lfs_max,c='k',ls='--')

        
        ax[0].axhline(0,c='k')
        ax[1].axhline(0,c='k')

        ax[0].grid(linestyle='--', alpha=0.3)
        ax[1].grid(linestyle='--', alpha=0.3)


        self.multi = MultiCursor(f.canvas, ax.flatten(), color='c', lw=1)
        ax[0].set_xlim(self.lfs_min-.01, self.lfs_max+.01)

        ax[1].set_ylim(0, np.percentile(self.y.max(1),95)*1.1)
        ax[0].set_ylim(0, np.percentile((self.data).max(1),95)*1.1)
        ax[1].set_xlabel('R [m]')
        ax[0].set_xlabel('$R_{tang}$ [m]')
        ax[1].set_ylabel('Emissivity [W/m$^3$]')
        ax[0].set_ylabel('Brightness [W/m$^2$]')
        
        title = f.suptitle('')


        def update(val):
            it = np.argmin(np.abs(self.tvec-val))
            update_fill_between(confidence,r,self.y[it]-self.y_err[it],self.y[it]+self.y_err[it],-np.inf,np.inf)
            
            tomo_mean.set_data(r,self.y[it])

            update_errorbar(errorbar,self.R_tg, self.data[it], self.err[it])
            
            retro.set_data(r, self.backprojection_grid[it])
            retro2.set_data(self.R_tg, self.backprojection[it])
            
            title.set_text(rf'#{self.shot},  {self.system}, {self.tvec[it]:.3f}s, $\chi_{{reduced}}^2$ = { self.chi2[it]:.2f}  $\gamma$ = {self.gamma[it] :.2f}')
   
            f.canvas.draw_idle()


        def on_key(event):
            dt = (self.tvec[-1]-self.tvec[0])/(len(self.tvec)-1)
            tnew = slide_time.val
            
            if hasattr(event,'step'):
                #scroll_event
                tnew += event.step*dt

            elif 'left' == event.key:
                #key_press_event
                tnew -= dt
                    
            elif 'right' == event.key:
                tnew += dt
                
            tnew = min(max(tnew,self.tvec[0]),self.tvec[-1])
            slide_time.set_val(tnew)
            update(tnew)
        
        def update_reg(val):
            self.calc_tomo( n_blocks = self.n_blocks, reg_value=val)
            update(slide_time.val)

            
            

        self.cid = f.canvas.mpl_connect('key_press_event',   on_key)
        self.cid_scroll = f.canvas.mpl_connect('scroll_event',on_key)


        slide_time.on_changed(update)
        slide_reg.on_changed(update_reg)


        update(0)

        plt.show()


def tomoCMOD(shot,system, r_end=0.93, sys_err=5, reg = 0, calib_factor=1, apply_offsets=None, t_window = None, smooth_brightness_in_time = False, n_blocks=None, show_reconstruction=False):
    '''
    shot (int): CMOD shot number
    system (str): 'LYMID' or 'WB1LY' or 'WB4LY'
    r_end (float): the inversion needs a zero. This should be an R (major radius) value where we would expect the emission to be zero (e.g at the wall).
    sys_err (float): systematic error on raw brightness points in percent. Default is 5%
    reg (float): regularisation value for the inversion (between 0 and 1). This determines how smooth the inversion is. NOTE: recommended to plot with a few different values of reg to understand how it affects the inversion.
    calib_factor (float): calibration factor to adjust for diode degradation. For LYMID, 1 in 2007, 1/0.8 in 2008 and 1/0.32 in 2009 is recommended.
    apply_offsets (bool): apply offsets to the brightness data before inverting. Default is False. NOTE: these offsets are pretty ad-hoc and raw signals (time-traces of each channel) should be checked before applying them.
    t_window (s): two-element list containing the time window to invert. Default is None, which means the entire shot is inverted.
    smooth_brightness_in_time (bool): smooth the brightness data in time before inverting. Default is False. Smoothing is achieved using a savgol filter (NOTE: currently hardcoded savgol of window length 19 and polynomial order 1).
    n_blocks (int): number of blocks to split the data into for inversion. Default is None, which means the entire shot is inverted in one go. NOTE: Highly recommend just leaving as None, since doing the inversion in blocks leads to funny edge effects.
    show_reconstruction (bool): generate a GUI to show the inversion and backprojection (thanks to Tomas). Default is False.
    '''

    print(f'Inversion has zero at {r_end} m')
    tomo = LLAMA_tomography(shot,system) # cmod brightness data already smoothed in time
    tomo.load_geometry(r_end=r_end)
    tomo.calibration( sys_err=sys_err)
    tomo.load_data(calib_factor=calib_factor, apply_offsets=apply_offsets, t_window=t_window, smooth_brightness_in_time=smooth_brightness_in_time)
    if n_blocks is None:
        n_blocks = len(tomo.tvec)
    tomo.calc_tomo(n_blocks=n_blocks, reg_value=reg)

    if show_reconstruction:
        tomo.show_reconstruction()

    return tomo.tvec,tomo.R_grid_b,tomo.y,tomo.y_err,tomo.backprojection, tomo.backprojection_grid




#if __name__ == "__main__":
#    tomo = tomoCMOD(None,None,r_end=0.93,sys_err=5,n_blocks=None)


#tomo = tomoCMOD(shot=1070511010,system='LYMID',r_end=0.93,sys_err=5, reg=0, show_reconstruction=True)

tomo = tomoCMOD(shot=1091210027,system='WB4LY',r_end=0.84,sys_err=5, apply_offsets=True, show_reconstruction=False)

