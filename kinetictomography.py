"""
This module defines classes and functions needed to perform KineticTomography
"""
from __future__ import division
import os, sys
from ctypes import *
try:
    import numpy as np    
    from astropy.io import fits
    from astropy.wcs import WCS
    from scipy.ndimage import median_filter
    from scipy import optimize
    from tqdm import tqdm
except ModuleNotFoundError:
    sys.exit("This package requires numpy, scipy, astropy and tqdm packages. Please install them.")
import momentum_cube_ops as mco


# Loading functions in C++ library
array_1d_double  = np.ctypeslib.ndpointer(dtype=np.double, ndim=1,flags="CONTIGUOUS")
libdir = os.path.dirname(os.path.realpath(__file__))
libobj = np.ctypeslib.load_library("libobj.so",libdir)

libobj.objective_function.restype = c_double
libobj.objective_function.argtypes = [c_int,c_int,c_int,c_int,array_1d_double,
                                      array_1d_double,array_1d_double,array_1d_double,
                                      array_1d_double,array_1d_double,c_double,c_int,c_bool]
libobj.PPV_model.restype = None
libobj.PPV_model.argtypes = [c_int,c_int,c_int,c_int,array_1d_double,array_1d_double,
                             array_1d_double,array_1d_double,array_1d_double,c_int]


class KineticTomography(object):
    """ 
    A class to perform KineticTomography:
    
    Two ways of initializing the class:
    
    1) Giving the name of CO, HI and EBV cubes:
        
       COcube (str):        name of fitsfile with CO data
       HIcube (str):        name of fitsfile with HI data
       EBVcube (str):       name of fitsfile with EBV data
       
       Optional parameters:
    
       lrange (list):       cut in longitude [lmin,lmax]
       brange (list):       cut in latitude [bmin,bmax]
       drange (list):       cut in distace [ebv_min,ebv_max]
       vrange (list):       cut in velocity [vmin,vmax]
       
    2) Giving two 3D arrays and four 1D arrays:
    
       gas_cube (array):    PPV array with gas column densities
       dust_cube (array):   PPD array with dust column densities 
       l_vals (array):      1D array with longitude values
       b_vals (array):      1D array with latitude values
       d_vals (array):      1D array with distance values
       v_vals (array):      1D array with velocity values
    
    Common parameters:
    
       filtering (bool):    Whether to filter the gas PPV cube
    
    """
        
    def __init__(self, filtering=True, **kwargs):
        super(KineticTomography, self).__init__()
        
        # Check if input parameters are fitscubes or just arrays
        isFits  = all(k in kwargs for k in ("COcube","HIcube","EBVcube"))
        isArray = all(k in kwargs for k in ("dust_cube","gas_cube","l_vals","b_vals","d_vals","v_vals"))
        if isFits:
            lrange=brange=drange=vrange=None
            if 'lrange' in kwargs: lrange = kwargs['lrange']
            if 'brange' in kwargs: brange = kwargs['brange']
            if 'drange' in kwargs: drange = kwargs['drange']
            if 'vrange' in kwargs: vrange = kwargs['vrange']

            self.__read_fitscubes(kwargs['COcube'],kwargs['HIcube'],kwargs['EBVcube'],
                                  lrange,brange,drange,vrange)
        elif isArray:
            self.__setup_from_arrays(kwargs['dust_cube'], kwargs['gas_cube'], 
                                     kwargs['l_vals'], kwargs['b_vals'], 
                                     kwargs['d_vals'], kwargs['v_vals'])
        else:
            raise NameError("KineticTomography ERROR: constructor requires either \
                            (COcube,HIcube,EBVcube) or (dust_cube,gas_cube,l_vals,b_vals,d_vals,v_vals)")
            
        if filtering:
            # Filtering gas cube
            self.gas_cube = median_filter(input=self.gas_cube, size=[3, 3, 1])
    
        self.gas_cube[self.gas_cube!=self.gas_cube] = 0
        self.dust_cube[self.dust_cube!=self.dust_cube] = 0
        
        # Initial model is null
        self.N_cells   = self.dust_cube.size
        self.N_model   = 2*self.N_cells
        self.v_model   = None
        self.gradient  = None
        self.bounds    = None


    def __setup_from_arrays(self,dust_cube,gas_cube,l_vals,b_vals,d_vals,v_vals):
        ##### Cubes properties 
        self.dust_cube = dust_cube
        self.gas_cube  = gas_cube
        self.l_vals    = l_vals
        self.b_vals    = b_vals
        self.d_vals    = d_vals
        self.v_vals    = v_vals
        self.N_l, self.N_b, self.N_d = dust_cube.shape
        self.N_v       = gas_cube.shape[2]
        
        # Creating a WCS
        d_vals = mco.distance_to_distmod(self.d_vals)
        self.wcs = WCS(naxis=4)
        self.wcs.wcs.ctype = ['GLON', 'GLAT', 'DISTMOD', 'VRAD']
        self.wcs.wcs.crpix = [1,1,1,1]
        self.wcs.wcs.cdelt = [l_vals[1]-l_vals[0],b_vals[1]-b_vals[0],d_vals[1]-d_vals[0],v_vals[1]-v_vals[0]]
        self.wcs.wcs.crval = [l_vals[0],b_vals[0],d_vals[0],v_vals[0]]


    def __read_fitscubes(self,COcube,HIcube,EBVcube,lrange,brange,drange,vrange):
        
        # Getting header information, first of all
        h_CO  = fits.open(COcube)[0].header
        h_EBV = fits.open(EBVcube)[0].header
    
        # Creating a new WCS for a PPDV data
        w = WCS(naxis=4)
        w.wcs.ctype = [h_CO['CTYPE1'], h_CO['CTYPE2'], h_EBV['CTYPE3'], h_CO['CTYPE3']]
        w.wcs.crpix = [h_CO['CRPIX1'], h_CO['CRPIX2'], h_EBV['CRPIX3'], h_CO['CRPIX3']]
        w.wcs.cdelt = [h_CO['CDELT1'], h_CO['CDELT2'], h_EBV['CDELT3'], h_CO['CDELT3']]
        w.wcs.crval = [h_CO['CRVAL1'], h_CO['CRVAL2'], h_EBV['CRVAL3'], h_CO['CRVAL3']]
        #w.wcs.cunit = ['deg', 'deg', 'mag', 'km/s']
        
        # Selecting a sub-region if requested
        l_idx = b_idx = v_idx = d_idx = [None, None]
        select = (lrange is not None) or (brange is not None) or (vrange is not None)
        if select:
            if lrange is not None:
                l_idx = np.round((lrange-w.wcs.crval[0])/w.wcs.cdelt[0]+(w.wcs.crpix[0]-1)).astype('int')
                #l_idx = (w.wcs_world2pix(lrange,0,0,0,0)[0]).astype(np.int)
                if l_idx.size==1: l_idx = [int(l_idx),int(l_idx)]
                else: l_idx = np.sort(l_idx)
                l_idx[1] += 1 
                w.wcs.crval[0] = w.wcs_pix2world(l_idx[0],0,0,0,0)[0]
                w.wcs.crpix[0] = 1
            if brange is not None:
                b_idx = np.round((brange-w.wcs.crval[1])/w.wcs.cdelt[1]+(w.wcs.crpix[1]-1)).astype('int') 
                #b_idx = (w.wcs_world2pix(0,brange,0,0,0)[1]).astype(np.int)
                if b_idx.size==1: b_idx = [int(b_idx),int(b_idx)]
                else: b_idx = np.sort(b_idx)
                b_idx[1] += 1 
                #w.wcs.crval[1] = w.wcs_pix2world(0,b_idx[0],0,0,0)[1]
                w.wcs.crpix[1] -= b_idx[0]
            if drange is not None:
                drangemod = mco.distance_to_distmod(drange)
                d_idx = np.round((drangemod-w.wcs.crval[2])/w.wcs.cdelt[2]+(w.wcs.crpix[2]-1)).astype('int') 
                if d_idx.size==1: d_idx = [int(d_idx),int(d_idx)]
                else: d_idx = np.sort(d_idx)
                if drange[0]==0: d_idx[0] = 0
                d_idx[1] += 1
                w.wcs.crval[2] = w.wcs_pix2world(0,0,d_idx[0],0,0)[2]
                w.wcs.crpix[2] = 1
            if vrange is not None:
                v_idx = np.round((vrange-w.wcs.crval[3])/w.wcs.cdelt[3]+(w.wcs.crpix[3]-1)).astype('int') 
                #v_idx = (w.wcs_world2pix(0,0,0,vrange,0)[3]).astype(np.int)
                if v_idx.size==1: v_idx = [int(v_idx),int(v_idx)]
                else: v_idx = np.sort(v_idx)
                v_idx[1] += 1
                w.wcs.crval[3] = w.wcs_pix2world(0,0,0,v_idx[0],0)[3]
                w.wcs.crpix[3] = 1
            
                
        # Opening CO data-cube (LON,LAT,VELO)
        with fits.open(COcube) as f_CO:
            gas_cube = f_CO[0].data[v_idx[0]:v_idx[1],b_idx[0]:b_idx[1],l_idx[0]:l_idx[1]]
            # Converting to column using an X_CO = 2E20 cm^-2 / (K km/s)
            gas_cube[np.abs(gas_cube) < 0.1] = 0.
            gas_cube[gas_cube!=gas_cube] = 0.
            gas_cube *= 2*2*h_CO['CDELT3']                    # N_CO protons 1E20 cm^-2
    
        # Opening HI data-cube, converting to column and adding to gas cube
        with fits.open(HIcube) as f_HI:
            h_HI = f_HI[0].header 
            gas_cube += 1.82E-02*\
                        f_HI[0].data[v_idx[0]:v_idx[1],b_idx[0]:b_idx[1],l_idx[0]:l_idx[1]]*\
                        h_HI['CDELT3'] # N_HI in protons in 1E20 cm^-2 
    
        # Opening Dust datacube (LON,LAT,DISTMOD) and 
        # converting to column using the conversion factor in Peek+13
        with fits.open(EBVcube) as f_EBV:
            h_EBV =  f_EBV[0].header
            dust_cube = 70*f_EBV[0].data[d_idx[0]:d_idx[1],b_idx[0]:b_idx[1],l_idx[0]:l_idx[1]]  # N_H in 1E20 cm^-2

        # Ordering axis such that (LONG, LAT, VELO)
        self.gas_cube  = gas_cube.swapaxes(0,2)
        self.dust_cube = dust_cube.swapaxes(0,2)
        
        self.wcs = w
        # Array sizes
        self.N_l, self.N_b, self.N_v = self.gas_cube.shape
        self.N_d = self.dust_cube.shape[2]
    
        # WCS values (following does not work, not sure why)
        #l_vals = w.wcs_pix2world(np.arange(N_l),0,0,0,0)[0]
        #b_vals = w.wcs_pix2world(0,np.arange(N_b),0,0,0)[1]
        #d_vals = w.wcs_pix2world(0,0,np.arange(N_d),0,0)[2]
        #v_vals = w.all_pix2world(0,0,0,np.arange(N_v),0)[3]
        self.l_vals = (np.arange(self.N_l)+1-w.wcs.crpix[0])*w.wcs.cdelt[0]+w.wcs.crval[0]
        self.b_vals = (np.arange(self.N_b)+1-w.wcs.crpix[1])*w.wcs.cdelt[1]+w.wcs.crval[1]
        self.d_vals = (np.arange(self.N_d)+1-w.wcs.crpix[2])*w.wcs.cdelt[2]+w.wcs.crval[2]
        self.v_vals = (np.arange(self.N_v)+1-w.wcs.crpix[3])*w.wcs.cdelt[3]+w.wcs.crval[3]
        self.d_vals = mco.distmod_to_distance(self.d_vals)
        

    def initialize_model(self,ic=None,max_dv=35.,max_sd=15.,sd=8.,smooth=0.1,RSun=8.2,VSun=240.,falloff=0):
        """ Define the initial guesses for the fit. If ic is None, a default flat rotation
            model is used. Otherwise, ic is the name of a 3D fitsfile containing the 
            initial LSR velocities
        """
        self.smooth = smooth
        if ic is None:
            # Set a default model with flat rotation curve v0
            vlsrs = mco.rotation_curve(self.l_vals[:, None, None], 
                                       self.b_vals[None, :, None], 
                                       self.d_vals[None, None, :], 
                                       V_rot=VSun,
                                       R_sol=RSun,
                                       falloff=falloff)
        else:
            _ic = fits.open(ic)[0].data
            _nd,_nb,_nl = _ic.shape
            ok = _nd==self.N_d and _nb==self.N_b and _nl==self.N_l
            if not ok:
                raise ValueError("Initial condition cube dimensions do not match with input cube: \
                                  %s x %s x %s vs %s x %s x %s"%(_nl,_nb,_nd,self.N_l,self.N_b,self.N_d)) 
            vlsrs = _ic.swapaxes(0,2)
        
        # vlsrs is shaped (N_l,N_b,N_d)
        # Velocity and dispersion boundaries 
        cen_bounds, sd_bounds = [], []
        for vr in vlsrs.flat:
            cen_bounds.append((vr - max_dv, vr + max_dv))
            sd_bounds.append((1., max_sd))
        self.bounds = cen_bounds
        self.bounds.extend(sd_bounds)
        
        # v_model is a 1d vector containing initial vlsr and disp
        self.v_model = np.zeros(self.N_model)
        self.v_model[0:self.N_cells] = vlsrs.ravel()
        self.v_model[self.N_cells:] += sd


    def objfunc_c(self,mu,sd,dust_cube,gas_cube,nl,nb,nthreads=1,regularize=False):
        """ Returns the objective function and its gradient 
            using the C++ implementation defined in objfunction.cc 
        """
        grad = np.zeros(shape=self.N_d*nl*nb).astype(np.double)
        objf = libobj.objective_function(nl,nb,self.N_d,self.N_v,
                                  mu.astype(np.double),
                                  sd.astype(np.double),
                                  self.v_vals.astype(np.double),
                                  np.ravel(dust_cube.astype(np.double)),
                                  np.ravel(gas_cube.astype(np.double)),
                                  grad,self.smooth,nthreads,regularize)
        return (objf,grad)
    
    
    def objfunc_py(self,mu,sd,dust_cube,gas_cube,nl,nb,nthreads=1,regularize=False):
       
        """ 
        This function calculates the objective function and its gradient 
        given input central velocity (mu) and velocity dispersion (sds) 
        """
        _mu = mu.reshape([nl, nb, self.N_d, 1])
        _sd = sd.reshape([nl, nb, self.N_d, 1])
        _dc = dust_cube.reshape([nl, nb, self.N_d, 1])
        _gc = gas_cube.reshape([nl, nb, self.N_v])
        _v_vals = self.v_vals[None, None, None, :]
        dv = np.abs(self.v_vals[-1]-self.v_vals[-2])

        # Calculating model PPV cube
        diff = _mu - _v_vals
        unsc = np.exp(-0.5 * diff**2 /_sd**2)
        ppv_model = np.nansum(unsc * _dc / _sd, axis=2)
        ppv_model *= dv / np.sqrt(2. * np.pi)
    
        # Model residual
        diff_mod = (ppv_model - _gc)
        
        # Unregularized objective function
        obj = 0.5 * np.nansum(diff_mod**2)
        
        # Now, calculating gradient of objective function
        diff_mod = diff_mod[:, :, None, :]
        _sd = _sd[:,:,:,0]
        _dc = _dc[:,:,:,0]
        
        grad = np.nansum(diff*unsc*diff_mod,axis=3)
        grad *= -_dc*dv/(self.smooth*np.sqrt(2.*np.pi)*_sd**3)

        # Regularized objective function
        if regularize:
            # Adding regularization to objective function
            diff_I = _mu[1:] - _mu[:-1]             # Differences in l
            diff_J = _mu[:, 1:] - _mu[:, :-1]       # Differences in b
            obj += 0.5 * self.smooth * np.nansum(diff_I**2)
            obj += 0.5 * self.smooth * np.nansum(diff_J**2)
            # Adding regularization to gradient
            _mu = _mu[:,:,:,0]
            grad[1:-1]    += _mu[2:] - _mu[:-2]
            grad[0]       += _mu[1] - _mu[0]
            grad[-1]      += _mu[-1] - _mu[-2]
            grad[:, 1:-1] += _mu[:, 2:] - _mu[:, :-2]
            grad[:, 0]    += _mu[:, 1] - _mu[:, 0]
            grad[:, -1]   += _mu[:, -1] - _mu[:, -2]
        
        grad *= self.smooth
        
        return obj, np.ravel(grad)


    def run_mu(self,nthreads=2,maxiter=5000,save_every=None,C_func=True,regularize=True):
        """ This function optimises only the vlsr """
        
        if self.v_model is None:
            raise ValueError('KineticTomography ERROR: call initialize_model() before run_mu()')
        
        # Considering only boundaries in velocity
        bounds = self.bounds[0:self.N_cells]
        
        subiter = maxiter - 1 if save_every is None else save_every
        func    = self.objfunc_c if C_func else self.objfunc_py
        sds     = self.v_model[self.N_cells:]
        
        done = False
        total_iter = 0
        print ("Fitting ...")
        while not done:
            # Initial guesses for vlsr
            x0 = self.v_model[:self.N_cells]
            args = (sds,self.dust_cube,self.gas_cube,self.N_l,self.N_b,nthreads,regularize)
            # Run optimizer
            output = optimize.minimize(fun=func, x0=x0,args=args,jac=True,\
                                       bounds=bounds,method='L-BFGS-B',
                                       options=dict(maxiter=subiter))
            
            # Writing optimization results to v_model for next iteration
            self.v_model[:self.N_cells] = output['x']
            total_iter += output['nit']
            done = (total_iter >= maxiter) | (output['status']!=1) 
            print (" iter=%s/%s fun=%s"%(total_iter,maxiter,output['fun']))
            if output['status']==0: print ("Convergence achieved!!")
            elif output['status']!=1: print (output)
            # Saving partial results if requested
            if save_every is not None:
                #self.v_model.dump(save_name)
                self.save_model(fname='partial.fits',ftype='fits')


    def run_per_single(self, maxiter=5000,start_i=0,save_every=None,C_func=True):
        """ Fit on a single sightline at a time """
        
        d_inds = np.arange(self.N_d)
        N_lb = self.N_l * self.N_b
        
        func    = self.objfunc_c if C_func else self.objfunc_py
        
        for onsky_i in tqdm(range(start_i, N_lb)):
            
            #print ("Working on sightline %s/%s"%(onsky_i,N_lb))
            cell_inds = np.ravel_multi_index([onsky_i, d_inds],
                                                 [N_lb, self.N_d])
            l_i = onsky_i // self.N_b
            b_i = onsky_i % self.N_b
            
            #x0 = np.hstack([self.v_model[cell_inds], 
            #               self.v_model[cell_inds + self.N_cells]])
            x0 = self.v_model[cell_inds]
            bounds = [self.bounds[c_i] for c_i in cell_inds]
            #bounds.extend([self.bounds[c_i] for c_i in (cell_inds+self.N_cells)])
            sds = self.v_model[cell_inds + self.N_cells]
            dc = self.dust_cube[l_i, b_i]
            if np.all(dc==0) or np.all(dc!=dc): continue
            args = (sds,dc,self.gas_cube[l_i, b_i],1,1,1,False)
            
            output = optimize.minimize(fun=func, x0=x0,args=args,jac=True,\
                                       bounds=bounds,method='L-BFGS-B',
                                       options=dict(maxiter=maxiter,ftol=1E-11))
            
            self.v_model[cell_inds] = output['x'][:self.N_d]
            #self.v_model[cell_inds+self.N_cells] = output[0][self.N_d:]
            if output['status']!=0:
                print ("NO CONVERGENCE: %s"%output['message'])
            if save_every is not None :
                if (onsky_i % save_every) == 0:
                    #self.v_model.dump(save_name)
                    self.save_model(fname='partial_single.fits',ftype='fits')

    
    def save_model(self,fname,ftype='fits'):
        """ Saving vlsr model to a file """
        if 'fits' in ftype.lower():
            array = self.v_model[:self.N_cells].reshape([self.N_l,self.N_b,self.N_d]).swapaxes(0,2)
            head  = self.wcs.to_header()
            head['BUNIT'] = "km/s"
            head['BTYPE'] = "VRAD"
            head['WCSAXES'] = 3
            del head['LATPOLE']
            del head['LONPOLE']
            del head['CRPIX4']
            del head['CDELT4']
            del head['CRVAL4']
            del head['CTYPE4']
            del head['CUNIT4']
            fits.writeto(fname,array.astype(np.float32),head,overwrite=True)
        else:
            self.v_model.reshape([2, self.N_l, self.N_b, self.N_d]).dump(fname)
    
    
    def get_ppv_model(self,nthreads=2):
        """ Returns the PPV model """
        ppv = np.zeros(shape=self.N_l*self.N_b*self.N_v).astype(np.double)
        libobj.PPV_model(self.N_l,self.N_b,self.N_d,self.N_v,
                         self.v_model[:self.N_cells].astype(np.double),
                         self.v_model[self.N_cells:].astype(np.double),
                         self.v_vals.astype(np.double),
                         np.ravel(self.dust_cube.astype(np.double)),ppv,nthreads)
        return ppv.reshape([self.N_l,self.N_b,self.N_v])
    
    
    def plot_deviations(self,RSun=8.2,VSun=240.):
        """ Returns a matplotlib figure with polar plots """
        
        import matplotlib.pyplot as plt
        
        _vflats = mco.rotation_curve(self.l_vals[:, None, None], 
                                     self.b_vals[None, :, None], 
                                     self.d_vals[None, None, :], 
                                     V_rot=VSun,
                                     R_sol=RSun)
        _vlsrs = self.v_model[:self.N_cells].reshape([self.N_l,self.N_b,self.N_d])
        
        # Velocity field
        vf = np.nanmean(_vlsrs,axis=1)
        # Polar plot
        fig, ax = plt.subplots(nrows=1,ncols=2, subplot_kw=dict(polar=True))
        fig.tight_layout() 
        
        # Plotting velocity field
        im = ax[0].pcolormesh(np.radians(self.l_vals), self.d_vals, vf.T,cmap=plt.get_cmap('RdYlBu_r'),rasterized=True) 
        cb = fig.colorbar(im,orientation='horizontal',ax=ax[0])
        cb.set_label("LSR velocity (km/s)")
       
        # Plotting residual from vflat field
        im = ax[1].pcolormesh(np.radians(self.l_vals), self.d_vals, (vf-np.nanmean(_vflats,axis=1)).T,cmap=plt.get_cmap('RdYlBu_r'),rasterized=True) 
        cb = fig.colorbar(im,orientation='horizontal',ax=ax[1])
        cb.set_label("Residual LSR velocity (km/s)")

        for a in ax:
            # Plotting GC position
            a.scatter(0,RSun,s=60,c='k',marker='s',edgecolors='k')
            # Plot settings
            a.set_rlabel_position(155)     # Position of radial labels
            a.set_theta_direction(1)       # Make the labels go anticlockwise (-1 for clockwise)
            a.set_theta_offset(-np.pi/2)   # Place 0 at the bottom
            a.grid(ls='--')
        
        return fig, ax

"""
    def run_full(self, maxiter=5000, verbose=False, subiter=50):
        bounds = self.bounds
        func = self.obj_grad
        done = False
        disp = verbose
        total_iter = 0
        while not done:
            x0 = self.v_model
            output = fmin_l_bfgs_b(func, x0, maxiter=subiter,
                                   disp=disp, bounds=bounds)
            self.v_model = output[0]
            d = output[2]
            total_iter += d['nit'] - 1
            reason = d['warnflag']
            done = (total_iter >= maxiter) or (reason != 1)
            #if verbose:
            #    print total_iter, output[1]
"""

