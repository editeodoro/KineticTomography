import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import median_filter
from myutils import MWParticles, emptyfits
import matplotlib.pyplot as plt
import myutils as myu
import matplotlib as mpl



if __name__ == '__main__':

    ls,bs,ps,vs = np.genfromtxt("reid14.cat",usecols=(0,1,2,5),unpack=True)
    ds = 1/ps # Distance in kpc
    ls[ls>180] -= 360.
    # Distance modulus
    mod = 5. * (np.log10(ds) + 2.)
    
    # Values from flat rotation curve
    R0, V0 = 8.2, 220
    R,_,_ = myu.solsphr2galcyl(np.radians(ls),np.radians(bs),ds,R_Sun=R0)
    vflat_lsr = (V0*R0/R-V0) * np.sin(np.radians(ls)) * np.cos(np.radians(bs))
    
    with fits.open("bestfit_single_full.fits") as f:
        d, h = f[0].data, f[0].header
        nd,nb,nl = d.shape
        
    li = np.round((ls-h['CRVAL1'])/h['CDELT1'] + (h['CRPIX1']-1)).astype('int')
    bi = np.round((bs-h['CRVAL2'])/h['CDELT2'] + (h['CRPIX2']-1)).astype('int') 
    di = np.round((mod-h['CRVAL3'])/h['CDELT3'] + (h['CRPIX3']-1)).astype('int') 
    
    mk = (li>=0) & (li<nl) & (bi>=0) & (bi<nb) & (di>=0) & (di<nd)
    
    vels = d[di[mk],bi[mk],li[mk]]
    vflat_lsr = vflat_lsr[mk]    

    ls,bs,vs,ds = ls[mk],bs[mk],vs[mk],ds[mk]
    #vels -= vflat_lsr
    #vs -= vflat_lsr
    
    sl = np.linspace(np.nanmin(vels),np.nanmax(vels),100)
    
    #"""
    plt.plot(vels-vflat_lsr,vs-vflat_lsr,'o')
    #plt.plot(vflat_lsr,vs,'o')
    plt.plot(sl,sl,'-')
    plt.show()
    #"""
    
    # Polar plot
    cmap = plt.get_cmap('RdYlBu_r')
    norm = mpl.colors.Normalize(vmin=-30.,vmax=30.)
    
    # Getting a cube with vlsr for a flat rotation curve
    ll = (np.arange(nl)+1-h['CRPIX1'])*h['CDELT1']+h['CRVAL1']
    bb = (np.arange(nb)+1-h['CRPIX2'])*h['CDELT2']+h['CRVAL2']
    dd = (np.arange(nd)+1-h['CRPIX3'])*h['CDELT3']+h['CRVAL3']
    dd = 10.**(1 + 0.2 * dd - 3.)
    lll, bbb, ddd = np.meshgrid(ll,bb,dd,indexing='ij')
    R,_,_ = myu.solsphr2galcyl(np.radians(lll),np.radians(bbb),ddd,R_Sun=R0)
    vflat_cube = (V0*R0/R-V0) * np.sin(np.radians(lll)) * np.cos(np.radians(bbb))
    
    print (np.max(bb))
    
    # Residual cube
    res_cube = d - vflat_cube.T
    # Average velocity field over latitude
    ktav = np.nanmean(res_cube,axis=1)
    
    
    ll,dd = np.meshgrid(np.radians(ll),dd) 
    
    ax = plt.subplot(111, polar=True)
    # Plotting velocity field
    im = ax.pcolormesh(ll, dd, ktav,cmap=cmap,norm=norm) 
    cb = plt.colorbar(im)
    cb.set_label("Residual velocity (km/s)")
    # Plotting KT values at Reid14 positions
    ax.scatter(np.deg2rad(ls),ds,s=60,c=vels-vflat_lsr,cmap=cmap,norm=norm,edgecolors='k')
    # Plotting Reid14 datapoints
    ax.scatter(np.deg2rad(ls),ds,s=5,c=vs-vflat_lsr,cmap=cmap,norm=norm)
    
    # Plotting GC position
    ax.scatter(0,R0,s=60,c='k',marker='s',edgecolors='k')
    
    # Plot settings
    ax.set_ylim(0,10)

    # Set the circumference labels
    #ax.set_xticks(np.linspace(0, 2*np.pi, 24, endpoint=False))
    #ax.set_xticklabels(range(24))      
    ax.set_rlabel_position(155)     # Position of radial labels
    ax.set_theta_direction(1)       # Make the labels go anticlockwise (-1 for clockwise)
    ax.set_theta_offset(-np.pi/2)   # Place 0 at the bottom
    ax.grid(ls='--')
    ax.set_xlabel("Galactic Longitude")
    

    plt.show()
    
    
    #print (li,bi,di)
    exit()
