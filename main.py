from kinetictomography import KineticTomography


############################# USER INPUTS #############################
CO_fname  = 'data/CO_Dame_regrid.fits'#'data/COGAL_all_interp.fits'         # CO datacube
HI_fname  = 'data/HI_LAB_GASS_GALFA_regrid.fits'#'data/LAB_GASS_GALFA_DamePlane.fits' # HI datacube
EBV_fname = 'data/EBV_512_regrid.fits'#'data/EBV_512_DamePlane.fits'        # E(B-V) datacube
lrange = [90,0]                               # Longitude range in deg
brange = [-10,10]                              # Latitude range in deg
vrange = [-300,300]                              # Velocity range in km/s
restart = None#'bestfit_QI_noreg.fits'                                   # Fits file with initial vlsr
nthreads = 4
Rsun, Vsun = 8.2, 240.
#######################################################################


if __name__ == '__main__':

    # Initializing a strict KineticTomography instance
    #kt = KineticTomography(dust_cube, gas_cube_filt, l_vals, b_vals, d_vals, v_vals)
    kt = KineticTomography(COcube=CO_fname,HIcube=HI_fname,EBVcube=EBV_fname,
                           lrange=lrange,brange=brange,vrange=vrange)
    
    print ("PPDV dimensions: %i x %i x %i x %i"%(kt.N_l,kt.N_b,kt.N_d,kt.N_v))
    
    kt.initialize_model(ic=restart,smooth=1E-02,max_dv=35.,sd=8.,max_sd=15.,RSun=Rsun,VSun=Vsun,falloff=0)
    
    #fits.writeto("IC.fits",sp.v_model[0:kt.N_cells].reshape([N_l, N_b, N_d]).swapaxes(0,2).astype(np.float32),newh,overwrite=True)

    kt.run_mu(nthreads=nthreads, save_every=30, maxiter=100,C_func=True,regularize=False)
    #kt.run_per_single(save_every=500,C_func=True)
    
    
    kt.save_model('pippo.fits',ftype='fits')

    fig, ax = kt.plot_deviations(RSun=Rsun,VSun=Vsun)
    fig.savefig("pippo.pdf",bbox_inches='tight')
