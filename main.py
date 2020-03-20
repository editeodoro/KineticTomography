from kinetictomography import KineticTomography

############################# USER INPUTS ######################################
CO_fname  = 'data/CO_Dame_regrid.fits'           # CO datacube
HI_fname  = 'data/HI_LAB_GASS_GALFA_regrid.fits' # HI datacube
EBV_fname = 'data/EBV_512_regrid.fits'           # E(B-V) datacube
lrange = [90,0]                                  # Longitude range in deg
brange = [-10,10]                                # Latitude range in deg
vrange = [-300,300]                              # Velocity range in km/s
restart = None                                   # Fits file with initial vlsr
nthreads = 8                                     # Number of cpus
Rsun, Vsun = 8.2, 240                            # Radius and velocity of the Sun
#################################################################################

if __name__ == '__main__':

    # Initializing a KineticTomography instance
    kt = KineticTomography(COcube=CO_fname,HIcube=HI_fname,EBVcube=EBV_fname,
                           lrange=lrange,brange=brange,vrange=vrange)
    
    print ("PPDV dimensions: %i x %i x %i x %i"%(kt.N_l,kt.N_b,kt.N_d,kt.N_v))
    
    # Initialize a mode with a flat rotation curve or with given inital conditions
    kt.initialize_model(ic=restart,smooth=1E-02,max_dv=35.,sd=8.,max_sd=15.,RSun=Rsun,VSun=Vsun,falloff=0)
    
    # Fitting velocities
    kt.run_mu(nthreads=nthreads, save_every=30, maxiter=100,C_func=True,regularize=False)
    #kt.run_per_single(save_every=500,C_func=True)
    
    # Save best fit model
    kt.save_model('KT_model.fits',ftype='fits')

    # Plotting 
    fig, ax = kt.plot_deviations(RSun=Rsun,VSun=Vsun)
    fig.savefig("KT_plot.pdf",bbox_inches='tight')
