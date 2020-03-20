#include<iostream>
#include<cmath>

extern "C" {

double objective_function(int nl, int nb, int nd, int nv, double *vcen, double *sds, 
                          double *vvals, double *dust_cube, double *gas_cube, double *grad,
                          double smooth=0.1,int nthreads=2,bool regularize=true){
    ///
    /// C++ implementantion of the KT objective function
    ///
    /// Paramenters
    /// nl,nb,nd,nv: grid size in longitude, latitude, distance and velocity
    /// vcen :       array of size (nd*nb*nl) with vlsr central velocity for the model
    /// sds  :       array of size (nd*nb*nl) with standard deviations for the model
    /// vvals:       array of size (nv) with velocity values for the PPV cube
    /// dust_cube:   array of size (nd*nb*nl) containing the PPD data
    /// gas_cube:    array of size (nv*nb*nl) containing the PPV data
    /// grad:        array of size (nd*nb*nl) where the gradient will be written.
    /// smooth:      a smoothing parameter
    /// nthreads:    number of threads for parallel execution.
    
    /// The function returns the value of objective function and 
    /// writes its gradient in grad.
    ///

    double dv = fabs(vvals[1] - vvals[0]);              // Velocity step
    double obj = 0;                                     // Objective function to return
        
#pragma omp parallel for num_threads(nthreads) reduction(+ : obj)
    for (int l=0; l<nl; l++) {                         // Loop in longitude
        for (int b=0; b<nb; b++) {                     // Loop in latitude
            
            // Initialize a spectrum at (l,b)
            double *spec = new double[nv];
            for (int v=0; v<nv; v++) spec[v]=0;
            
            for (int d=0; d<nd; d++) {                 // Loop in distance
                // Pixel index in the PPD space
                size_t ppd_pix = d+b*nd+l*nb*nd;        // Arrays are ordered (d,b,l)
                // Value of dust cube at (l,b,d)
                double dc = dust_cube[ppd_pix];
                if (dc!=dc) continue;
                // Central velocity and dispersion at (l,b,d)
                double vc = vcen[ppd_pix];
                double sc = sds[ppd_pix];
                // Normalization of gaussian for model
                double gg = dv*dc / (sc*sqrt(2*M_PI));

                // Filling in spectrum at (l,b)
                for (int v=0; v<nv; v++) {
                    double diff = vvals[v]-vc;
                    spec[v] += gg*exp(-0.5*diff*diff/(sc*sc));
                }
                
                // Initizializing gradient
                grad[ppd_pix] = 0;
                
                // Adding regularization part of objective function and gradient
                if (regularize) {
                    if (l!=nl-1) obj += 0.5*smooth*pow(vcen[d+b*nd+(l+1)*nb*nd]-vc, 2.);
                    if (b!=nb-1) obj += 0.5*smooth*pow(vcen[d+(b+1)*nd+l*nb*nd]-vc, 2.);

                    if (l==0) grad[ppd_pix] += (vcen[d+b*nd+1*nb*nd]-vc);
                    else if (l==nl-1) grad[ppd_pix] += (vc-vcen[d+b*nd+(nl-2)*nb*nd]);
                    else grad[ppd_pix] += (vcen[d+b*nd+(l+1)*nb*nd]-vcen[d+b*nd+(l-1)*nb*nd]);
                    if (b==0) grad[ppd_pix] += (vcen[d+1*nd+l*nb*nd]-vc);
                    else if (b==nb-1) grad[ppd_pix] += (vc-vcen[d+(nb-2)*nd+l*nb*nd]);
                    else grad[ppd_pix] += (vcen[d+(b+1)*nd+l*nb*nd]-vcen[d+(b-1)*nd+l*nb*nd]);
                    grad[ppd_pix] *= smooth;
                }
            }
            
            // Now comparing with gas_cube and calculating objective function
            for (int v=0; v<nv; v++) {
                // Pixel index in the PPV space
                size_t ppv_pix = v+b*nv+l*nb*nv;        // Arrays are ordered (v,b,l)
                // Calculating difference between model and gas cube
                double diff_mod = spec[v] - gas_cube[ppv_pix];
                if (diff_mod!=diff_mod) continue;
                // Adding to the objective function
                obj += 0.5*diff_mod*diff_mod; 
                
                // Computing gradient
                for (int d=0; d<nd; d++){
                    // Pixel index in the PPD space
                    size_t ppd_pix = d+b*nd+l*nb*nd;
                    // Value of dust cube at (l,b,d)
                    double dc = dust_cube[ppd_pix];
                    if (dc!=dc) continue;
                    // Central velocity and dispersion at (l,b,d)
                    double vc = vcen[ppd_pix];
                    double sc = sds[ppd_pix];
                    // Constants of gradient
                    double gg = dv*dc / ((sc*sc*sc*sqrt(2*M_PI)));
                    
                    // Computing gradient of objective function (Gaussian part)
                    double diff = vvals[v]-vc;
                    grad[ppd_pix] += gg*diff*diff_mod*exp(-0.5*diff*diff/(sc*sc));
                }
            }
            delete [] spec;
        }
    }
    
    return obj;
}


void PPV_model(int nl, int nb, int nd, int nv, double *vcen, double *sds, 
               double *vvals, double *dust_cube, double *ppv_cube, int nthreads=2) {
        
    ///
    /// Compute a PPV model given a PPD vlsr array and dispersion array
    ///
    /// Paramenters
    /// nl,nb,nd,nv: grid size in longitude, latitude, distance and velocity
    /// vcen :       array of size (nd*nb*nl) with vlsr central velocity for the model
    /// sds  :       array of size (nd*nb*nl) with standard deviations for the model
    /// vvals:       array of size (nv) with velocity values for the PPV cube
    /// dust_cube:   array of size (nd*nb*nl) containing the PPD data
    /// ppv_cube:    array of size (nv*nb*nl) where the PPV will be written.
    /// nthreads:    number of threads for parallel execution.
    
    /// The function returns the PPV array in the ppv_cube variable.

    double dv = fabs(vvals[1] - vvals[0]);              // Velocity step
        
#pragma omp parallel for num_threads(nthreads)
    for (int l=0; l<nl; l++) {                         // Loop in longitude
        for (int b=0; b<nb; b++) {                     // Loop in latitude
            
            // Initialize a spectrum at (l,b)
            for (int v=0; v<nv; v++) ppv_cube[v+b*nv+l*nb*nv]=0;
            
            for (int d=0; d<nd; d++) {                 // Loop in distance
                // Pixel index in the PPD space
                size_t ppd_pix = d+b*nd+l*nb*nd;        // Arrays are ordered (d,b,l)
                // Value of dust cube at (l,b,d)
                double dc = dust_cube[ppd_pix];
                if (dc!=dc) continue;
                // Central velocity and dispersion at (l,b,d)
                double vc = vcen[ppd_pix];
                double sc = sds[ppd_pix];
                // Normalization of gaussian for model
                double gg = dv*dc / (sc*sqrt(2*M_PI));

                // Filling in spectrum at (l,b)
                for (int v=0; v<nv; v++) {
                    // Pixel index in the PPV space
                    size_t ppv_pix = v+b*nv+l*nb*nv;        // Arrays are ordered (v,b,l)
                    double diff = vvals[v]-vc;
                    ppv_cube[ppv_pix] += gg*exp(-0.5*diff*diff/(sc*sc));
                }
            }
        }
    }
    
    return;
}




}
