import numpy as np
import wcs_defs
from astropy.io import fits
from common import *
import pandas as pd
from scipy import interpolate
import xarray

R_sol = 8.2 #kpc
V_rot = 240. #km/sec
#elliptical motion coefficients
ell_alpha = 8.67
ell_beta = -1.08
#Clemens polynomials
A_coeffs = np.asarray([-17731., 54904., 
                      -68287.3, 43980.1,
                      -15809.8, 3069.81,
                      0.])
A_poly = np.poly1d(A_coeffs)
B_coeffs = np.asarray([-2.110625, 25.073006, 
                      -110.73531, 231.87099,
                      -248.1467, 325.0912])
B_poly = np.poly1d(B_coeffs)
C_coeffs = np.asarray([0.00129348, -0.08050808, 
                      2.0697271, -28.4080026,
                      224.562732, -1024.06876,
                      2507.60391, -2342.6564])
C_poly = np.poly1d(C_coeffs)
D_poly = np.poly1d(234.88)
AB_bound = 0.09 * R_sol
BC_bound = 0.45 * R_sol
CD_bound = 1.6 * R_sol


def lsr_shift(glon, glat, U, V, W):
    """ 
        Signs are set up so that (IAU vlos) + (shift) is right
        If you want to instead apply it to put something shifted 
        in IAU terms, do (shifted - shift)
    """
    dU = U - 10.3
    dV = V - 15.3
    dW = W - 7.7
    l_r = np.radians(glon)
    b_r = np.radians(glat)
    shift = (dW * np.sin(l_r) + np.cos(b_r) * 
            (dU * np.cos(l_r) + dV * np.sin(l_r)))
    return shift


def edges_from_centers(centers):
    edges = np.zeros(centers.size + 1)
    edges[1:-1] = 0.5 * (centers[1:] + centers[:-1])
    edges[0] = centers[1] - 0.5*(centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5*(centers[-1] - centers[-2])
    return edges


def distmod_to_distance(mu_vals):
    d_vals = 10.**(1 + 0.2 * mu_vals - 3.) #-3 to get kpc
    return d_vals


def distance_to_distmod(dist_vals):
    mu_vals = 5. * (np.log10(dist_vals) + 2.)
    return mu_vals


def galcyl_from_solsphr(l_vals, b_vals, d_vals, lb_rad=False, R_sol=R_sol):
    """ Convert from l,b,d to Galactic R,phi,z """    
    if lb_rad:
        l_in_rad = l_vals
        b_in_rad = b_vals
    else:
        l_in_rad = np.radians(l_vals)
        b_in_rad = np.radians(b_vals)
        
    d_in_plane = d_vals * np.cos(b_in_rad)
        
    R_vals = d_in_plane**2 + R_sol**2
    R_vals = R_vals - (2 * d_in_plane * R_sol * np.cos(l_in_rad))
    R_vals = np.sqrt(R_vals)

    phi_vals = np.arcsin(np.sin(l_in_rad) * d_in_plane / R_vals)
    z_vals = d_vals * np.sin(b_in_rad)
    return R_vals, phi_vals, z_vals


def galcar_from_solsphr(l_vals, b_vals, d_vals, R_sol=R_sol):
    """ Convert from l,b,d to Galactic Cartesian (x,y,z) """    
    
    l_in_rad = np.radians(l_vals)
    b_in_rad = np.radians(b_vals)
    d_in_plane = d_vals * np.cos(b_in_rad)

    x_vals = d_in_plane * np.cos(l_in_rad) - R_sol
    y_vals = d_in_plane * np.sin(l_in_rad)
    z_vals = d_vals * np.sin(b_in_rad)
    return x_vals, y_vals, z_vals


def rotation_curve(l_vals, b_vals, d_vals, with_elliptical=False, V_rot=V_rot, R_sol=R_sol, falloff=0):
    """ This function returns the lsr velocity for a flat rotation curve V_rot

        Output is an array with shape (N_l, N_b, N_d)
    """
    
    l_in_rad = np.radians(l_vals)
    b_in_rad = np.radians(b_vals)
    
    R, _, z = galcyl_from_solsphr(l_vals, b_vals, d_vals, R_sol=R_sol)
    v_theta = V_rot
    if falloff>0: v_theta*=np.exp(-abs(z)/falloff)
    
    v_r = np.sin(l_in_rad) * np.cos(b_in_rad)
    v_r = v_r * (v_theta * R_sol / R - V_rot)
    
    if with_elliptical:
        v_r += elliptical_correction(l_vals, b_vals, d_vals, R_sol=R_sol)    
    return v_r


def clemens_rotation_curve(l_vals, b_vals, d_vals,
                           V_rot=V_rot, R_sol=R_sol):
    l_in_rad = np.radians(l_vals)
    b_in_rad = np.radians(b_vals)

    R_vals, _, _ = galcyl_from_solsphr(l_vals, b_vals, d_vals)
    R_vals = np.asarray(R_vals)
    initial_shape = R_vals.shape
    R_vals = np.ravel(R_vals)
    condlist = (R_vals <= AB_bound,
                (AB_bound < R_vals) & (R_vals <= BC_bound),
                (BC_bound < R_vals) & (R_vals <= CD_bound),
                (CD_bound < R_vals))
    funclist = (A_poly, B_poly, C_poly, D_poly)
    v_r = np.piecewise(R_vals, condlist, funclist)
    v_r = v_r.reshape(initial_shape)

    v_r = (R_sol * v_r / R_vals.reshape(initial_shape) - V_rot)
    v_r *= np.sin(l_in_rad) * np.cos(b_in_rad)
    return v_r


def elliptical_correction(l_vals, b_vals, d_vals, R_sol=R_sol):
    l_in_rad = np.radians(l_vals)
    b_in_rad = np.radians(b_vals)

    R_vals, phi_vals, _ = galcyl_from_solsphr(l_vals, b_vals, d_vals, R_sol=R_sol)

    _x = (R_vals - R_sol)/R_sol
    v_Pi = ell_alpha * _x + ell_beta * _x**2
    v_Pi_r = v_Pi * np.cos(phi_vals) * np.cos(b_in_rad)
    v_Pi_r *= np.sqrt(1 - (np.sin(l_in_rad) * R_sol / R_vals)**2)
    return v_Pi_r


def build_rotation_curve_3cube(l_inds, b_inds, d_inds, meshed=False, 
                               with_elliptical=False, d_samp=3):
    wcs = wcs_defs.get_ppdv_wcs()

    if meshed:
        l_g = l_inds
        b_g = b_inds
        d_g = d_inds
    else:
        l_g, b_g, d_g = np.meshgrid(l_inds, b_inds, d_inds, indexing='ij')

    l_vals, b_vals, mu_vals, _ = wcs.wcs_pix2world(l_g, b_g, d_g, 0,
                                                   wcs_defs.origin)

    mu_edges = edges_from_centers(np.unique(mu_vals))
    rotation_curve_3cube = np.zeros(l_g.shape)
    #d_vals = distmod_to_distance(mu_vals)
    N_d = rotation_curve_3cube.shape[-1]

    l_samps = np.zeros([l_vals.shape[0], l_vals.shape[1], d_samp])
    l_samps[:, :, :] = l_vals[:, :, 0, None]
    b_samps = np.zeros([b_vals.shape[0], b_vals.shape[1], d_samp])
    b_samps[:, :, :] = b_vals[:, :, 0, None]

    for d_i in range(N_d):
        d_samps = np.linspace(mu_edges[d_i], mu_edges[d_i+1], d_samp)
        d_samps = distmod_to_distance(d_samps)
        v_r_samps = rotation_curve(l_samps,
                                   b_samps,
                                   d_samps,
                                   with_elliptical= with_elliptical)
        rotation_curve_3cube[:, :, d_i] = np.average(v_r_samps, axis=-1)
    return rotation_curve_3cube


def build_clemens_rotation_curve_3cube(l_inds, b_inds, d_inds):
    wcs = wcs_defs.get_ppdv_wcs()

    l_g, b_g, d_g = np.meshgrid(l_inds, b_inds, d_inds, indexing='ij')

    l_vals, b_vals, mu_vals, _ = wcs.wcs_pix2world(l_g, b_g, d_g, 0,
                                                   wcs_defs.origin)
    d_vals = distmod_to_distance(mu_vals)
    clemens_rotation_curve_3cube = clemens_rotation_curve(l_vals, b_vals, d_vals)
    return clemens_rotation_curve_3cube


def build_rotation_curve_offset_4cube(l_inds, b_inds, d_inds, v_inds, 
                                      meshed=False, with_elliptical=False):
    rotation_curve_3cube = build_rotation_curve_3cube(l_inds, 
                                                      b_inds,
                                                      d_inds,
                                                      meshed=meshed,
                                                      with_elliptical=
                                                      with_elliptical,
                                                      d_samp=3)
    velo_wcs = wcs_defs.get_v_wcs()
    v_vals = np.asarray(velo_wcs.wcs_pix2world(v_inds,
                        wcs_defs.origin)).ravel()

    rotation_curve_offset_4cube = (v_vals[None, None, None, :] - 
                                   rotation_curve_3cube[:, :, :, None])
    return rotation_curve_offset_4cube


def build_velocity_moment_3cube(full_4cube, v_inds):
    ppdv_wcs = wcs_defs.get_ppdv_wcs()
    _, _, _, v_vals = ppdv_wcs.wcs_pix2world(0, 0, 0, 
                                             v_inds, wcs_defs.origin)
    moment_3cube = np.zeros(full_4cube.shape[0:3])
    moment_3cube = (full_4cube * v_vals).sum(axis=3)
    collapsed_4cube = full_4cube.sum(axis=3)
    moment_3cube /= collapsed_4cube
    moment_3cube = np.ma.masked_where(collapsed_4cube==0, moment_3cube)
    return moment_3cube


def build_velocity_moment_3cube_from_components(U, spatial_shape, 
                                                V, veldist_shape,
                                                v_inds):
    N_x, N_y = spatial_shape
    N_d, N_v = veldist_shape
    N_k = V.shape[1]
    ppdv_wcs = wcs_defs.get_ppdv_wcs()
    _, _, _, v_vals = ppdv_wcs.wcs_pix2world(0, 0, 0, 
                                             v_inds, wcs_defs.origin)
    v_vals = np.asarray(v_vals).ravel()

    V_ur = V.reshape([N_d, N_v, N_k])
    comp_moments = np.sum(V_ur * v_vals[None, :, None], axis=1)
    moment_3cube = np.dot(U, comp_moments.T)
    moment_3cube /= np.dot(U, V_ur.sum(axis=1).T)
    moment_3cube = np.nan_to_num(moment_3cube)

    moment_3cube = moment_3cube.reshape([N_x, N_y, N_d])
    moment_3cube = moment_3cube.swapaxes(0,1)
    return moment_3cube


def ppv_cube_from_decomposition(spatial, veldist, full_shape):
    veldist_unraveled = veldist.reshape([full_shape[2], 
                                        full_shape[3], -1])
    ppv = np.dot(spatial, veldist_unraveled.sum(axis=0).T)
    ppv = ppv.reshape([full_shape[0], full_shape[1],
                      full_shape[3]])
    ppv = ppv.swapaxes(0, 1)
    return ppv


def ppd_cube_from_decomposition(spatial, veldist, full_shape):
    veldist_unraveled = veldist.reshape([full_shape[2], 
                                        full_shape[3], -1])
    ppd = np.dot(spatial, veldist_unraveled.sum(axis=1).T)
    ppd = ppd.reshape([full_shape[0], full_shape[1],
                      full_shape[2]])
    ppd = ppd.swapaxes(0, 1)
    return ppd

    
def get_field_inds(field):
    if field == 'anticenter':
        b_inds = np.arange(0, 240)
        l_inds = np.arange(-200, -100)
        d_inds = np.arange(wcs_defs.N_d)
        v_inds = np.arange(130, 310)

    #Orion
    if field == 'Orion':
        b_inds = np.arange(50, 240)
        l_inds = np.arange(2500-wcs_defs.N_l, 2830-wcs_defs.N_l)
        d_inds = np.arange(wcs_defs.N_d)
        v_inds = np.arange(185, 350)

    #stripe:
    if field.split('_')[0] == 'stripe':
        stripe_i = int(field.split('_')[1].rstrip('d'))
        b_inds = np.arange(241 - 80, 241 + 80)
        l_inds = np.arange(stripe_i * 100, (stripe_i+1) * 100)
        if stripe_i == -1:
            l_inds = l_inds[0:-1]
        d_inds = np.arange(wcs_defs.N_d)
        v_inds = np.arange(50, 330)

    if field == 'band':
        b_inds = np.arange(241 - 80, 241 + 80)
        d_inds = np.arange(wcs_defs.N_d)
        v_inds = np.arange(50, 330)
        l_inds = []
        for stripe_i in range(-5, 15):
            l_inds.append(np.arange(stripe_i * 100, (stripe_i+1) * 100))
        l_inds = np.concatenate(l_inds)[:-100]

    #Aquila
    if field == 'Aquila':
        l_inds = np.arange(1121, 1399)
        b_inds = np.arange(273, 481)
        d_inds = np.arange(wcs_defs.N_d)
        v_inds = np.arange(wcs_defs.N_v)

    #distance correction
    #d_inds -= 1.5
    return l_inds, b_inds, d_inds, v_inds


def load_field(field, N_k):
    l_inds, b_inds, d_inds, v_inds = get_field_inds(field)

    ppdv_wcs = wcs_defs.get_ppdv_wcs()
    l_g, d_g = np.meshgrid(l_inds, d_inds, indexing='ij')
    l_g, _, mu_g, _ = ppdv_wcs.wcs_pix2world(l_g, 0, d_g, 0, 
                                             wcs_defs.origin)
    _, b_vals, _, _ = ppdv_wcs.wcs_pix2world(0, b_inds, 0, 0, 
                                             wcs_defs.origin)
    d_g = distmod_to_distance(mu_g)
    x_g, y_g, _ = galcar_from_solsphr(l_g, 0, d_g)
    _, _, _, v_vals = ppdv_wcs.wcs_pix2world(0, 0, 0, v_inds, 
                                             wcs_defs.origin)
    position_grids = [l_g, d_g, b_vals, v_vals]

    spatial_shape = [b_inds.size, l_inds.size]
    veldist_shape = [d_inds.size,  v_inds.size]
    U = np.load(solution_location + 'U_{}_{}.np'.format(field, N_k))
    V = np.load(solution_location + 'V_{}_{}.np'.format(field, N_k))

    lsr_moment_cube = build_velocity_moment_3cube_from_components(U, spatial_shape, V, veldist_shape, v_inds)
    flat_rotation_cube = build_rotation_curve_3cube(l_inds, b_inds, d_inds, with_elliptical=False)
    clemens_rotation_cube = build_clemens_rotation_curve_3cube(l_inds, b_inds, d_inds)

    ppd = ppd_cube_from_decomposition(U, V, 
                                      [spatial_shape[0], 
                                      spatial_shape[1], 
                                      veldist_shape[0], 
                                      veldist_shape[1]])
    return ppd, lsr_moment_cube, flat_rotation_cube, clemens_rotation_cube, position_grids


def stitch_stripes(stripe_inds, Nk_vals):
    ppd_cubes = []
    lsr_moment_cubes = []
    flat_rotation_cubes = []
    clemens_rotation_cubes = []
    l_gs = []
    d_gs = []
    v_vals = None
    for stripe_i, Nk_val in zip(stripe_inds, Nk_vals):
        info = load_field('stripe_{}'.format(stripe_i), Nk_val)
        ppd_cubes.append(info[0])
        lsr_moment_cubes.append(info[1])
        flat_rotation_cubes.append(info[2])
        clemens_rotation_cubes.append(info[3])
        position_grids = info[4]
        l_gs.append(position_grids[0])
        d_gs.append(position_grids[1])
        b_vals = position_grids[2]
        v_vals = position_grids[3]
    ppd_cube = np.concatenate(ppd_cubes, axis=0)
    lsr_moment_cube = np.concatenate(lsr_moment_cubes, axis=0)
    flat_rotation_cube = np.concatenate(flat_rotation_cubes, axis=0)
    clemens_rotation_cube = np.concatenate(clemens_rotation_cubes, axis=0)
    l_g = np.concatenate(l_gs, axis=0)
    d_g = np.concatenate(d_gs, axis=0)
    position_grids = [l_g, d_g, b_vals, v_vals]
    return ppd_cube, lsr_moment_cube, flat_rotation_cube, clemens_rotation_cube, position_grids
            

def get_steps(kind):
    l_inds, _, _, _ = get_field_inds('band')
    N_l = l_inds.size

    #full (100) even (start at 0)
    if kind == 'fe':
        step_size = 100

    #half even
    if kind == 'he':
        step_size = 50
        start = 0
        stop = start + step_size
        starts = []
        stops = []

        while start < (N_l - 1):
            starts.append(start)
            stops.append(stop)
            start = stop
            stop = start + step_size
            stop = min(stop, N_l - 1)
    return starts, stops


def load_band(step_kind, N_k, v_pad=20):
    starts, stops = get_steps(step_kind)
    starts = starts[1:]
    stops = stops[1:]
    template = '_{}_'.format(step_kind)
    template += '{}_'
    template += '{}_{}.np'.format(N_k, v_pad)
    U_template = solution_location + 'U' + template
    V_template = solution_location + 'V' + template

    ppd_wcs = wcs_defs.get_ppd_wcs()
    l_inds, b_inds, d_inds, v_inds = get_field_inds('band')
    _, b_vals, _ = ppd_wcs.wcs_pix2world(0, b_inds, 0,
                                         wcs_defs.origin)
    v_wcs = wcs_defs.get_v_wcs()
    v_vals = v_wcs.wcs_pix2world(v_inds, wcs_defs.origin)

    ppd_cubes = []
    lsr_moment_cubes = []
    flat_rotation_cubes = []
    clemens_rotation_cubes = []
    l_gs = []
    d_gs = []

    for stripe_i, (start, stop) in enumerate(zip(starts, stops)):
        U = np.load(U_template.format(stripe_i))
        V = np.load(V_template.format(stripe_i))
        s_l_inds = l_inds[start:stop]
        spatial_shape = [b_inds.size, s_l_inds.size]
        veldist_shape = [d_inds.size, v_inds.size]
        full_shape = (spatial_shape[0], spatial_shape[1],
                      veldist_shape[0], veldist_shape[1])
        print (l_inds.size, start, stop)
        l_g, d_g = np.meshgrid(s_l_inds, d_inds, indexing='ij')
        l_g, _, mu_g = ppd_wcs.wcs_pix2world(l_g, 0, d_g, 
                                             wcs_defs.origin)
        d_g = distmod_to_distance(mu_g)
        l_gs.append(l_g)
        d_gs.append(d_g)

        ppd_cube = ppd_cube_from_decomposition(U, V, full_shape)
        ppd_cubes.append(ppd_cube)

        lsr_moment_cube = build_velocity_moment_3cube_from_components(U, spatial_shape, V, veldist_shape, v_inds)
        lsr_moment_cubes.append(lsr_moment_cube)

        flat_rotation_cube = build_rotation_curve_3cube(s_l_inds, b_inds, d_inds)
        flat_rotation_cubes.append(flat_rotation_cube)

        clemens_rotation_cube = build_clemens_rotation_curve_3cube(s_l_inds, b_inds, d_inds)
        clemens_rotation_cubes.append(clemens_rotation_cube)
    ppd_cube = np.concatenate(ppd_cubes, axis=0)
    lsr_moment_cube = np.concatenate(lsr_moment_cubes, axis=0)
    flat_rotation_cube = np.concatenate(flat_rotation_cubes, axis=0)
    clemens_rotation_cube = np.concatenate(clemens_rotation_cubes, axis=0)
    l_g = np.concatenate(l_gs, axis=0)
    d_g = np.concatenate(d_gs, axis=0)
    position_grids = [l_g, d_g, b_vals, v_vals]
    return ppd_cube, lsr_moment_cube, flat_rotation_cube, clemens_rotation_cube, position_grids


def extract_value(cube, l_value, b_value, d_value, 
                  l_inds, b_inds, d_inds):
    ppdv_wcs = wcs_defs.get_ppdv_wcs()
    mu_value = distance_to_distmod(d_value)
    l_ind, b_ind, d_ind, _ = ppdv_wcs.wcs_world2pix(l_value, 
                                                    b_value, 
                                                    mu_value, 
                                                    0, 
                                                    wcs_defs.origin)
    l_ind = np.round(l_ind).astype('int')
    b_ind = np.round(b_ind).astype('int')
    d_ind = np.round(d_ind).astype('int')
    if (l_ind in l_inds) & (b_ind in b_inds) & (d_ind in d_inds):
        l_ind -= l_inds.min()
        b_ind -= b_inds.min()
        d_ind -= d_inds.min()
        value = cube[l_ind, b_ind, d_ind]
    else:
        value = np.nan
    return value


def extract_spline_value(cube, l_value, b_value, d_value,
                  l_inds, b_inds, d_vals):
    ppdv_wcs = wcs_defs.get_ppdv_wcs()
    l_ind, b_ind, d_ind, _ = ppdv_wcs.wcs_world2pix(l_value, 
                                                    b_value, 
                                                    0,
                                                    0, 
                                                    wcs_defs.origin)
    l_ind = np.round(l_ind).astype('int')
    b_ind = np.round(b_ind).astype('int')
    if (l_ind in l_inds) & (b_ind in b_inds):
        l_ind -= l_inds.min()
        b_ind -= b_inds.min()
        value = interpolate.spline(d_vals, cube[l_ind, b_ind, :], d_value,
                                   order=1)
    else:
        value = np.nan
    return value


def extract_profile(cube, l_value, b_value, 
                  l_inds, b_inds):
    ppdv_wcs = wcs_defs.get_ppdv_wcs()
    l_ind, b_ind, d_ind, _ = ppdv_wcs.wcs_world2pix(l_value, 
                                                    b_value, 
                                                    0,
                                                    0, 
                                                    wcs_defs.origin)
    l_ind = np.round(l_ind).astype('int')
    b_ind = np.round(b_ind).astype('int')
    if (l_ind in l_inds) & (b_ind in b_inds):
        l_ind -= l_inds.min()
        b_ind -= b_inds.min()
        value = cube[l_ind, b_ind, :]
    else:
        value = np.nan
    return value


if __name__=='__main__':
    import matplotlib.pyplot as plt
    import os
    #import argparse
    #parser = argparse.ArgumentParser()
    #parser.add_argument('field')
    #parser.add_argument('-Nk', default=10, type=int)
    #args = parser.parse_args()
    #field = args.field
    #N_k = args.Nk
    
    if os.path.isdir('/Users/K/'):
        subset_dir = '/Users/K/Dropbox/KTNMF_local/RealData/subsets/'
        solution_location = '/Users/K/Dropbox/KTNMF_local/RealData/band_solutions/'
    else:
        subset_dir = '/user/ktchernyshyov/RealData/subsets/'
        solution_location = '/user/ktchernyshyov/RealData/band_solutions/'

    if False:
        #stuff = load_field(field, N_k)
        step_kind = 'he'
        N_k = 10
        stuff = load_band(step_kind, N_k, v_pad=20)
        ppd, lsr_moment_cube, flat_rotation_cube, clemens_rotation_cube, position_grids = stuff
        l_g, d_g, b_vals, v_vals = position_grids

        b_min = -20
        b_max = 20

    l_inds, b_inds, d_inds, v_inds = get_field_inds('band')
    b_inds = b_inds#[65:95]
    #b_inds = b_inds[70:91] #(narrow)
    #b_inds = b_inds[50:111] #(medium)
    #l_inds = l_inds[1050:1350]
    l_inds = l_inds[:-1]
    d_inds = d_inds[:]
    ppdv_wcs = wcs_defs.get_ppdv_wcs()
    l_g, b_g, d_g = np.meshgrid(l_inds, b_inds, d_inds, indexing='ij')
    l_g, b_g, d_g, _ = ppdv_wcs.wcs_pix2world(l_g, b_g, d_g, 0, 
                                            wcs_defs.origin)
    d_g = distmod_to_distance(d_g)
    _, b_vals, _, _ = ppdv_wcs.wcs_pix2world(0, b_inds, 0, 0, 
                                             wcs_defs.origin)

    b_min = -1.5
    b_max = 1.5

    if False:
        fname = solution_location + 'moment_cube_vrot_220_dmax_15_vpad_30_narrow.fits'
    else:
        fname = solution_location + 'gd_band_smooth2_moment.fits'
    with fits.open(fname) as hdu:
        print ("Loading moment cube")
        lsr_moment_cube = hdu[0].data#[:, 65:95, :]
    if False:
        fname = solution_location + 'ppd_cube_vrot_220_dmax_15_vpad_30_narrow.fits'
    else:
        fname = subset_dir + 'EBV_band.fits'
    with fits.open(fname) as hdu:
        print ("Loading column cube")
        ppd = hdu[0].data
        ppd = ppd.swapaxes(0, 1)#[:, 65:95, :]

    print ("Building rotation cubes")

    if False:
        b_range, = np.where((-2.5 <= b_vals) & (b_vals <= 2.5))
        l_g = l_g[:, b_range]
        b_g = b_g[:, b_range]
        d_g = d_g[:, b_range]
        #make some xarray structures for easier loading
        coords = dict(GLON=l_g[:, 0, 0], GLAT=b_g[0, :, 0], 
                      distance=d_g[0, 0, :])

        smooth_gd = (('GLON', 'GLAT', 'distance'), 
                     lsr_moment_cube[:, b_range])

        flat_220_IAU = rotation_curve(l_g, b_g, d_g, V_rot=220., R_sol=8.1)
        flat_220_IAU = (('GLON', 'GLAT', 'distance'), flat_220_IAU)

        clemens = build_clemens_rotation_curve_3cube(l_inds, 
                                                     b_inds[b_range], 
                                                     d_inds)
        clemens = (('GLON', 'GLAT', 'distance'), clemens)

        """
        Flat IAU parameters:
            R0 = 8.1
            Vc = 220.
            U = 10.3
            V = 15.3
            W = 7.7
        
        Bovy parameters:
            R0 = 8.1
            Vc = 218.
            U = 10.5
            V = 23.9
            W = 7.7

        Reid parameters:
            R0 = 8.34
            Vc = 240.
            U = 10.7
            V = 15.6
            W = 8.9
        """
        bovy = rotation_curve(l_g, b_g, d_g, V_rot=218., R_sol=8.1)
        bovy -= lsr_shift(l_g, b_g, 10.5, 23.9, 7.7)
        bovy = (('GLON', 'GLAT', 'distance'), bovy)

        reid = rotation_curve(l_g, b_g, d_g, V_rot=240., R_sol=8.34)
        reid -= lsr_shift(l_g, b_g, 10.7, 15.6, 8.9)
        reid = (('GLON', 'GLAT', 'distance'), reid)

        ppd = (('GLON', 'GLAT', 'distance'), ppd[:, b_range])

        data_vars = dict(smooth_gd=smooth_gd, 
                         flat_220_IAU=flat_220_IAU, 
                         clemens=clemens, 
                         bovy=bovy, 
                         reid=reid,
                         ppd=ppd)
        kinematic_sets = xarray.Dataset(data_vars=data_vars, coords=coords)
        kinematic_sets.to_netcdf('../../kinematic_sets.nc')

    U_old = 10.3
    V_old = 15.3
    W_old = 7.7
    U_new = 10.7
    V_new = 15.6
    W_new = 8.9
    l_rad = np.radians(l_g)
    b_rad = np.radians(b_g)
    lsr_correction = ((W_new - W_old) * np.sin(l_rad) + np.cos(b_rad) * 
                      ((U_new - U_old) * np.cos(l_rad) + 
                      (V_new - V_old) * np.sin(l_rad)))
    lsr_moment_cube += lsr_correction
    flat_rotation_cube = flat_rotation_cube
    print ("Interpreting masers")
    try:
        reid_masers = pd.read_pickle('../../reid_wloc_meta.pi')
    except:
        reid_masers = pd.read_pickle('/user/ktchernyshyov/RealData/reid_masers.pi')
    #reid_masers['v_rot_cl'] = clemens_rotation_curve(reid_masers.GLON, reid_masers.GLAT, reid_masers.distance)
    reid_masers['v_rot_flat'] = rotation_curve(reid_masers.GLON, reid_masers.GLAT, reid_masers.distance)

    moments_at_masers = []
    moment_e_estimates = []
    ppd_at_masers = []
    moment_profiles = []
    ppd_profiles = []
    for e, row in reid_masers.iterrows():
        moment = extract_spline_value(lsr_moment_cube, 
                               row.GLON, row.GLAT, row.distance, 
                               l_inds, b_inds, d_g[0,0])
        dist_points = np.linspace(row.distance * 0.9, row.distance * 1.1, 
                                  30)
        moment_e = extract_spline_value(lsr_moment_cube, 
                               row.GLON, row.GLAT, dist_points, 
                               l_inds, b_inds, d_g[0,0])
        moment_e = np.nanmedian(np.abs(moment_e - moment))
        #moment_profile = extract_profile(lsr_moment_cube - 
        #                                 flat_rotation_cube, 
        #                                 row.GLON, row.GLAT,
        #                                 l_inds, b_inds)
        ppd_val = extract_value(ppd, 
                               row.GLON, row.GLAT, row.distance, 
                               l_inds, b_inds, d_g[0,0])
        #ppd_profile = extract_profile(ppd,
        #                              row.GLON, row.GLAT, 
        #                              l_inds, b_inds)
        moments_at_masers.append(moment)
        moment_e_estimates.append(moment_e)
        #moment_profiles.append(moment_profile)
        ppd_at_masers.append(ppd_val)
        #ppd_profiles.append(ppd_profile)

    #moment_e_estimates = np.ma.masked_where(np.isnan(moments_at_masers),
    #                                        moment_e_estimates)
    #moments_at_masers = np.ma.masked_where(np.isnan(moments_at_masers), 
    #                                       moments_at_masers)
    reid_masers['V_est'] = np.asarray(moments_at_masers)
    reid_masers['V_est_e'] = moment_e_estimates 
    reid_masers['V_est_e'] *= 1.48 #assuming ~approximate normality
    resid = reid_masers.V_LSR - reid_masers.V_est
    sd = np.sqrt(reid_masers.V_est_e**2 + reid_masers.V_LSR_e**2)
    reid_masers['st_resid'] = resid / sd
    assert False

    if False:
        plt.figure(figsize=[18, 5])
        plt.subplots_adjust(wspace=0.02, left=0.05, right=0.96, top=0.98)
        s = 50
        x_ticks = [-80, -60, -40, -20, 0, 20, 40]
        linewidth = 2.5
        axes = []
        ax = plt.subplot(1, 3, 1, aspect='equal')
        axes.append(ax)
        ax.set_xlim(-50, 50)
        ax.set_ylim(-100, 60)
        ax.set_yticks(x_ticks)
        plt.plot([-80, 80], [-80, 80], color='red', 
                 linewidth=1.5, scalex=False, scaley=False, 
                 zorder=1)
        plt.scatter(reid_masers.v_rot_cl - reid_masers.v_rot_flat.values,
                    reid_masers.V_LSR - reid_masers.v_rot_flat,
                    zorder=2, color='gray', s=s, marker='o', 
                    linewidth=linewidth)
        plt.ylabel('Maser velocity - flat rotation', fontsize=16)
        plt.xlabel('Clemens (1985) - flat rotation', fontsize=16)

        ax = plt.subplot(1, 3, 2, aspect='equal')
        axes.append(ax)
        ax.set_xlim(-50, 50)
        ax.set_ylim(-100, 60)
        ax.set_yticks(x_ticks)
        ax.set_yticklabels([])
        plt.plot([-80, 80], [-80, 80], color='red', 
                 linewidth=1.5, scalex=False, scaley=False, 
                 zorder=1)
        plt.scatter(reid_masers.gd_moment - reid_masers.v_rot_flat.values,
                    reid_masers.V_LSR - reid_masers.v_rot_flat,
                    zorder=2, color='gray', s=s, marker='o', 
                    linewidth=linewidth)
        plt.xlabel('Unsmoothed KT - flat rotation', fontsize=16)

        ax = plt.subplot(1, 3, 3, aspect='equal')
        axes.append(ax)
        ax.set_xlim(-50, 50)
        ax.set_ylim(-100, 60)
        ax.set_yticks(x_ticks)
        ax.yaxis.set_tick_params(labelleft=False,
                                 labelright=True)
        plt.plot([-80, 80], [-80, 80], color='red', 
                 linewidth=1.5, scalex=False, scaley=False, 
                 zorder=1)
        plt.scatter(moments_at_masers - reid_masers.v_rot_flat.values,
                    reid_masers.V_LSR - reid_masers.v_rot_flat,
                    zorder=2, color='gray', s=s, marker='o', 
                    linewidth=linewidth)
        plt.xlabel('Smoothed KT - flat rotation', fontsize=16)

    if False:
        moments = []
        rotation_curves = []
        b_means = []
        matched_masers = []
        ppdv_wcs = wcs_defs.get_ppdv_wcs()
        lon_vals = np.linspace(130, 140, 2)
        o_lon_inds, _ = ppdv_wcs.sub(2).all_world2pix(lon_vals, 0, 0)
        o_lon_inds = np.round(o_lon_inds).astype(np.int)[::-1]
        lon_inds = o_lon_inds - l_inds.min()
        lon_vals = lon_vals[::-1]
        ma_lsr_moment_cube = np.ma.masked_where(np.isnan(lsr_moment_cube), lsr_moment_cube)
        _, _, _, v_vals = ppdv_wcs.wcs_pix2world(0, 0, 0, v_inds, 1)
        for i in range(lon_vals.size - 1):
            lsr_subset = ma_lsr_moment_cube[lon_inds[i]:lon_inds[i+1]]
            rot_subset = flat_rotation_cube[lon_inds[i]:lon_inds[i+1]]
            ppd_subset = ppd[lon_inds[i]:lon_inds[i+1]]
            moment = np.ma.average(lsr_subset, axis=(0, 1), 
                                   weights=ppd_subset)
            rotation = np.ma.average(rot_subset, axis=(0, 1),
                                     weights=ppd_subset)
            moments.append(moment)
            rotation_curves.append(rotation)
            b_mean = np.ma.average(lsr_subset, axis=1,
                                   weights=ppd_subset)
            b_means.append(b_mean)
            lon_lower = lon_vals[i + 1]
            lon_upper = lon_vals[i]
            maser_subset = reid_masers.query('(@lon_lower <= GLON) & +'
                                             '(GLON < @lon_upper)')
            matched_masers.append(maser_subset)

        #import seaborn
        #seaborn.set_style('whitegrid')
        #seaborn.set_context('talk')
        N_sg = len(moments)
        print ("Plotting masers")
        plt.figure(figsize=[9, 9])
        colors = {}
        colors['...'] = 'white'
        colors['3-k'] = 'yellow'
        colors['4-k'] = 'yellow'
        colors['Con'] = 'yellow'
        colors['Loc'] = 'blue'
        colors['Out'] = 'red'
        colors['Per'] = 'black'
        colors['Sct'] = 'cyan'
        colors['Sgr'] = 'magenta'
        for i in range(N_sg):
            ax = plt.subplot(1, 1, i+1)
            plt.xlim(0.2, 9.7)
            if True:
                plt.plot(d_g[0, 0], rotation_curves[i], color='black', 
                         linestyle='dashed', zorder=2, linewidth=1.5)
                plt.plot(d_g[0, 0], b_means[i].T, color='gray', alpha=0.3, zorder=1,
                         linewidth=0.5)
                plt.plot(d_g[0, 0], moments[i], color='black', zorder=3, 
                         linewidth=2)
                plt.ylim(-35 + rotation_curves[i].min(),
                         15 + rotation_curves[i].max())
            else:
                plt.plot(d_g[0, 0], (b_means[i] - rotation_curves[i]).T, 
                         color='gray', alpha=0.3, zorder=1, linewidth=0.5)
                plt.plot(d_g[0, 0], moments[i] - rotation_curves[i], 
                         color='black', zorder=3, linewidth=2)
                plt.axhline(0, 0, color='black', linestyle='dashed',
                            zorder=2, linewidth=1.5)
                plt.ylim(-35, 15)
            for e, row in matched_masers[i].iterrows():
                color = colors[row.Arm]
                print (color, row.Arm)
                if False:
                    plt.scatter(row.distance, row.V_LSR,
                                color=color, marker='s', s=40, zorder=4, 
                                edgecolor='black')
                else:
                    plt.scatter(row.distance, row.V_LSR - row.v_rot_flat,
                                color=color, marker='s', s=40, zorder=4, 
                                edgecolor='black')
            ax.set_yticks([-20, -10, 0, 10, 20])
            plt.text(0.6, 0.8, 
                     r'${:.0f}^\circ \! < \ell < {:.0f}^\circ$'.format(lon_vals[i + 1], 
                      lon_vals[i]),
                      fontsize=15, 
                      va='center',
                      bbox=dict(facecolor='white', edgecolor='white', 
                                alpha=0.75),
                      transform=ax.transAxes)
            if i < (N_sg - 3):
                ax.set_xticklabels([])
            if (i % 3) != 0 :
                ax.set_yticklabels([])

    if False:
        plt.figure(figsize=[8, 11])
        non_nan_inds, = np.where(moments_at_masers.mask == False)
        N_non_nan = non_nan_inds.size
        N_rows = 6
        N_cols = 5
        for i in range(N_non_nan):
            ax = plt.subplot(N_rows, N_cols, i+1)
            n_i = non_nan_inds[i] 
            row = reid_masers.iloc[n_i]; 
            plt.plot(d_g[0,0], moment_profiles[n_i], color='black')
            plt.errorbar(row.distance, row.V_LSR - row.v_rot_flat, 
                        color='red', markersize=30, 
                        yerr=row.V_LSR_e)
            plt.errorbar(row.distance, moments_at_masers[n_i] - 
                        row.v_rot_flat, color='black', markersize=15,
                        yerr=moment_e_estimates[n_i])
            plt.axhline(0, color='black', linestyle='dashed')
            plt.xlim(0.9, 20)
            plt.xscale('log')
            plt.ylim(-40, 40)
            ax.set_yticks([-40, -20, 0, 20, 40])
            if (i % N_cols) != 0:
                ax.set_yticklabels([])
            if (i // N_rows) < (N_cols - 1):
                ax.set_xticklabels([])