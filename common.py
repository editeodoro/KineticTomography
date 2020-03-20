from __future__ import division
import numpy as np
from astropy.io import fits
import shape_ops
import re
from scipy.stats import pearsonr


small_constant = 1.e-20


def pearson(truth, recovered):
    flat_truth = truth.ravel()
    flat_recovered = recovered.ravel()
    return pearsonr(flat_truth, flat_recovered)[0]


def mean_normalized_square_discrepancy(truth, recovered):
    flat_truth = truth.ravel()
    flat_recovered = recovered.ravel()
    discrepancy = np.linalg.norm(flat_truth - flat_recovered, ord=2)
    discrepancy /= np.linalg.norm(flat_truth)
    return discrepancy


def support_discrepancies(truth, recovered):
    #clip recovered, since it's been filled with small nonzero values
    recovered_mask = recovered > small_constant
    truth_mask = truth > 0
    
    difference = truth_mask - recovered_mask

    fraction_missing = np.sum(difference == 1) / truth.size
    fraction_extra = np.sum(difference == -1) / truth.size
    return fraction_missing, fraction_extra


def component_number_discrepancy(case_args, recovered_component_list):
    return case_args[0] - (recovered_component_list.shape[1] - 1)


def fractional_sparsity(array):
    return np.count_nonzero(array) / array.size


def underdetermination(spatial_shape, veldist_shape):
    data_size = (spatial_shape[0] * spatial_shape[1] * 
                 (veldist_shape[0] + veldist_shape[1]))
    model_size = (spatial_shape[0] * spatial_shape[1] + 
                  veldist_shape[0] * veldist_shape[1]) * spatial_shape[2]
    return data_size / model_size


def case_from_args(args):
    fstr = 'ind_NCL{:03d}_CFMN{:03d}_CFMX{:03d}_DSIG{:03d}_VSIG{:03d}_I{:03d}'
    try:
        case = fstr.format(args.NCL, args.CFMN, args.CFMX, 
                           args.DSIG, args.VSIG, args.I)
    except:
        case = fstr.format(*args)
    return case


def args_from_case(case):
    NCL = int(re.search('NCL(\d+)', case).group()[3:])
    CFMN = int(re.search('CFMN(\d+)', case).group()[4:])
    CFMX = int(re.search('CFMX(\d+)', case).group()[4:])
    DSIG = int(re.search('DSIG(\d+)', case).group()[4:])
    VSIG = int(re.search('VSIG(\d+)', case).group()[4:])
    ID = int(re.search('I(\d+)', case).group()[1:])
    return NCL, CFMN, CFMX, DSIG, VSIG, ID


def load_truth(truth_dir, case):
    spatial_list_name = truth_dir + case + '_spatial_list.fits'
    veldist_list_name = truth_dir + case + '_veldist_list.fits'

    with fits.open(spatial_list_name, memmap=False) as fits_file:
        #hard convert to double to avoid bit order issues
        spatial_jp_truth = np.array(fits_file[0].data, dtype=np.double)
        spatial_truth, spatial_shape = shape_ops.jp_array_to_kt_array(spatial_jp_truth)


    with fits.open(veldist_list_name, memmap=False) as fits_file:
        #hard convert to double to avoid bit order issues
        veldist_jp_truth = np.array(fits_file[0].data, dtype=np.double)
        veldist_truth, veldist_shape = shape_ops.jp_array_to_kt_array(veldist_jp_truth)

    return spatial_truth, spatial_shape, veldist_truth, veldist_shape


def prepare_test(truth_dir, case):
    spatial_truth, spatial_shape, veldist_truth, veldist_shape = load_truth(truth_dir, case)
    N_y, N_x, N_comp = spatial_shape
    N_d, N_v, _ = veldist_shape
    projections = shape_ops.get_projections(spatial_truth,
                                            spatial_shape,
                                            veldist_truth,
                                            veldist_shape)
    full_shape = [N_y, N_x, N_d, N_v]
    return projections, full_shape


def load_recovery(recovery_dir, case):
    spatial_rec_name = recovery_dir + case + '_spatial_rec_fista.fits'
    veldist_rec_name = recovery_dir + case + '_veldist_rec_fista.fits'

    #munge data
    with fits.open(spatial_rec_name, memmap=False) as fits_file:
        #hard convert to double to avoid bit order issues
        spatial_rec = np.array(fits_file[0].data, dtype=np.double)

    with fits.open(veldist_rec_name, memmap=False) as fits_file:
        #hard convert to double to avoid bit order issues
        veldist_rec = np.array(fits_file[0].data, dtype=np.double)

    return spatial_rec, veldist_rec


def save_recovery(spatial_recovery, veldist_recovery, recovery_dir, case):
    spatial_rec_name = recovery_dir + case + '_spatial_rec_fista.fits'
    veldist_rec_name = recovery_dir + case + '_veldist_rec_fista.fits'

    hdu = fits.PrimaryHDU(data=spatial_recovery)
    hdu.writeto(spatial_rec_name, clobber=True)

    hdu = fits.PrimaryHDU(data=veldist_recovery)
    hdu.writeto(veldist_rec_name, clobber=True)


def decomposition_to_4cube(spatial, spatial_shape, veldist, veldist_shape):
    flat_4cube = np.dot(spatial, veldist.T)
    unraveled_4cube = flat_4cube.reshape([spatial_shape[0],
                                         spatial_shape[1],
                                         veldist_shape[0],
                                         veldist_shape[1]])
    return unraveled_4cube


def compute_cm_3cube(_4cube):
    """

    """
    v = np.arange(_4cube.shape[3])
    v -= v.mean()

    cm_cube = np.sum(_4cube * v[None, None, None, :], axis=3).cumsum(axis=2)
    return cm_cube


def compute_cm_3cube_from_components(spatial, spatial_shape, 
                                     veldist, veldist_shape):
    """

    """
    v = np.arange(veldist_shape[1])
    v -= v.mean()

    m_components = veldist.reshape([veldist_shape[0], 
                                   veldist_shape[1], 
                                   -1])
    cm_components = (m_components * v[None, :, None]).sum(axis=1).cumsum(axis=0)
    cm_cube = np.dot(spatial, cm_components.T).reshape([spatial_shape[0],
                                                       spatial_shape[1],
                                                       veldist_shape[0]])
    return cm_cube


def compute_1m_3cube(_4cube):
    """

    """
    v = np.arange(_4cube.shape[3])
    v -= v.mean()

    m_cube = np.sum(_4cube * v[None, None, None, :], axis=3)
    return m_cube


def compute_1m_3cube_from_components(spatial, spatial_shape, 
                                     veldist, veldist_shape):
    """

    """
    #v_min = -319.8 - 1.3
    #delta_v = 1.3
    #N_v = 493
    #v_max = v_min + N_v * delta_v
    #v = np.linspace(v_min, v_max, N_v)
    v = np.arange(veldist_shape[1])
    v -= v.mean()

    m_components = veldist.reshape([veldist_shape[0], 
                                   veldist_shape[1], 
                                   -1])
    m_components = (compute_1m_3cube_from_components * v[None, :, None]).sum(axis=1)
    m_cube = np.dot(spatial, m_components.T).reshape([spatial_shape[0],
                                                     spatial_shape[1],
                                                     veldist_shape[0]])
    return m_cube


def stack_projections(spatial, spatial_shape, veldist, veldist_shape):
    """

    """
    N_x, N_y = spatial_shape
    N_d, N_v = veldist_shape
    N_k = spatial.shape[1]

    veldist_2d = veldist.reshape([N_d, N_v, N_k])
    dist = veldist_2d.sum(axis=1)
    vel = veldist_2d.sum(axis=0)

    proj_components = np.zeros([N_x * N_y,
                               N_d + N_v,
                               N_k])

    for k_i in range(N_k):
        proj_components[:, 0:N_d, k_i] = spatial[:, None, k_i] * dist[None, :, k_i]
        proj_components[:, N_d:, k_i] = spatial[:, None, k_i] * vel[None, :, k_i]

    proj_components = proj_components.reshape([N_x, N_y, N_d+N_v, N_k])
    return proj_components
