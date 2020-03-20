from __future__ import division
import numpy as np
from scipy import sparse
import numexpr as ne


def jp_array_to_kt_array(jp_array):
    """
        Turn a (components)-by-(A)-by-(B) matrix into a 
        (A)*(B)-by-(components) matrix. 
        Maintain last-axis-fast ordering.
        Return reshaped matrix and [(A), (B), (components)].

    """
    kt_array = np.transpose(jp_array, (1,2,0))
    kt_shape = kt_array.shape
    raveled_kt_array = kt_array.reshape(-1, kt_array.shape[2])
    return raveled_kt_array, kt_shape


def get_projections(spatial_array, spatial_shape, veldist_array, veldist_shape):
    """
        Assumes kt shape convention
    """
    raveled_4cube = np.dot(spatial_array, veldist_array.T)
    unraveled_4cube = raveled_4cube.reshape([-1, veldist_shape[0], veldist_shape[1]])

    projections = np.zeros([spatial_shape[0] * spatial_shape[1], veldist_shape[0] + veldist_shape[1]])
    projections[:,0:veldist_shape[0]] = unraveled_4cube.sum(axis=2)
    projections[:,veldist_shape[0]:] = unraveled_4cube.sum(axis=1)

    return projections


def grid_adjacency_matrix(N_x, N_y, 
                          max_xpix=1, max_ypix=1, 
                          xscale=2., yscale=2.,
                          pad=False):
    """
       Generate the (max_xpix, max_ypix)-step adjacency matrix for a
       raveled N_x*N_y grid. Assumes y is the fast axis.
    """

    if pad:
        N_x = N_x + 1
        N_y = N_y + 1

    N_dpts = N_x * N_y

    adjacency_matrix = sparse.lil_matrix((N_dpts, N_dpts))
    base_x, base_y = np.meshgrid(np.arange(N_x), np.arange(N_y))

    base_x = base_x.flatten()
    base_y = base_y.flatten()

    for dx in range(max_xpix + 1):
        for dy in range(max_ypix + 1):
            if (dx==0) and (dy==0):
                pass
            else:
                add_shifted_weight(adjacency_matrix, (base_x, base_y),
                   (dx, dy),
                   N_x, N_y,
                   weight=np.exp(-0.5*((dx/xscale)**2 + (dy/yscale)**2)),
                   symmetric=True)

    if pad:
        adjacency_matrix[:, -1] = 0.
        adjacency_matrix[-1, :] = 0.

    adjacency_matrix = adjacency_matrix.tocsc()
    adjacency_laplacian, degree_marix = sparse.csgraph.laplacian(adjacency_matrix, return_diag=True, normed=False)
    return adjacency_laplacian


def add_shifted_weight(adjacency_matrix, base, shift, N_x, N_y, weight=1.,
                       symmetric=True):
    shifted_x = base[0] + shift[0]
    shifted_y = base[1] + shift[1]

    within_bounds = (shifted_x < N_x)
    within_bounds *= (shifted_x >= 0)
    within_bounds *= (shifted_y < N_y)
    within_bounds *= (shifted_y >= 0)

    base_1d = np.ravel_multi_index([base[0][within_bounds], base[1][within_bounds]], [N_x, N_y])

    shifted_1d = np.ravel_multi_index([shifted_x[within_bounds], shifted_y[within_bounds]], [N_x, N_y])

    adjacency_matrix[base_1d, shifted_1d] = weight
    if symmetric:
        adjacency_matrix[shifted_1d, base_1d] = weight


def orthogonal_projection(N_x, N_y, pad=False):
    """

        Generate the projection matrix from a raveled Nx*Ny vector to a 
        stapled Nx+Ny vector. Assumes y is the fast axis.

    """

    if pad:
        N_x_f = N_x + 1
        N_y_f = N_y + 1
    else:
        N_x_f = N_x
        N_y_f = N_y

    N_samples = (N_x + N_y)
    N_features = (N_x_f * N_y_f)

    design = sparse.lil_matrix((N_samples, N_features))

    x_is = np.arange(N_x)

    for y_i in xrange(N_y):
        feature_ind = np.ravel_multi_index([x_is, y_i], [N_x_f, N_y_f])

        obs_ind = N_x + y_i
        try:
            design[obs_ind, feature_ind] = 1.
        except:
            print (y_i, obs_ind)
            print (feature_ind)
            assert False

    y_is = np.arange(N_y)

    for x_i in xrange(N_x):
        feature_ind = np.ravel_multi_index([x_i, y_is], [N_x_f, N_y_f])

        obs_ind = x_i
        try:
            design[obs_ind, feature_ind] = 1.
        except:
            print (x_i, obs_ind)
            print (feature_ind)
            assert (False)

    if pad:
        feature_ind = np.ravel_multi_index([x_is, N_y_f - 1], 
                                           [N_x_f, N_y_f])
        design[x_is, feature_ind] = 1.
        feature_ind = np.ravel_multi_index([N_x_f - 1, y_is], 
                                           [N_x_f, N_y_f])
        design[N_x + y_is, feature_ind] = 1.

    design = design.tocsr()
    return design


def x_deltas_y_gaussians(N_x, N_y, y_cens, y_sigmas, threshmin=1.e-3):
    """

        Gaussians along the fast (y) axis, delta functions/width-1 tophats
        along the slow (x) axis.

    """
    N_samples = N_x * N_y
    N_y_cens = len(y_cens)
    N_y_sigmas = len(y_sigmas)
    N_features = N_x * N_y_cens * N_y_sigmas

    x_axis = np.arange(N_x)
    y_axis = np.arange(N_y)

    design = sparse.lil_matrix((N_samples, N_features))

    #velocity part
    y_cens_g, y_sigmas_g = np.meshgrid(y_cens, y_sigmas, indexing='ij')
    y_cens_g = y_cens_g.reshape([1, -1])
    y_sigmas_g = y_sigmas_g.reshape([1, -1])
    y_axis = y_axis.reshape([N_y, 1])
    consts = 1. / (y_sigmas_g * np.sqrt(2. * np.pi))

    velo_part = ne.evaluate('consts * exp(-0.5 * ((y_cens_g - y_axis) / y_sigmas_g)**2)')

    velo_part = ne.evaluate('where(velo_part > threshmin, velo_part, 0.)')

    cen_sigma_axis = np.arange(N_y_cens * N_y_sigmas)

    for x_i in range(N_x):
        for cen_sigma_i in range(cen_sigma_axis.size):
            obs_i = np.ravel_multi_index([x_i, y_axis], [N_x, N_y])
            feature_i = np.ravel_multi_index([x_i, cen_sigma_i],
                                             [N_x, N_y_cens * N_y_sigmas])
            design[obs_i, feature_i] = velo_part[:, cen_sigma_i, None]

    design = design.tocsr()
    return design
