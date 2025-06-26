from .spatial_qmodel_sparse import spatial_qmodel_sparse

import numpy as np
from ..interaction import cxx_spin_mapping as csm
from .ijc import ijc_to_Jh
from .qmodel_vanilla import qmodel_vanilla


def qmodel_from_J_and_h(J, h, const=0):
    """
    Construct a quadratic objective function from matrix representation (J, h).
    The matrix representation uses a symmetric coupling matrix J and bias vector h to define 
    the quadratic model in its standard form.

    Parameters
    ----------
    J : np.ndarray
        A symmetric NxN coupling matrix where J[i,j] represents the coupling strength 
        between spins i and j (the interaction matrix)
    h : np.ndarray
        A vector of length N containing the bias terms for each spin (the bias vector)
    const : float, optional
        The constant term in the function, by default 0

    Returns
    -------
    qmodel_vanilla
        The constructed quadratic function using the matrix representation.
    """
    N = J.shape[0]
    return qmodel_vanilla(N, J, h, const=const)


def qmodel_from_ijc(N, M, i_list, j_list, c_list, const=0):
    """
    Construct a quadratic objective function from list representation (i, j, c).
    The list representation uses lists of indices i, j and corresponding coefficients c 
    to define sparse interactions between spins.
    
    Parameters
    ----------
    N : int
        The number of spins 
    M : int
        The number of interactions (length of the index lists)
    i_list : array-like
        List of row indices for non-zero couplings and biases. For coupling terms,
        represents the first spin index. Must be of length M.
    j_list : array-like
        List of column indices for non-zero couplings. For coupling terms,
        represents the second spin index. Use -1 for bias terms. Must be of length M.
    c_list : array-like
        List of corresponding coupling/bias coefficients. Must be of length M.
    const : float, optional
        The constant term in the function, by default 0

    Returns
    -------
    qmodel_vanilla
        The constructed quadratic function. The list representation
        is converted to matrix representation internally.
    """
    J, h = ijc_to_Jh(N, i_list, j_list, c_list)
    return qmodel_vanilla(N, J, h, const=const)


def spatial_qmodel_from_masked_J_and_h(mask, J_arr, h_arr, xi_arr, **kwargs):
    """
    Construct a quadratic objective function with spatial coupling matrix (spCM).
    This constructor uses a non-periodic spatial interaction function where the spin 
    positions are specified via a mask array.

    Parameters
    ----------
    mask : np.ndarray
        A 2D array specifying the existence of spins in each position.
        Non-zero values indicate spin positions.
    J_arr : np.ndarray
        A 2D array representing the spatial interaction function.
        Must have the same shape as mask.
    h_arr : np.ndarray
        A 2D array containing the bias terms for each position.
        Must have the same shape as mask.
    xi_arr : np.ndarray
        A 2D array of coefficients for the spatial coupling matrix.
        Must have the same shape as mask.
    kwargs:
        Additional parameters to be passed to the constructor of spatial_qmodel_sparse

    Returns
    -------
    spatial_qmodel_sparse
        The constructed sparse spatial quadratic function.
    """

    # get spin positions from mask
    assert (mask.shape == J_arr.shape)
    assert (mask.shape == h_arr.shape)
    Ly, Lx = mask.shape
    pos = []
    h = []
    assert (np.all(mask >= 0))
    for i in np.arange(Ly):
        for j in np.arange(Lx):
            if mask[i, j] > 0:
                pos.append([i, j])
    pos = np.array(pos, dtype="i")
    N = pos.shape[0]

    # pick parameters for each spins
    h = csm.spinarr_to_spinvector(Ly, Lx, pos, h_arr)
    xi = csm.spinarr_to_spinvector(Ly, Lx, pos, xi_arr)

    # call constructor
    return spatial_qmodel_sparse(Ly, Lx, J_arr, h, xi, pos, **kwargs)


def spatial_qmodel_dense_through_mask(Wy, Wx, pad_y, pad_x, J, h_arr, xi_arr, **kwargs):
    """
    Construct a quadratic objective function with spatial coupling matrix (spCM).
    This constructor assumes spins are aligned in a 2D array with padding, using
    a non-periodic spatial interaction function. It automatically creates a mask 
    for the active spin region.

    Parameters
    ----------
    Wy : int
        Number of rows in the positions grid.
    Wx : int
        Number of columns in the positions grid.
    pad_y : int
        Padding size in row direction
    pad_x : int
        Padding size in column direction
    J : np.ndarray
        A 2D array representing the spatial interaction function
    h_arr : np.ndarray
        A 2D array containing the bias terms, with shape (Wy, Wx)
    xi_arr : np.ndarray
        A 2D array of coefficients for the spatial coupling matrix,
        with shape (Wy, Wx)
    kwargs:
        Additional parameters to be passed to the constructor of spatial_qmodel_sparse

    Returns
    -------
    spatial_qmodel_sparse
        The constructed sparse spatial quadratic function.
    """

    Ly = Wy + pad_y
    Lx = Wx + pad_x
    mask = np.zeros((Ly, Lx), dtype="i")
    mask[:Wy, :Wx] = 1
    _h_arr = np.zeros((Ly, Lx), dtype="f")
    _h_arr[:Wy, :Wx] = h_arr
    _xi_arr = np.zeros((Ly, Lx), dtype="f")
    _xi_arr[:Wy, :Wx] = xi_arr
    return spatial_qmodel_from_masked_J_and_h(
        mask, J, _h_arr, _xi_arr, **kwargs)
