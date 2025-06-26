from .spatial_qmodel_sparse import spatial_qmodel_square_sparse
from .spatial_qmodel_dense import spatial_qmodel_square_dense
import numpy as np


def function_J_to_dense_J(Wy, Wx, pad_y, pad_x, fn_R_y, fn_R_x, fn):
    """Convert an interaction function to a dense J matrix with rectangular dimensions.

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
    fn_R_y : int
        The domain size [-fn_R_y, fn_R_y] of the input function in row direction
    fn_R_x : int
        The domain size [-fn_R_x, fn_R_x] of the input function in column direction
    fn : callable
        Function that takes (dy, dx) as arguments and returns interaction strength

    Returns
    -------
    numpy.ndarray
        Dense J matrix with shape (Ly, Lx) where Ly = Wy + pad_y and Lx = Wx + pad_x

    Raises
    ------
    AssertionError
        If domain size is larger than the position grid size
    """
    Ly = Wy + pad_y
    Lx = Wx + pad_x
    assert (fn_R_y < Ly)
    assert (fn_R_x < Lx)
    J = np.zeros((Ly, Lx), dtype="f")
    index_y = np.arange(-fn_R_y, fn_R_y+1)
    index_x = np.arange(-fn_R_x, fn_R_x+1)
    for dy in index_y:
        for dx in index_x:
            i, j = dy, dx
            if i < 0:
                i = i + Ly
            if j < 0:
                j = j + Lx
            J[i, j] = fn(dy, dx)
    return J


def function_J_to_dense_J_square(W, pad, fn_R, fn):
    """Convert an interaction function to a dense J matrix with square dimensions.

    This is a convenience wrapper around function_J_to_dense_J for square systems
    where dimensions and parameters are the same in both directions.

    Parameters
    ----------
    W : int
        Number of rows/columns in the positions grid
    pad : int
        Padding size in both row and column directions
    fn_R : int
        The domain size [-fn_R, fn_R] of the input function in both directions
    fn : callable
        Function that takes (dy, dx) as arguments and returns interaction strength

    Returns
    -------
    numpy.ndarray
        Square dense J matrix with shape (L, L) where L = W + pad
    """
    return function_J_to_dense_J(W, W, pad, pad, fn_R, fn_R, fn)


def load_spatial_qmodel_from_function(W, fn_R, r, h, xi, pos, **kwargs):
    """Create a sparse spatial quantum model from an interaction function.

    Parameters
    ----------
    W : int
        Number of rows/columns in the positions grid
    fn_R : int
        The domain size [-fn_R, fn_R] of the input function in both directions
    r : callable
        Function that takes (dy, dx) as arguments and returns interaction strength
    h : float or array-like
        Local field strength(s) at each position
    xi : array-like
        Spin variables at each position
    pos : array-like
        Positions of the spins in the grid
    **kwargs : dict
        Additional arguments passed to spatial_qmodel_square_sparse

    Returns
    -------
    spatial_qmodel_square_sparse
        Sparse spatial quadratic function instance
    """
    J = function_J_to_dense_J_square(W, fn_R, fn_R, r)
    J = J.T  # order of dy, dx. compatibility. [TODO] fix it ###
    L = W + fn_R
    return spatial_qmodel_square_sparse(L, J, h, xi, pos, **kwargs)


def load_spatial_dense_qmodel_from_function(W, pad, fn_R, r, h_arr, xi_arr, **kwargs):
    """Create a dense spatial quantum model from an interaction function.

    Parameters
    ----------
    W : int
        Number of rows/columns in the positions grid
    pad : int
        Padding size in both row and column directions
    fn_R : int
        The domain size [-fn_R, fn_R] of the input function in both directions
    r : callable
        Function that takes (dy, dx) as arguments and returns interaction strength
    h_arr : array-like
        Array of local field strengths at each position
    xi_arr : array-like
        Array of spin variables at each position
    **kwargs : dict
        Additional arguments passed to spatial_qmodel_square_dense

    Returns
    -------
    spatial_qmodel_square_dense
        Dense spatial quadratic function instance
    """
    J = function_J_to_dense_J_square(W, pad, fn_R, r)
    J = J.T  # order of dy, dx. compatibility. [TODO] fix it ###
    return spatial_qmodel_square_dense(W, pad, J, h_arr, xi_arr, **kwargs)


def check_spatialJ(fn_R, J_sp):
    """Validate properties of a spatial J matrix.

    This function checks three essential properties:
    1. The diagonal elements (J[0,0]) are zero
    2. The matrix is symmetric: J[i,j] == J[-i,-j]
    3. All elements outside the domain [-fn_R, fn_R] are zero

    Parameters
    ----------
    fn_R : int
        The domain size [-fn_R, fn_R] to check for non-zero elements
    J_sp : numpy.ndarray
        Spatial J matrix to validate, must be square

    Raises
    ------
    ValueError
        If J[0,0] is not zero
        If the matrix is not symmetric
        If there are nonzero values outside the specified domain
    """
    L, _ = J_sp.shape
    index = np.arange(-fn_R, fn_R+1)
    mask = np.ones((L, L))

    try:
        assert (np.isclose(J_sp[0, 0], 0))
    except Exception as e:
        raise ValueError("J[0, 0] must be zero") from e

    for dx in index:
        for dy in index:
            i, j = dy, dx
            if i < 0:
                i = i + L
            if j < 0:
                j = j + L

            i_, j_ = -dy, -dx
            if i_ < 0:
                i_ = i_ + L
            if j < 0:
                j_ = j_ + L

            try:
                assert (J_sp[i, j] == J_sp[i_, j_])
            except Exception as e:
                raise ValueError("not symmetric") from e
            mask[i, j] = 0

    try:
        assert (np.allclose(J_sp * mask, np.zeros((L, L))))
    except Exception as e:
        raise ValueError("nonzero value outside the specified region") from e
