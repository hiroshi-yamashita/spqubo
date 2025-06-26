import numpy as np
from . import spin_mapping_py as psm

nax = np.newaxis


def _spatial_xJy(LyLx, spins_x, f_J, spins_y):
    """
    Computes the quadratic form between two 2D arrays for a spatial interaction function 
    using fast Fourier transforms (FFT).

    This function is as a backend for the `spatial_xJy_sparse` and `spatial_xJy_dense` functions.
    
    ----------
    LyLx : int
        Total number of elements in the spin array (Ly * Lx).
    spins_x : np.ndarray
        Input 2D array representing the spin configuration of the first input.
    f_J : np.ndarray
        Precomputed Fourier transform of the interaction function.
    spins_y : np.ndarray
        Input 2D array representing the spin configuration of the first input.

    Returns
    -------
    np.number
        The computed value of the quadratic form.
    """
    assert (f_J.dtype == np.float32)
    assert (spins_x.dtype == np.float32)
    assert (spins_y.dtype == np.float32)
    f_spins_x = np.fft.fft2(spins_x)
    f_spins_y = np.fft.fft2(spins_y)
    # f_J = np.fft.fft2(J)
    return np.sum(f_spins_x*f_spins_y.conj() *
                  f_J).real.astype("f") / np.float32(LyLx)


def _spatial_Jx(f_J, spins_x):
    """
    Computes the matrix-vector product for a 2D array and a spatial interaction function 
    using fast Fourier transforms (FFT). The result is a 2D array.

    This function is as a backend for the `spatial_Jx_sparse` and `spatial_Jx_dense` functions.

    Parameters
    ----------
    f_J : np.ndarray
        Precomputed Fourier transform of the interaction function.
    spins_x : np.ndarray
        Input 2D array representing the spin configuration.

    Returns
    -------
    np.ndarray
        The resulting 2D array after the matrix-vector product.
    """
    assert (f_J.dtype == np.float32)
    assert (spins_x.dtype == np.float32)
    f_spins_x = np.fft.fft2(spins_x)
    # f_J = np.fft.fft2(J)
    return np.fft.ifft2((f_spins_x*f_J)).real.astype("f")


def spatial_xJy_sparse(Ly, Lx, x, pos_x, f_J, y, pos_y):
    """
    Computes the quadratic form between two spin vectors for a sparse spatial interaction 
    function using fast Fourier transforms (FFT).

    Parameters
    ----------
    Ly : int
        Number of rows in the spatial grid.
    Lx : int
        Number of columns in the spatial grid.
    x : np.ndarray
        Input spin vector for the first input.
    pos_x : np.ndarray
        (N, 2) array of variable positions for `x`.
    f_J : np.ndarray
        Precomputed Fourier transform of the interaction function.
    y : np.ndarray
        Input spin vector for the second input.
    pos_y : np.ndarray
        (N, 2) array of variable positions for `y`.

    Returns
    -------
    np.number
        The computed value of the quadratic form.
    """
    spins_x = psm.spinvector_to_spinarr(Ly, Lx, pos_x, x)
    spins_y = psm.spinvector_to_spinarr(Ly, Lx, pos_y, y)
    return _spatial_xJy(Ly*Lx, spins_x, f_J, spins_y)


def spatial_Jx_sparse(Ly, Lx, f_J, x, pos_x):
    """
    Computes the matrix-vector product for a sparse spin vector and a spatial interaction 
    function using fast Fourier transforms (FFT).

    Parameters
    ----------
    Ly : int
        Number of rows in the spatial grid.
    Lx : int
        Number of columns in the spatial grid.
    f_J : np.ndarray
        Precomputed Fourier transform of the interaction function.
    x : np.ndarray
        Input spin vector.
    pos_x : np.ndarray
        (N, 2) array of variable positions for `x`.

    Returns
    -------
    np.ndarray
        The resulting spin vector after the matrix-vector product.
    """
    spins_x = psm.spinvector_to_spinarr(Ly, Lx,  pos_x, x)
    a_arr = _spatial_Jx(f_J, spins_x)
    a = psm.spinarr_to_spinvector(Ly, Lx, pos_x, a_arr)
    return a


def spatial_xJy_dense(Ly, Lx, x, pad_x_y, pad_x_x, f_J, y, pad_y_y, pad_y_x):
    """
    Computes the quadratic form between two spin vectors for a dense spatial interaction 
    function using fast Fourier transforms (FFT).

    Parameters
    ----------
    Ly : int
        Number of rows in the spatial grid.
    Lx : int
        Number of columns in the spatial grid.
    x : np.ndarray
        Input spin vector for the first input.
    pad_x_y : int
        Padding size along the y-axis for `x`.
    pad_x_x : int
        Padding size along the x-axis for `x`.
    f_J : np.ndarray
        Precomputed Fourier transform of the interaction function.
    y : np.ndarray
        Input spin vector for the second input.
    pad_y_y : int
        Padding size along the y-axis for `y`.
    pad_y_x : int
        Padding size along the x-axis for `y`.

    Returns
    -------
    np.number
        The computed value of the quadratic form.
    """
    Nx_y = Ly - pad_x_y
    Nx_x = Lx - pad_x_x
    spins_x = np.zeros((Ly, Lx), dtype="f")
    spins_x[0:Nx_y, 0:Nx_x] = x.reshape((Nx_y, Nx_x))

    Ny_y = Ly - pad_y_y
    Ny_x = Lx - pad_y_x
    spins_y = np.zeros((Ly, Lx), dtype="f")
    spins_y[0:Ny_y, 0:Ny_x] = y.reshape((Ny_y, Ny_x))

    return _spatial_xJy(Ly*Lx, spins_x, f_J, spins_y)


def spatial_Jx_dense(Ly, Lx, f_J, x, pad_x_y, pad_x_x):
    """
    Computes the matrix-vector product for a spin vector and a dense spatial interaction 
    function using fast Fourier transforms (FFT).

    Parameters
    ----------
    Ly : int
        Number of rows in the spatial grid.
    Lx : int
        Number of columns in the spatial grid.
    f_J : np.ndarray
        Precomputed Fourier transform of the interaction function.
    x : np.ndarray
        Input spin vector.
    pad_x_y : int
        Padding size along the y-axis for `x`.
    pad_x_x : int
        Padding size along the x-axis for `x`.

    Returns
    -------
    np.ndarray
        The resulting spin vector after the matrix-vector product.
    """
    Nx_y = Ly - pad_x_y
    Nx_x = Lx - pad_x_x
    spins_x = np.zeros((Ly, Lx), dtype="f")
    spins_x[0:Nx_y, 0:Nx_x] = x.reshape((Nx_y, Nx_x))

    a_arr = _spatial_Jx(f_J, spins_x)

    return a_arr[0:Nx_y, 0:Nx_x].ravel()
