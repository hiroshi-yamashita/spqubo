import numpy as np


def spinvector_to_spinarr(Ly, Lx, pos, x):
    """
    Rearranges a spin vector into a 2D array.

    Parameters
    ----------
    Ly : int
        Number of rows in the spatial grid.
    Lx : int
        Number of columns in the spatial grid.
    pos : np.ndarray
        (N, 2) array of variable positions in the grid.
    x : np.ndarray
        Input spin vector.

    Returns
    -------
    np.ndarray
        A 2D array where the spin vector is rearranged based on the positions.
    """
    arr = np.zeros((Ly, Lx), dtype="f")
    for k, ij in enumerate(pos):
        i, j = ij
        arr[i, j] = x[k]
    return arr


def spinarr_to_spinvector(Ly, Lx, pos, arr):
    """
    Rearranges a 2D spin array into a vector. Only the positions corresponding 
    to the spin positions are considered.

    Parameters
    ----------
    Ly : int
        Number of rows in the spatial grid.
    Lx : int
        Number of columns in the spatial grid.
    pos : np.ndarray
        (N, 2) array of variable positions in the grid.
    arr : np.ndarray
        Input 2D spin array.

    Returns
    -------
    np.ndarray
        A vector containing the spin values at the specified positions.
    """
    x = np.zeros((pos.shape[0], ), dtype="f")
    for k, ij in enumerate(pos):
        i, j = ij
        x[k] = arr[i, j]
    return x
