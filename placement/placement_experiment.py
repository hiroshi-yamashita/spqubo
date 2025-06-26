import time
import contextlib
from typing import Tuple

import numpy as np
nax = np.newaxis


class timer_class:
    def __init__(self):
        self.t = None


@contextlib.contextmanager
def timer():
    start_time = time.process_time()
    t = timer_class()
    try:
        yield t
    finally:
        end_time = time.process_time()
        elapsed_time = end_time - start_time
        t.t = elapsed_time
    return t


# spqubolib.qmodel.J_function.function_J_to_dense_J can be used,
# but we keep this source self-contained

def function_J_to_dense_J(Wy, Wx, pad_y, pad_x, fn_R_y, fn_R_x, fn):
    """
    Generate a spatial coupling matrix from a given function.

    Parameters
    ----------
    Wy : int
        Number of rows in the configuration space.
    Wx : int
        Number of columns in the configuration space.
    pad_y : int
        Padding size in the y direction.
    pad_x : int
        Padding size in the x direction.
    fn_R_y : int
        Range of the function in the y direction.
    fn_R_x : int
        Range of the function in the x direction.
    fn : Callable[[int, int], float]
        Spatial coupling function defined in the range [-R, R].
        The function values outside the range are ignored.

    Returns
    -------
    np.ndarray
        Spatial coupling matrix of size (Wy + pad_y, Wx + pad_x).
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


### problem size ###

def get_W_R(H, W, skip_y, skip_x, r):
    """
    Compute discretized problem sizes and locality parameters.

    Parameters
    ----------
    H : float
        Height of the rectangular space.
    W : float
        Width of the rectangular space.
    skip_y : float
        Step size in the y direction.
    skip_x : float
        Step size in the x direction.
    r : float
        Interaction radius.

    Returns
    -------
    Tuple[int, int, int, int]
        Wy: Number of rows in the positions grid.
        Wx: Number of columns in the positions grid.
        Ry: Locality parameter in the y direction.
        Rx: Locality parameter in the x direction.
    """
    Wy = np.ceil(H / skip_y).astype("i") + 1
    Wx = np.ceil(W / skip_x).astype("i") + 1
    Ry = np.ceil(r / skip_y).astype("i")
    Rx = np.ceil(r / skip_x).astype("i")
    return Wy, Wx, Ry, Rx


### normalized interaction ###

def normalized_coupling(x):
    assert (x.size == 1)
    x = x.ravel()[0] # = rij / 2R
    if x >= 1:
        return 0
    theta = np.arccos(x) 
    return (theta - x*x*np.tan(theta)) * 2 / np.pi
    # 2 (θij - (rij/2R)^2 tan θij) / π
    # = 2 (R^2 θij - rij^2 tan θij / 4) / (π R^2)
    # = 2 (R^2 θij - rij^2 tan θij / 4) / S
    # = W'ij / S

def get_J(Wy, Wx, Ry, Rx, skip_y, skip_x, r):
    """
    Compute the spatial coupling matrix for the placement problem.

    Parameters
    ----------
    Wy : int
        Number of rows in the positions grid.
    Wx : int
        Number of columns in the positions grid.
    Ry : int
        Locality parameter in the y direction.
    Rx : int
        Locality parameter in the x direction.
    skip_y : float
        Step size in the y direction.
    skip_x : float
        Step size in the x direction.
    r : float
        Reference distance for the coupling function.

    Returns
    -------
    np.ndarray
        Spatial coupling matrix.
    """
    def coupling(dy, dx):
        rxy = np.sqrt((dx * skip_x) ** 2 + (dy * skip_y) ** 2) # rij
        return normalized_coupling(rxy / r) # r = 2R

    J_sp_ = function_J_to_dense_J(Wy, Wx, Ry, Rx, Ry, Rx, coupling)
    return J_sp_ # = 2 (R^2 θij - rij^2 tan θij / 4) / S


### placement costs ###

def get_placement_cost_fixed_line_and_blob(H, W, Wy, Wx, skip_y, skip_x,
                                           coef_line, lengthscale_line,
                                           lengthscale_blob, K_blob):
    """
    Generate a random placement cost array with line and blob components.

    Parameters
    ----------
    H : float
        Height of the rectangular space.
    W : float
        Width of the rectangular space.
    Wy : int
        Number of rows in the positions grid.
    Wx : int
        Number of columns in the positions grid.
    skip_y : float
        Step size in the y direction.
    skip_x : float
        Step size in the x direction.
    coef_line : float
        Coefficient for the line component.
    lengthscale_line : float
        Scale of the Gaussian for the line component.
    lengthscale_blob : float
        Scale of the Gaussian for the blob component.
    K_blob : int
        Number of blobs for the blob component.

    Returns
    -------
    np.ndarray
        Placement cost array of size (Wy, Wx), scaled to [0, 1].
    """

    ### prepare data points ###

    x = np.zeros((Wy, Wx))
    y = np.zeros((Wy, Wx))
    x[:, :] = (np.arange(Wx)*skip_x)[nax, :]
    y[:, :] = (np.arange(Wy)*skip_y)[:, nax]
    x_ = x.ravel()
    y_ = y.ravel()
    v_ = np.array([y_, x_]).T
    print(v_.shape)


    ### blob values ###

    ### ### blob parameters ### ###
    mu = np.random.random((K_blob, 2)) * np.array([H, W])[nax, :]
    sigma = np.ones((K_blob, 2)) * lengthscale_blob
    ### ### distance to the centers ### ###
    diff_normalized = ((v_[nax, :, :] - mu[:, nax, :]) /
                       sigma[:, nax, :])  # (K_blob, N, 2)
    sqdist_normalized = np.sum(
        diff_normalized ** 2,
        axis=2
    )  # (K_blob, N)
    ### ### draw signs ### ###
    coef = np.random.randint(2, size=(sqdist_normalized.shape[0],)) * 2 - 1
    ### ### the weighted sum of Gaussian ### ###
    c_ = np.sum(coef[:, nax] * np.exp(-sqdist_normalized/2), axis=0)
    ### ### format the array ### ###
    c_blob = c_.reshape((Wy, Wx))  # (Wy, Wx)


    ### line values ###

    ### ### line parameters ### ###
    K_line = 4
    angle = np.array([
        0,   # (x, y) = (1, 0)
        1/4, #          (0, 1)
        0,   #          (1, 0)
        1/4  #          (0, 1)
    ]) * np.pi * 2
    sigma = np.ones((K_line, )) * lengthscale_line
    vec = np.stack([np.sin(angle), np.cos(angle)], axis=1)
    center = np.array([
        [0.5, 0.25], # (x, y) = (1, 2)
        [0.25, 0.5], #          (2, 1)
        [0.5, 0.75], #          (3, 2)
        [0.75, 0.5]  #          (2, 3)
    ]) * np.array([H, W])[nax, :]
    ### ### distance to the lines ### ###
    dist = np.sum((v_[nax, :, :] - center[:, nax, :]) *
                  vec[:, nax, :], axis=2)  # (K_line, N)
    sqdist_normalized = (dist / sigma[:, nax]) ** 2
    ### ### the maximum of Gaussian ### ###
    c_ = np.max(np.exp(-sqdist_normalized/2), axis=0)
    ### ### format the array ### ###
    c_line = c_.reshape((Wy, Wx))  # (Wy, Wx)


    ### aggregate values ###
    c = c_blob + coef_line * c_line


    ### normalize and soft clip ###
    c = (c - np.mean(c)) / np.std(c)
    c = 1 / (1 + np.exp(-c * 1.5)) # (σ^(f))^2 = 1.5
    cmin, cmax = np.min(c), np.max(c)
    print(f"c: {cmin} - {cmax}")


    ### invert the array to be costs ###
    c = 1 - c
    return c  # (Wy, Wx)


### get QUBO parameters ###

def get_J_and_h(H, W, skip_y, skip_x, coef_line, lengthscale_line,
                lengthscale_blob, K_blob, r, coef, density_ref,
                coef_placement_cost, const_placement_cost):
    """
    Compute spQUBO parameters based on spatial coupling and placement costs.

    Parameters
    ----------
    H : float
        Height of the rectangular space.
    W : float
        Width of the rectangular space.
    skip_y : float
        Step size in the y direction.
    skip_x : float
        Step size in the x direction.
    coef_line : float
        Coefficient for the line component.
    lengthscale_line : float
        Scale of the Gaussian for the line component. [length]
    lengthscale_blob : float
        Scale of the Gaussian for the blob component. [length]
    K_blob : int
        Number of blobs for the blob component.
    r : float
        Interaction radius. [length]
    coef : float
        Global coefficient for the objective function. [utility/number]
    density_ref : float
        Reference density for the placement utility function. [number/area]
    coef_placement_cost : float
        Slope value for scaling the placement cost.
    const_placement_cost : float
        Intercept value for scaling the placement cost.

    Returns
    -------
    Tuple[int, int, int, int, np.ndarray, np.ndarray, np.ndarray]
        Wy: Number of rows in the positions grid.
        Wx: Number of columns in the positions grid.
        Ry: Locality parameter in the y direction.
        Rx: Locality parameter in the x direction.
        J_sp: Spatial coupling matrix.
        h: Bias vector of the problem.
        placement_cost: Placement cost array.
    """

    ### get problem size ###
    Wy, Wx, Ry, Rx = get_W_R(H, W, skip_y, skip_x, r)


    ### get relative placement cost and convert to placement cost by linear transformation ###
    relative_placement_cost = get_placement_cost_fixed_line_and_blob(
        H, W, Wy, Wx, skip_y, skip_x,
        coef_line, lengthscale_line,
        lengthscale_blob, K_blob)
    placement_cost = const_placement_cost + \
        coef_placement_cost * relative_placement_cost  # [utility]


    ### get normalized coupling coefficient ###
    J_sp_ = get_J(Wy, Wx, Ry, Rx, skip_y, skip_x, r) # [1]
    # = 2 (R^2 θij - rij^2 tan θij / 4) / S = W'ij / S


    ### get coefficients ###
    area_ref = np.pi * r * r # = S [area]
    # density_ref = K [number/area]
    num_ref = area_ref * density_ref # = SK = N [number]
    print("N:", num_ref)
    coef_2 = -1/2 / num_ref * coef # [utility/number^2]
    # = -(1/2) * (coef / N)
    # = -(1/2) * (A/N^2) * S
    coef_1 = coef # [utility/number]
    # = A/N * S


    ### print summary ###
    print(f"coef1: {coef_1}")
    print(f"coef2: {coef_2}")
    rmin, rmax = np.min(relative_placement_cost), np.max(
        relative_placement_cost)
    print(f"r: {rmin} - {rmax}")
    pmin, pmax = np.min(placement_cost), np.max(placement_cost)
    print(f"p: {pmin} - {pmax}")
    jmin, jmax = np.min(J_sp_),  np.max(J_sp_)
    print(f"J: {jmin} - {jmax}")


    ### merge components ###
    J_sp = coef_2 * J_sp_ # [utility/number^2]
    # = -(1/2) * (A / N^2) * (2 (R^2 θij - rij^2 tan θij / 4)) 
    # = aW'ij
    
    h = coef_1 * np.ones((Wy, Wx)) - placement_cost # [utility/number]
    return Wy, Wx, Ry, Rx, J_sp, h, placement_cost
