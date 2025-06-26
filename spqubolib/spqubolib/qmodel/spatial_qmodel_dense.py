from .spatial_qmodel import spatial_qmodel_base
from ..interaction import fourier_interaction as spqf
from ..interaction import fourier_function_cxx as spqcf

import numpy as np

nax = np.newaxis


def get_vanilla_J_dense(Wy, Wx, J_sp, xi):
    """
    Computes the coefficient matrix in the basic format from the parameters of the dense spatial quadratic model.
    This function is used for the backend of the `spatial_qmodel_dense` class.

    Parameters
    ----------
    Wy : int
        The number of rows in the spin array.
    Wx : int
        The number of columns in the spin array.
    J_sp : np.ndarray
        The array representing the spatial interaction function.
    xi : np.ndarray
        The coefficient vector in the spatial coupling matrix (spCM).

    Returns
    -------
    np.ndarray
        The coefficient matrix in the basic format.
    """
    Ly, Lx = J_sp.shape
    J = np.zeros((Ly*2, Lx*2, Ly*2, Lx*2), dtype="f")
    for i in np.arange(Ly):
        for j in np.arange(Lx):
            J[i, j, i:(i+Ly), j:(j+Lx)] = J_sp
    J = J[:, :, :, :Lx] + J[:, :, :, Lx:]
    J = J[:, :, :Ly, :] + J[:, :, Ly:, :]
    J = J[:, :Lx, :, :] + J[:, Lx:, :, :]
    J = J[:Ly, :, :, :] + J[Ly:, :, :, :]
    J = J[:Wy, :Wx, :Wy, :Wx]
    J = J.reshape((Wy*Wx, Wy*Wx))

    return np.einsum("i,ij,j->ij", xi, J, xi)


class spatial_qmodel_dense(spatial_qmodel_base):
    def __init__(self, Wy, Wx, pad_y, pad_x, J, h_arr, xi_arr, const=0, mode="naive", fn_R_y=None, fn_R_x=None):
        """
        The quadratic objective function with spatial coupling matrix (spCM), constructed from
        non-periodic spatial interaction function assuming that the spins are aligned in a 2-dim array.


        Parameters
        ----------
        Wy : int
            The number of rows in the spin array.
        Wx : int
            The number of columns in the spin array.
        pad_y : int
            The padding in the y-dimension for the interaction function.
        pad_x : int
            The padding in the x-dimension for the interaction function.
        J : np.ndarray
            The array representing the spatial interaction function.
        h_arr : np.ndarray
            The 1st-order coefficient 2D array (bias vector).
        xi_arr : np.ndarray
            The coefficient 2D array in the spatial coupling matrix (spCM).
        const : float, optional
            The constant term in the function, by default 0.
        mode : Literal["naive", "fourier", "cxx_fourier"], optional
            The mode for matrix operation, by default "naive".
            When "naive", it uses naive matrix operation and
            when "fourier" or "cxx_fourier", it uses fast Fourier transform to optimize the calculation.
            When "cxx_fourier", it also uses c++ implementation for mapping spins in their positions.
        """

        self.Wy = Wy
        self.Wx = Wx
        self.pad_y = pad_y
        self.pad_x = pad_x
        self.Ly = Wy + pad_y
        self.Lx = Wx + pad_x
        super().__init__(self.Ly, self.Lx, J, h_arr.ravel(),
                         xi_arr.ravel(), const=const, mode=mode)

        assert (h_arr.shape == (self.Wy, self.Wx))
        assert (xi_arr.shape == (self.Wy, self.Wx))

        self.h_arr = h_arr
        self.xi_arr = xi_arr

        self.fn_R_y = fn_R_y
        self.fn_R_x = fn_R_x

    #### evaluation ####

    # def set_mode(self, mode): [inherit]

    # def evaluate(self, spins): [inherit]

    #### evaluation energy ####

    #### evaluation interaction ####

    def evaluate_fourier_interaction(self, spins): # [implementation]
        x = (self.xi * spins).astype("f")
        return spqf.spatial_xJy_dense(self.Ly, self.Lx, x, self.pad_y, self.pad_x, self.get_f_J(), x, self.pad_y, self.pad_x)

    def evaluate_cxx_fourier_interaction(self, spins): # [implementation]
        x = (self.xi * spins).astype("f")
        return spqcf.spatial_xJy_dense(self.Ly, self.Lx, x, self.pad_y, self.pad_x, self.get_f_J(), x, self.pad_y, self.pad_x)

    #### evaluation Jx ####

    # def evaluate_Jx(self, x): [inherit]

    def evaluate_Jx_fourier(self, x): # [implementation]
        return self.xi * \
            spqf.spatial_Jx_dense(
                self.Ly, self.Lx, self.get_f_J(), self.xi * x, self.pad_y, self.pad_x)

    def evaluate_Jx_cxx_fourier(self, x): # [implementation]
        return self.xi * \
            spqcf.spatial_Jx_dense(
                self.Ly, self.Lx, self.get_f_J(), self.xi * x, self.pad_y, self.pad_x)

    #### to_vanilla ####

    # def to_vanilla_qm(self): [inherit]

    def get_vanilla_J(self): # [implementation]
        return get_vanilla_J_dense(self.Wy, self.Wx, self.J, self.xi)

    #### convert ####
    def convert_spin_to_binary(self): # [implementation]
        W, b, c = self._convert_spin_to_binary()
        b_arr = b.reshape((self.Wy, self.Wx))
        return spatial_qmodel_dense(self.Wy, self.Wx,
                                    self.pad_y, self.pad_x,
                                    W, b_arr,
                                    self.xi_arr,
                                    const=c, mode=self.mode,
                                    fn_R_y=self.fn_R_y,
                                    fn_R_x=self.fn_R_x)

    def convert_binary_to_spin(self): # [implementation]
        J, h, c = self._convert_binary_to_spin()
        h_arr = h.reshape((self.Wy, self.Wx))
        return spatial_qmodel_dense(self.Wy, self.Wx,
                                    self.pad_y, self.pad_x,
                                    J, h_arr,
                                    self.xi_arr,
                                    const=c, mode=self.mode,
                                    fn_R_y=self.fn_R_y,
                                    fn_R_x=self.fn_R_x)

    def negate(self): # [implementation]
        return spatial_qmodel_dense(self.Wy, self.Wx,
                                    self.pad_y, self.pad_x,
                                    -self.J, -self.h_arr,
                                    self.xi_arr,
                                    const=-self.const, mode=self.mode,
                                    fn_R_y=self.fn_R_y,
                                    fn_R_x=self.fn_R_x)

    #### to_sparse ####
    def get_pos(self):
        """
        Get the position of spins in the 2-dim array.
        This function, which is inefficient for large W,
        is only for debugging and testing.

        Returns
        -------
        np.ndarray
            (N, 2) array of spin positions
        """
        return get_pos_dense(self.W)


def get_pos_dense(W):
    assert (W <= 5)  # only for debug
    posx = np.zeros((W, W), dtype="i")
    posy = np.zeros((W, W), dtype="i")

    posy[:, :] = np.arange(W)[:, nax]
    posx[:, :] = np.arange(W)[nax, :]
    posy = posy.ravel()
    posx = posx.ravel()
    pos = np.array([posy, posx]).T
    return pos


class spatial_qmodel_square_dense(spatial_qmodel_dense):
    def __init__(self, W, pad, J, h_arr, xi_arr, const=0, mode="naive", fn_R=None):
        """
        The quadratic objective function with square spatial coupling matrix (spCM), constructed from
        non-periodic spatial interaction function assuming that the spins are aligned in a 2-dim array.


        Parameters
        ----------
        W : int
            The size of the spin array.
        pad : int
            The padding for the interaction function.
        J : np.ndarray
            The array representing the spatial interaction function.
        h_arr : np.ndarray
            The 1st-order coefficient 2D array (bias vector), which has the shape of (Wy, Wx).
        xi_arr : np.ndarray
            The coefficient 2D array in the spatial coupling matrix (spCM), which has the shape of (Wy, Wx).
        const : float, optional
            The constant term in the function, by default 0.
        mode : Literal["naive", "fourier", "cxx_fourier"], optional
            The mode for matrix operation, by default "naive".
            When "naive", it uses naive matrix operation and
            when "fourier" or "cxx_fourier", it uses fast Fourier transform to optimize the calculation.
            When "cxx_fourier", it also uses c++ implementation for mapping spins in their positions
        fn_R : _type_, optional
            _description_, by default None
        """

        super().__init__(W, W, pad, pad, J, h_arr, xi_arr,
                         const=const, mode=mode, fn_R_y=fn_R, fn_R_x=fn_R)
