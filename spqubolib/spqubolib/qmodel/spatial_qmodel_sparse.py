import numpy as np

from .spatial_qmodel import spatial_qmodel_base
from ..interaction import fourier_interaction as spqf
from ..interaction import fourier_function_cxx as spqcf


def get_vanilla_J_sparse(J_sp, xi, pos):
    """
    Computes the coefficient matrix in the basic format from the parameters of the sparse spatial quadratic model.
    This function is used for the backend of the `spatial_qmodel_sparse` class.

    Parameters
    ----------
    J_sp : np.ndarray
        The array representing the spatial interaction function.
    xi : np.ndarray
        The coefficient vector.
    pos : np.ndarray
        An (N, 2) array of spin positions.

    Returns
    -------
    np.ndarray
        The coefficient matrix in the basic format.
    """
    N = pos.shape[0]
    Ly, Lx = J_sp.shape
    J = np.zeros((N, N), dtype="f")

    for k1, ij1 in enumerate(pos):
        i1, j1 = ij1
        for k2, ij2 in enumerate(pos):
            i2, j2 = ij2
            dy, dx = i1-i2, j1-j2
            i = dy
            j = dx
            if i < 0:
                i = i + Ly
            if j < 0:
                j = j + Lx
            w = J_sp[i, j]
            J[k1, k2] = w
    ret = np.einsum("i,ij,j->ij", xi, J, xi, dtype="f")
    return ret


class spatial_qmodel_sparse(spatial_qmodel_base):
    def __init__(self, Ly, Lx, J, h, xi, pos, const=0, mode="naive"):
        """
        The quadratic objective function with spatial coupling matrix (spCM), constructed from
        non-periodic spatial interaction function assuming that the spins are aligned in a 2-dim array.

        Parameters
        ----------
        Ly : int
            The number of rows in the spatial shape.
        Lx : int
            The number of columns in the spatial shape.
        J : np.ndarray
            The array representing the spatial interaction function.
        h : np.ndarray
            The 1st-order coefficient vector (bias vector).
        xi : np.ndarray
            The coefficient vector.
        pos : np.ndarray
            An (N, 2) array of spin positions.
        const : float, optional
            The constant term in the function, by default 0.
        mode : Literal["naive", "fourier", "cxx_fourier"], optional
            The mode for matrix operation, by defalt "naive".
            When "naive", it uses naive matrix operation and
            when "fourier" or "cxx_fourier", it uses fast Fourier transform to optimize the calculation.
            When "cxx_fourier", it also uses c++ implementation for mapping spins in their positions.
        """
        super().__init__(Ly, Lx, J, h, xi, const=const, mode=mode)

        assert (pos.dtype == np.int32)
        self.pos = pos
        self.f_J = None

    #### evaluation ####

    # def set_mode(self, mode): [inherit]

    # def evaluate(self, spins): [inherit]

    #### evaluation energy ####

    def evaluate_naive(self, spins):  # [override] 
        spins = spins.astype("f")
        E = np.sum((self.h * spins))
        for k1, ij1 in enumerate(self.pos):
            i1, j1 = ij1
            for k2, ij2 in enumerate(self.pos):
                i2, j2 = ij2
                dy, dx = i1-i2, j1-j2
                i, j = dy, dx
                if i < 0:
                    i = i + self.Ly
                if j < 0:
                    j = j + self.Lx
                w = self.J[i, j]
                E += w * self.xi[k1] * spins[k1] * self.xi[k2] * spins[k2]
        E = E + self.const
        return E

    #### evaluation interaction ####

    def evaluate_fourier_interaction(self, spins): # [implementation]
        x = (self.xi * spins).astype("f")
        return spqf.spatial_xJy_sparse(self.Ly, self.Lx, x, self.pos, self.get_f_J(), x, self.pos) \


    def evaluate_cxx_fourier_interaction(self, spins): # [implementation]
        x = (self.xi * spins).astype("f")
        return spqcf.spatial_xJy_sparse(self.Ly, self.Lx, x, self.pos, self.get_f_J(), x, self.pos) \

    #### evaluation Jx ####

    # def evaluate_Jx(self, x): [inherit]

    def evaluate_Jx_fourier(self, x): # [implementation]
        _x = (self.xi * x).astype("f")
        return self.xi * \
            spqf.spatial_Jx_sparse(
                self.Ly, self.Lx, self.get_f_J(), _x, self.pos)

    def evaluate_Jx_cxx_fourier(self, x): # [implementation]
        _x = (self.xi * x).astype("f")
        return self.xi * \
            spqcf.spatial_Jx_sparse(
                self.Ly, self.Lx, self.get_f_J(), _x, self.pos)

    #### to_vanilla ####

    # def to_vanilla_qm(self): [inherit]

    def get_vanilla_J(self): # [implementation]
        return get_vanilla_J_sparse(self.J, self.xi, self.pos)

    #### convert ####
    def convert_spin_to_binary(self): # [implementation]
        W, b, c = self._convert_spin_to_binary()
        return spatial_qmodel_sparse(self.Ly, self.Lx,
                                     W, b, self.xi,
                                     self.pos, const=c, mode=self.mode)

    def convert_binary_to_spin(self): # [implementation]
        J, h, c = self._convert_binary_to_spin()
        return spatial_qmodel_sparse(self.Ly, self.Lx,
                                     J, h, self.xi,
                                     self.pos, const=c, mode=self.mode)

    def negate(self): # [implementation]
        return spatial_qmodel_sparse(self.Ly, self.Lx,
                                     -self.J, -self.h, self.xi,
                                     self.pos, const=-self.const,
                                     mode=self.mode)


class spatial_qmodel_square_sparse(spatial_qmodel_sparse):
    def __init__(self, L, J, h, xi, pos, const=0, mode="naive"):
        """
        The quadratic objective function with square spatial coupling matrix (spCM), constructed from
        non-periodic spatial interaction function assuming that the spins are aligned in a 2-dim array.

        Parameters
        ----------
        L : int
            The size of the spatial shape (assumed square).
        J : np.ndarray
            The array representing the spatial interaction function.
        h : np.ndarray
            The 1st-order coefficient vector (bias vector).
        xi : np.ndarray
            The coefficient vector.
        pos : np.ndarray
            An (N, 2) array of spin positions.
        const : float, optional
            The constant term in the function, by default 0.
        mode : Literal["naive", "fourier", "cxx_fourier"], optional
            The mode for matrix operation, by default "naive".
            When "naive", it uses naive matrix operation and
            when "fourier" or "cxx_fourier", it uses fast Fourier transform to optimize the calculation.
            When "cxx_fourier", it also uses c++ implementation for mapping spins in their positions.
        """
        super().__init__(L, L, J, h, xi, pos, const=const, mode=mode)
