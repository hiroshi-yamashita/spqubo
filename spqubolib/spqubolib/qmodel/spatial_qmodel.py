import numpy as np

from . import qmodel as qm
from .qmodel_vanilla import qmodel_vanilla
nax = np.newaxis


class spatial_qmodel_base(qm.qmodel):
    def __init__(self, Ly, Lx, J, h, xi, const=0, mode="naive"):
        r"""
        A base class for the quadratic objective function with spatial coupling matrix (spCM),
        which is a coupling matrix for spatial QUBOs (spQUBOs).

        Specifically, it represents a function 
        $$
        H(x) = \sum_{i,j=0, \ldots, N-1} \xi_i\xi_j J(d_i-d_j) x_ix_j + \sum_{i=0,\ldots,N-1} h_ix_i,
        $$
        where $d_i$ denotes the $i$-th spin position.
        The function $J$ will be specified as a matrix, and 
        the spin positions are specified in inherited classes.
        
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
            The coefficient vector in the coupling matrix.
        const : float, optional
            The constant term in the function, by default 0.
        mode : Literal["naive", "fourier", "cxx_fourier"], optional
            The mode for matrix operations, by default "naive".
        """
        N = h.shape[0]
        super().__init__(N)
        self.const = np.float32(const)
        self.Ly = Ly
        self.Lx = Lx

        assert (J.shape == (self.Ly, self.Lx))
        assert (J.shape == (self.Ly, self.Lx))
        self.J = J
        self.J[0, 0] = 0
        self.f_J = None

        self.vanilla_J = None

        self.h = h
        self.xi = xi

        assert (self.J.dtype == np.float32)
        assert (self.h.dtype == np.float32)
        assert (self.xi.dtype == np.float32)
        assert (mode == "naive" or mode == "fourier" or mode == "cxx_fourier")
        self.mode = mode

    #### evaluation ####

    def set_mode(self, mode):
        """
        Updates the mode for matrix operations.

        Parameters
        ----------
        mode : Literal["naive", "fourier", "cxx_fourier"]
            The new mode for matrix operations.
        """
        self.mode = mode

    def evaluate(self, spins): # [implementation]
        if self.mode == "naive":
            return self.evaluate_naive(spins)
        elif self.mode == "fourier":
            return self.evaluate_fourier(spins)
        elif self.mode == "cxx_fourier":
            return self.evaluate_cxx_fourier(spins)
        else:
            raise ValueError

    #### fourier ####

    def get_f_J(self):
        """
        Precomputes the Fourier transform of the interaction function.

        Returns
        -------
        np.ndarray
            The obtained Fourier coefficients of the interaction function.
        """

        # skip if already obtained
        if self.f_J is None:
            _f_J = np.fft.fft2(self.J)
            assert (np.allclose(_f_J.imag, 0))
            self.f_J = _f_J.real.astype("f")
        return self.f_J

    #### evaluate energy ####

    def evaluate_naive(self, spins):
        """
        Evaluates the function value using the naive computation mode.

        Parameters
        ----------
        spins : np.ndarray
            Input vector of variables.

        Returns
        -------
        np.number
            The function value of the model.
        """
        spins = spins.astype("f")
        Jx = self.evaluate_Jx_naive(spins)
        return np.sum(spins * Jx) \
            + np.sum(self.h * spins) + self.const

    def evaluate_fourier(self, spins):
        """
        Evaluates the function value using the Fourier computation mode.

        Parameters
        ----------
        spins : np.ndarray
            Input vector of variables.

        Returns
        -------
        np.number
            The function value of the model.
        """
        spins = spins.astype("f")
        return self.evaluate_fourier_interaction(spins) \
            + np.sum(self.h * spins) + self.const

    def evaluate_cxx_fourier(self, spins):
        """
        Evaluates the function value using the C++ Fourier computation mode.

        Parameters
        ----------
        spins : np.ndarray
            Input vector of variables.

        Returns
        -------
        np.number
            The function value of the model.
        """
        spins = spins.astype("f")
        return self.evaluate_cxx_fourier_interaction(spins) \
            + np.sum(self.h * spins) + self.const

    #### evaluation interaction ####

    def evaluate_fourier_interaction(self, spins):
        """
        Evaluates the interaction value using the Fourier computation mode.

        Parameters
        ----------
        spins : np.ndarray
            Input vector of variables.

        Returns
        -------
        np.number
            The evaluated interaction value.
        """
        raise NotImplementedError

    def evaluate_cxx_fourier_interaction(self, spins):
        """
        Evaluates the interaction value using the C++ Fourier computation mode.

        Parameters
        ----------
        spins : np.ndarray
            Input vector of variables.

        Returns
        -------
        np.number
            The evaluated interaction value.
        """
        raise NotImplementedError

    #### evaluate Jx ####

    def evaluate_Jx(self, x): # [implementation]
        if self.mode == "naive":
            return self.evaluate_Jx_naive(x)
        elif self.mode == "fourier":
            return self.evaluate_Jx_fourier(x)
        elif self.mode == "cxx_fourier":
            return self.evaluate_Jx_cxx_fourier(x)
        else:
            raise ValueError

    def evaluate_Jx_naive(self, x):
        """
        Evaluates the matrix-vector product (MVP) using the naive computation mode.

        Parameters
        ----------
        x : np.ndarray
            Input vector of variables.

        Returns
        -------
        np.ndarray
            The spin vector of the MVP result.
        """
        if self.vanilla_J is None:
            self.vanilla_J = self.get_vanilla_J()
        return self.vanilla_J @ x.astype("f")

    def evaluate_Jx_fourier(self, x):
        """
        Evaluates the matrix-vector product (MVP) using the Fourier computation mode.

        Parameters
        ----------
        x : np.ndarray
            Input vector of variables.

        Returns
        -------
        np.ndarray
            The spin vector of the MVP result.
        """
        raise NotImplementedError

    def evaluate_Jx_cxx_fourier(self, x):
        """
        Evaluates the matrix-vector product (MVP) using the C++ Fourier computation mode.

        Parameters
        ----------
        x : np.ndarray
            Input vector of variables.

        Returns
        -------
        np.ndarray
            The spin vector of the MVP result.
        """
        raise NotImplementedError

    #### to_vanilla ####
    def to_vanilla_qm(self):
        """
        Converts this model to the basic format.

        Returns
        -------
        qmodel_vanilla
            The equivalent model in the basic format.
        """
        J = self.get_vanilla_J()
        N = J.shape[0]
        return qmodel_vanilla(N, J, self.h, const=self.const)

    def get_vanilla_J(self):
        """
        Computes the coefficient matrix in the basic format.

        Returns
        -------
        np.ndarray
            The coefficient matrix in the basic format.
        """
        raise NotImplementedError

    #### convert ####

    # def convert(self, vartype_current, vartype_new): [inherit]

    def convert_spin_to_binary(self): # [implementation]
                raise NotImplementedError

    def convert_binary_to_spin(self): # [implementation]
                raise NotImplementedError

    def negate(self): # [implementation]
                raise NotImplementedError

    def _convert_spin_to_binary(self):
        """
        Obtains the spatial interaction array and the bias vector
        for the equivalent model with `binary` variable type,
        assuming that the current model is `spin` type.

        This function is a core implementation of qmodel conversion using the matrix-vector product (MVP), 
        which can be used in the inherited class.

        Returns
        -------
        W : np.ndarray
            The 2nd-order coefficient matrix (interaction matrix).
        b : np.ndarray
            The 1st-order coefficient vector (bias vector).
        c : np.number
            The constant term in the function.
        """
        # 
        J, h = self.J, self.h
        y = self.evaluate_Jx(np.ones(self.N, dtype="f"))
        W = 4 * J
        b1 = -4 * y
        b2 = 2 * h
        c1 = np.sum(y)
        c2 = -np.sum(h)
        b = b1 + b2
        c = self.const + c1 + c2
        return W, b, c

    def _convert_binary_to_spin(self):
        """
        Obtain the spatial interaction array and the bias vector
        for the equivalent model with `spin` variable type,
        assuming that the current model is `binary` type.

        This function is a core implementation of qmodel conversion using the matrix-vector product (MVP), 
        which can be used in the inherited class.

        Returns
        -------
        J : np.ndarray
            The 2nd-order coefficient matrix (the interaction matrix).
        h : np.ndarray
            The 1st-order coefficient vector (bias vector).
        c : np.number
            The constant term in the function.
        """
        W, b = self.J, self.h
        y = self.evaluate_Jx(np.ones(self.N, dtype="f"))
        J = W / 4
        h1 = y / 2
        h2 = b / 2
        c1 = np.sum(y) / 4
        c2 = np.sum(b) / 2
        h = h1 + h2
        c = self.const + c1 + c2
        return J, h, c

