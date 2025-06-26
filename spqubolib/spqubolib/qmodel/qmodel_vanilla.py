from .qmodel import qmodel

import numpy as np
nax = np.newaxis


class qmodel_vanilla(qmodel):
    def __init__(self, N, J, h, const=0):
        """
        Basic implementation of the quadratic function.

        Parameters
        ----------
        N : int
            The number of spins.
        J : np.ndarray
            The 2nd-order coefficient matrix (interaction matrix).
        h : np.ndarray
            The 1st-order coefficient vector (bias vector).
        const : float, optional
            The constant term in the function, by default 0.
        """
        super().__init__(N)
        self.const = np.float32(const)
        self.J = J.astype("f")
        self.h = h.astype("f")
        # assert (self.J.dtype == np.dtype("f"))
        # assert (self.h.dtype == np.dtype("f"))
        # assert (self.const.dtype == np.dtype("f"))

    #### evaluation ####

    def evaluate(self, spins): # [implementation]
        spins = spins.astype("f")
        y = self.J @ spins
        E = np.sum(spins * y) + np.sum(self.h * spins)
        return E + self.const

    def evaluate_Jx(self, x): # [implementation]
        ret = self.J @ x.astype("f")
        return self.J @ x.astype("f")

    #### convert ####

    # def convert(self, vartype_current, vartype_new): [inherit]

    def convert_binary_to_spin(self): # [implementation]
        W, b = self.J, self.h
        J = W / 4
        y = self.evaluate_Jx(np.ones((self.N, )))
        h1 = y / 2
        h2 = b / 2
        c1 = np.sum(y) / 4
        c2 = np.sum(b) / 2
        h = h1 + h2
        c = self.const + c1 + c2
        return qmodel_vanilla(self.N, J, h, const=c)

    def convert_spin_to_binary(self): # [implementation]
        J, h = self.J, self.h
        W = 4 * J
        y = self.evaluate_Jx(np.ones((self.N, )))
        b1 = -4 * y
        b2 = 2 * h
        c1 = np.sum(y)
        c2 = -np.sum(h)
        b = b1 + b2
        c = self.const + c1 + c2
        return qmodel_vanilla(self.N, W, b, const=c)

    def negate(self): # [implementation]
        return qmodel_vanilla(self.N, -self.J, -self.h, const=-self.const)
