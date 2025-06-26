import numpy as np
nax = np.newaxis


class qmodel:
    def __init__(self, N):
        """
        The base class for quadratic objective functions.
        This class, which is not intended to be used directly, 
        is inherited by qmodel_vanilla and spatial_qmodel.
        This class represents only the quadratic function 
        and has no information about the optimization mode or the variable type.
        These should be specified when constructing the QUBO object.

        Specifically, it represents a function 
        $$
        H(x) = \sum_{i,j=0, \ldots, N-1} J_{ij}x_ix_j + \sum_{i=0,\ldots,N-1} h_ix_i.
        $$

        Note that the coefficients may be different from some definitions of the Hamiltonian, such as
        $$
        H(x) = \sum_{i<j} J_{ij}x_ix_j + \sum_{i=0,\ldots,N-1} h_ix_i
        $$
        and
        $$
        H(x) = \frac{1}{2} \sum_{i,j=0, \ldots, N-1} J_{ij}x_ix_j + \sum_{i=0,\ldots,N-1} h_ix_i.
        $$

        Parameters
        ----------
        N : int
            The number of spins.
        """
        self.N = N

    #### evaluation ####

    def evaluate(self, spins):
        """
        Evaluates the function value for the given spin configuration.

        Parameters
        ----------
        spins : np.ndarray
            The input spin vector.

        Returns
        -------
        np.ndarray
            The function value for the given spin configuration.
        """
        raise NotImplementedError

    def evaluate_Jx(self, x):
        """
        Evaluates the matrix-vector product (MVP) for the given input vector.

        Parameters
        ----------
        x : np.ndarray
            The input vector of variables.

        Returns
        -------
        np.ndarray
            The result of the matrix-vector product (MVP).
        """
        raise NotImplementedError

    #### convert ####

    def convert(self, vartype_current, vartype_new):
        """
        Converts this objective function to an equivalent one that assumes the specified variable type.

        Parameters
        ----------
        vartype_current : Literal["binary", "spin"]
            The variable type currently assumed.
        vartype_new : Literal["binary", "spin"]
            The variable type to be assumed by the output.

        Returns
        -------
        qmodel
            The equivalent objective function that assumes the specified variable type.
        """
        if vartype_current == vartype_new:
            return self
        if vartype_current == "binary" and vartype_new == "spin":
            return self.convert_binary_to_spin()
        if vartype_current == "spin" and vartype_new == "binary":
            return self.convert_spin_to_binary()

    def convert_binary_to_spin(self):
        """
        Converts this objective function to an equivalent one that assumes the "spin" variable type,
        assuming the current variable type is "binary".

        Returns
        -------
        qmodel
            The equivalent objective function that assumes the "spin" variable type.
        """
        raise NotImplementedError

    def convert_spin_to_binary(self):
        """
        Converts this objective function to an equivalent one that assumes the "binary" variable type,
        assuming the current variable type is "spin".

        Returns
        -------
        qmodel
            The equivalent objective function that assumes the "binary" variable type.
        """
        raise NotImplementedError

    def negate(self):
        """
        Converts this objective function to by negating the sign of the output.

        Returns
        -------
        qmodel
            The objective function with the sign of the output negated.
        """
        raise NotImplementedError

    #### minimize ####

    def to_spins(self, i):
        """
        Converts a spin configuration index to the corresponding spin vector.

        Parameters
        ----------
        i : int
            The spin configuration index, which is a number between 0 and 2^N-1.

        Returns
        -------
        np.ndarray
            The configuration vectors corresponding to the given index.
            The values are in the `spin` variable type.
        """
        spins = [(i >> j) % 2 for j in range(self.N)]
        spins = np.array(spins) * 2 - 1
        return spins

    def minimize_brute_force(self):
        """
        Minimizes this objective function using a brute-force algorithm.
        Assumes the "spin" variable type for interpreting the target function.
        This function is only for debugging and testing.

        Returns
        -------
        opt : np.ndarray
            The optimal solution as a spin vector.
        value : np.number
            The function value for the optimal solution.
        """
        assert (self.N <= 20)
        L = (1 << self.N)
        i_opt = None
        E_opt = None
        for i in range(L):
            spins = self.to_spins(i)
            E = self.evaluate(spins)
            print(spins, E)
            if E_opt == None or E < E_opt:
                E_opt = E
                i_opt = i
        return self.to_spins(i_opt), E_opt
