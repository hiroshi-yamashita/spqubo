class solver:
    def __init__(self, mode="max", vartype="binary"):
        """
        The base class for the QUBO solver.
        The solver assumes the minimization/maximization mode and the variable type.

        Parameters
        ----------
        mode : Literal["max", "min"], optional
            The mode of the solver, by default "max".
        vartype : Literal["binary", "spin"], optional
            The type of variable the solver assumes, by default "binary".
        """
        self.mode = mode
        self.vartype = vartype
        assert(mode == "max" or mode == "min")
        assert(vartype == "binary" or vartype == "spin")

    def get_mode(self):
        """
        Get the mode of the solver, which it assumes for the target function.

        Returns
        -------
        Literal["max", "min"]
            The mode of the solver.
        """
        return self.mode

    def get_vartype(self):
        """
        Get the type of variable the solver assumes.

        Returns
        -------
        Literal["binary", "spin"]
            The variable type the solver assumes.
        """
        return self.vartype

    def solve(self, q):
        """
        Solve the optimization problem defined by the qmodel.
        The solver assumes the minimization/maximization mode and the variable type.
        Thus, the qmodel should be converted before passing to the solver
        for the assumed mode and variable type.

        Parameters
        ----------
        q : qmodel
            The quadratic function to be solved.

        Returns
        -------
        opt : np.ndarray
            1D array of the obtained solution, which is either binary or spin depending on the solver.
        value : np.number
            The target function value for `opt`.
        """
        raise NotImplementedError

