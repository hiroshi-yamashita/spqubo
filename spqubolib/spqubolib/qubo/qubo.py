import numpy as np

from ..qmodel import qmodel as qm
nax = np.newaxis


def check_if_spin(x):
    return np.all(np.logical_or(x == -1, x == 1))


def check_if_binary(x):
    return np.all(np.logical_or(x == 0, x == 1))


def check_vartype(x, vartype):
    if vartype == "binary":
        return check_if_binary(x)
    elif vartype == "spin":
        return check_if_spin(x)
    else:
        raise ValueError


def convert_spins(x, prob, solver):
    """
    Convert the input variable values from the QUBO's variable type 
    to the solver's variable type.
    This function can be used for specifing the initial configuration to the solver.

    Parameters
    ----------
    x : np.ndarray
        Input array of variables in the QUBO's variable type.
    prob : qubo
        The QUBO object defining the current variable type.
    solver : solver
        The solver object defining the output variable type.

    Returns
    -------
    np.ndarray
        The converted variable values in the solver's variable type.
    """
    vartype_current = prob.vartype
    vartype_new = solver.vartype
    if vartype_current == vartype_new:
        return x
    if vartype_current == "binary" and vartype_new == "spin":
        return x*2-1  # binary -> spin
    if vartype_current == "spin" and vartype_new == "binary":
        return (x+1)//2  # spin -> binary


def convert_back_spins(x, prob, solver):
    """
    Convert the input variable values from the solver's variable type 
    back to the QUBO's variable type.
    This function is used as a backend for the `solve` method of the `qubo` class.

    Parameters
    ----------
    x : np.ndarray
        Input array of variables in the solver's variable type.
    prob : qubo
        The QUBO object defining the target variable type.
    solver : solver
        The solver object defining the output variable type.

    Returns
    -------
    np.ndarray
        The converted variable values in the QUBO's variable type.
    """
    vartype_current = prob.vartype
    vartype_new = solver.vartype
    if vartype_current == vartype_new:
        return x
    elif vartype_current == "binary" and vartype_new == "spin":
        return (x+1)//2  # spin -> binary
    elif vartype_current == "spin" and vartype_new == "binary":
        return x*2-1  # binary -> spin
    else:
        raise ValueError


class qubo:
    def __init__(self, q, mode="max", vartype="binary"):
        """
        A class that represents quadratic binary optimization problems. 
        Binary optimization problems have optimization modes and variable types 
        and they may differ between what the user and the solver assume. 
        This class absorbs the difference of the assumptions.

        Parameters
        ----------
        q : qmodel
            the target quadratic function.
        mode : Literal["max", "min"], optional
            A string representing whether the function should be maximized or minimized. by default "max"
        vartype : Literal["binary", "spin"], optional
            When "binary", the variables take values of either 0 or 1.
            When "spin", the variables take values of either -1 or +1.
            By default "binary"
        """
        assert (isinstance(q, qm.qmodel))
        self.q = q
        self.mode = mode
        self.vartype = vartype
        assert (mode == "max" or mode == "min")
        assert (vartype == "binary" or vartype == "spin")
        assert (mode == "max" or mode == "min")
        assert (vartype == "binary" or vartype == "spin")

    def solve(self, solver):
        """
        Solve the QUBO problem using the specified solver.

        Parameters
        ----------
        solver : solver
            The solver object to be used for solving the QUBO.

        Returns
        -------
        opt : np.ndarray
            1D array of the solution in the QUBO's variable type.
        value : np.number
            The target function value of the solution on the current QUBO.
        """

        # adjust qmodel variable type to what the solver assumes
        q_ = self.q.convert(self.vartype, solver.vartype)

        # adjust qmodel to the mode that the solver assumes
        coef = 1
        if self.mode != solver.mode:
            coef = -1
            q_ = q_.negate()

        # solve and check
        opt_, E_opt_ = solver.solve(q_)
        assert (np.isclose(E_opt_, q_.evaluate(opt_)))

        # adjust the solution variable type the solver assumes to what the qubo assumes
        opt = convert_back_spins(opt_, self, solver)

        # adjust the target function value to the mode what the qubo assumes
        E_opt = coef * E_opt_

        return opt, E_opt

    def evaluate(self, spins):
        """
        Evaluate the objective function value for a given spin configuration.

        This method evaluates the target function value for the given spin configuration.

        Parameters
        ----------
        spins : np.ndarray
            The spin configuration to evaluate. Must be compatible with the QUBO's variable type:
            - For binary type: values must be either 0 or 1.
            - For spin type: values must be either -1 or +1.

        Returns
        -------
        float
            The objective function value for the given spin configuration.

        Raises
        ------
        ValueError
            If the input spin configuration is not compatible with the QUBO's variable type.
        """
        assert (check_vartype(spins, self.vartype))
        return self.q.evaluate(spins)
