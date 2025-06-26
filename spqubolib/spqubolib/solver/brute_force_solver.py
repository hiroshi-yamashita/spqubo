import numpy as np
from .solver import solver
import sys


def index_to_spins(i, N):
    """
    Convert a spin configuration index to a spin vector.

    Parameters
    ----------
    i : int
        The spin configuration index, which is a number between 0 and 2^N-1.
    N : int
        The number of spins.

    Returns
    -------
    np.ndarray
        The spin vector of the spin configuration in the `spin` variable type.
    """
    spins = [(i >> j) % 2 for j in range(N)]
    spins = np.array(spins) * 2 - 1
    return spins


def index_list_to_spins(i, N):
    """
    Convert a list of spin configuration indices to spin vectors.

    Parameters
    ----------
    i : np.ndarray
        The spin configuration indices, which are numbers between 0 and 2^N-1.
    N : int
        The number of spins.

    Returns
    -------
    np.ndarray
        The spin array of the spin configurations in the `spin` variable type.
        The shape of the array is (L, N), where L is the length of the input.
    """
    spins = [(i >> j) % 2 for j in range(N)]
    spins = np.array(spins).T
    spins = spins * 2 - 1
    return spins


class brute_force_solver(solver):
    def __init__(self, verbose=1):
        """
        The QUBO solver using a brute force algorithm,
        which evaluates all possible solutions and returns the best one.

        Parameters
        ----------
        verbose : int, optional
            The verbosity level of the solver, by default 1.
            If verbose > 0, the solver prints the progress of the solving process.
            If verbose > 1, the solver prints all steps of the solving process.
        """
        super().__init__(mode="min", vartype="spin")
        self.verbose = verbose

    def solve(self, q): # [implementation]
        assert(q.N <= 20)
        N = q.N
        L = (1 << N)
        i_opt = None
        E_opt = None

        index_list = np.arange(L)
        spins_list = index_list_to_spins(index_list, N)

        if self.verbose > 0:
            print("J:", q.J)
            print("h:", q.h)
            print("const:", q.const)

        for i in range(L):
            spins = spins_list[i, :]
            E = q.evaluate(spins)
            if self.verbose > 0:
                if np.mod(i, L // 20) == 0:
                    print(f"step {i}/{L}", file=sys.stderr)
            if self.verbose > 1:
                print(f"step {i}/{L}", file=sys.stderr)
                print(spins, E, file=sys.stderr)
            if E_opt == None or E < E_opt:
                E_opt = E
                i_opt = i
        return index_to_spins(i_opt, N), E_opt
