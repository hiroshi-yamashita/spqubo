import pulp as lp
import numpy as np

from ..qmodel import qmodel_vanilla as qmv
from ..qmodel import spatial_qmodel_sparse as sqm
from . import solver as qs
nax = np.newaxis


class pulp_solver(qs.solver):
    def __init__(self, mode):
        """
        The QUBO solver using the Pulp library,
        which is a wrapper for the COIN-OR LP solver.

        Parameters
        ----------
        mode : Literal["max", "min"]
            The mode of the solver.
        """
        super().__init__(mode=mode, vartype="binary")

    def solve(self, q): # [implementation]
        if isinstance(q, sqm.spatial_qmodel_sparse):
            q = q.to_vanilla_qm()

        assert(isinstance(q, qmv.qmodel_vanilla))

        if self.mode == "max":
            prob = lp.LpProblem(sense=lp.LpMaximize)
        elif self.mode == "min":
            prob = lp.LpProblem(sense=lp.LpMinimize)
        else:
            raise ValueError

        N = q.N
        x = [lp.LpVariable(f'x_{i}', cat='Binary') for i in range(N)]

        y = {}
        obj = [q.const]
        for i in range(N):
            obj.append(q.h[i] * x[i])
        for i in range(N):
            for j in range(N):
                y[(i, j)] = lp.LpVariable(f'y_{i}_{j}', cat='Binary')
                obj.append(q.J[i, j] * y[i, j])
                prob += y[i, j] >= x[i] + x[j] - 1
                prob += x[i] >= y[i, j]
                prob += x[j] >= y[i, j]
        prob += lp.lpSum(obj)
        prob.solve()
        E_opt = lp.value(prob.objective)
        spins = np.array([xi.value() for xi in x])
        return spins, E_opt
