import numpy as np

from ..qmodel import spatial_qmodel_sparse as sqm
from ..qmodel import spatial_qmodel_dense as sqmd
from . import solver as qs
from . import MA
nax = np.newaxis


class MA_solver(qs.solver):
    def __init__(self, save_log=False, **kwargs):
        """
        The QUBO solver using the Momentum Annealing (MA) algorithm.

        Parameters
        ----------
        save_log : bool, optional
            Whether to save the log of the solving process, by default False.
        kwargs : dict, optional
            Additional parameters for the MA algorithm.
        """
        super().__init__(mode="max", vartype="spin")
        self.kwargs = kwargs
        self.save_log = save_log

    def solve(self, q): # [implementation]
        # MA_main minimizes -1/2 xJx - hx, while the qmodel represents xJx + hx, 
        # Thus the coefficients in J are twiced before it is passed to MA_main.

        def fn_Jx(s):
            return q.evaluate_Jx(s) * 2  # [1]

        if isinstance(q, sqm.spatial_qmodel_sparse) or isinstance(q, sqmd.spatial_qmodel_dense):
            h, J_sp = q.h, q.J * 2  # [1]
            if not "mode" in self.kwargs:
                self.kwargs["mode"] = "spatial"
            s = MA.MA_main(h, J_sp=J_sp, fn_Jx=fn_Jx, **(self.kwargs))
        else:
            h, J = q.h, q.J * 2  # [1]
            if not "mode" in self.kwargs:
                self.kwargs["mode"] = "fast"
            s = MA.MA_main(h, J=J, fn_Jx=fn_Jx, **(self.kwargs))

        return s, q.evaluate(s)
