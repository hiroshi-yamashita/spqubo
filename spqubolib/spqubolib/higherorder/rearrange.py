import numpy as np
from .util import make_diff_list, make_dense_pos_list


class rearrange_helper:
    def __init__(self, N_list, R_list):
        """
        Helper object for rearranging spin positions to one dimension.

        This handles the rearrangement to one dimension. For 2D rearrangement,
        two instances of this class must be combined.

        Parameters
        ----------
        N_list : Sequential[int]
            The shape of the spin array with length D.
        R_list : Sequential[int]
            The localities of the interaction function with length D.
        """

        assert (len(N_list) == len(R_list))
        self.D = len(N_list)
        if isinstance(N_list, list):
            N_list = np.array(N_list)
        if isinstance(R_list, list):
            R_list = np.array(R_list)

        self.N = N_list
        self.R = R_list

        self.R2 = self.R*2+1
        self.L = self.N + self.R
        self.sh_arr = self.N
        self.sh_J = self.R2

        self.dense_pos_list = make_dense_pos_list(self.N)

        self.diff_list = make_diff_list(self.R)

        # computes coefficients for converting multi-dimensional indices 
        # to a single flattened index.
        # Specifically, C_i = \prod_{j=i+1}^{D-1} L_j 
        # where L_j is the padded dimension size
        self.coef = np.concatenate([
            np.cumprod(self.L[::-1])[::-1],
            np.array([1], dtype="i")
        ])[1:]
        self.L_all = np.prod(self.L)

    def rearrange_pos_sparse(self, pos_list):
        """
        Convert a position list to a rearranged position list.

        This is used for the sparse representation of the spin array.

        Parameters
        ----------
        pos_list : np.ndarray
            Position list with shape (N, D).

        Returns
        -------
        np.ndarray
            Rearranged position list with shape (N, D).
        """
        
        K = pos_list.shape[0]
        ret = []
        for k in np.arange(K):
            pos = pos_list[k]
            ret.append(np.sum(self.coef * pos))
        ret = np.array(ret)
        ret = np.mod(ret, self.L_all)
        return ret
