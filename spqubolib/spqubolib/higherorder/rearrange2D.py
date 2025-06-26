import numpy as np
from .util import make_diff_list
from .rearrange import rearrange_helper
# , make_dense_pos_list


class rearrange_helper_2D:
    def __init__(self, N_list, R_list, group_idx_y, group_idx_x):
        """
        Helper object for rearranging spin positions in 2D.

        Parameters
        ----------
        N_list : Sequential[int]
            The shape of the spin array with length D.
        R_list : Sequential[int]
            The localities of the interaction function with length D.
        group_idx_y : List[int]
            Indices of dimensions for the 1st axis in the rearranged result.
        group_idx_x : List[int]
            Indices of dimensions for the 2nd axis in the rearranged result.
        """
        assert (isinstance(group_idx_y, list))
        assert (isinstance(group_idx_x, list))

        assert (len(N_list) == len(R_list))
        self.N_list = np.array(N_list)
        self.R_list = np.array(R_list)
        self.L_list = self.N_list + self.R_list
        self.D = len(self.N_list)

        self.group_idx_y = group_idx_y
        self.group_idx_x = group_idx_x
        self.rh_y = rearrange_helper(
            [N_list[i] for i in group_idx_y],
            [R_list[i] for i in group_idx_y],
        )
        self.rh_x = rearrange_helper(
            [N_list[i] for i in group_idx_x],
            [R_list[i] for i in group_idx_x],
        )
        self.Ly = self.rh_y.L_all
        self.Lx = self.rh_x.L_all

    #### spin conversion ####

    #### sparse ####

    def py_poshigh_to_pos2D(self, pos_high):
        """
        Rearrange high-dimensional positions to 2D positions.

        Parameters
        ----------
        pos_high : np.ndarray
            High-dimensional positions with shape (N, D).

        Returns
        -------
        np.ndarray
            2D positions with shape (N, 2).
        """
        pos_high_y = pos_high[:, self.group_idx_y]
        pos_high_x = pos_high[:, self.group_idx_x]
        pos_y = self.rh_y.rearrange_pos_sparse(pos_high_y)
        pos_x = self.rh_x.rearrange_pos_sparse(pos_high_x)
        return np.stack([pos_y, pos_x], axis=1)

    #### dense ####

    def get_spinmask(self):
        """
        Get the mask of the spin array.

        The mask is 1 for positions corresponding to spin domains in the
        D-dimensional space and 0 for positions outside.

        Returns
        -------
        np.ndarray
            Mask of the spin array with shape (Ly, Lx).
        """
        mask_tensor = np.ones(tuple(self.N_list))
        return self.spintensor_to_spinarr(mask_tensor)

    def spintensor_to_spinarr(self, tensor):
        """
        Rearrange a D-dimensional spin tensor to a 2D spin array.

        Parameters
        ----------
        tensor : np.ndarray
            Input D-dimensional spin tensor.

        Returns
        -------
        np.ndarray
            Rearranged spin array with shape (Ly, Lx).
        """
        _tensor = np.transpose(tensor, self.group_idx_y+self.group_idx_x)
        pos_list_y = np.sum(self.rh_y.dense_pos_list *
                            self.rh_y.coef[np.newaxis, :], axis=1)
        pos_list_x = np.sum(self.rh_x.dense_pos_list *
                            self.rh_x.coef[np.newaxis, :], axis=1)

        ret = np.zeros((self.rh_y.L_all, self.rh_x.L_all), dtype="f")

        for y, p_y in zip(pos_list_y, self.rh_y.dense_pos_list):
            for x, p_x in zip(pos_list_x, self.rh_x.dense_pos_list):
                p = tuple(p_y) + tuple(p_x)
                ret[y, x] = _tensor[p]
        return ret

    def spinarr_to_spintensor(self, arr):
        """
        Convert a 2D spin array back to a D-dimensional spin tensor.

        Parameters
        ----------
        arr : np.ndarray
            Input spin array with shape (Ly, Lx).

        Returns
        -------
        np.ndarray
            Rearranged D-dimensional spin tensor.
        """
        pos_list_y = np.sum(self.rh_y.dense_pos_list *
                            self.rh_y.coef[np.newaxis, :], axis=1)
        pos_list_x = np.sum(self.rh_x.dense_pos_list *
                            self.rh_x.coef[np.newaxis, :], axis=1)

        ret = np.zeros(
            tuple(self.N_list[self.group_idx_y+self.group_idx_x]), dtype="f")

        for y, p_y in zip(pos_list_y, self.rh_y.dense_pos_list):
            for x, p_x in zip(pos_list_x, self.rh_x.dense_pos_list):
                p = tuple(p_y) + tuple(p_x)
                ret[p] = arr[y, x]
        return ret

    #### J conversion ####
    def Jtensor_to_Jarr(self, Jtensor):
        """
        Rearrange a spatial interaction tensor to a 2D array.

        Parameters
        ----------
        Jtensor : np.ndarray
            Input D-dimensional tensor.

        Returns
        -------
        np.ndarray
            Rearranged array with shape (Ly, Lx).
        """
        _Jtensor = np.transpose(Jtensor, self.group_idx_y+self.group_idx_x)
        pos_list_y = np.sum(self.rh_y.diff_list *
                            self.rh_y.coef[np.newaxis, :], axis=1)
        pos_list_x = np.sum(self.rh_x.diff_list *
                            self.rh_x.coef[np.newaxis, :], axis=1)

        ret = np.zeros((self.rh_y.L_all, self.rh_x.L_all), dtype="f")

        for y, d_y in zip(pos_list_y, self.rh_y.diff_list):
            for x, d_x in zip(pos_list_x, self.rh_x.diff_list):
                d = tuple(d_y) + tuple(d_x)
                ret[y, x] = _Jtensor[d]

        return ret


    #### build J tensor ####

    def distribute_to_corners(self, x):
        """
        Distribute tensor content to the corners of an extended tensor.

        Parameters
        ----------
        x : np.ndarray
            Input D-dimensional tensor with shape (R1*2+1, ..., RD*2+1).

        Returns
        -------
        np.ndarray
            Output D-dimensional tensor with shape (L1, ..., LD).
        """
        # make the (2L,...,2L)-array that has x content around (L,...,L)
        ret = np.zeros(tuple(self.L_list * 2), dtype="f")
        sli = []
        for k in np.arange(self.D):
            L = self.L_list[k]
            R = self.R_list[k]
            sli.append(slice(L-R, L+R+1))
        sli = tuple(sli)
        ret[sli] = x

        # fold the array dimension-wise into (L,...,L)-array,
        # such that the content is distributed to the corners
        for k in np.arange(self.D):
            L = self.L_list[k]
            sli1 = [slice(None)] * self.D
            sli2 = [slice(None)] * self.D
            sli1[k] = slice(0, L)
            sli2[k] = slice(L, L*2)
            sli1 = tuple(sli1)
            sli2 = tuple(sli2)
            ret = ret[sli1] + ret[sli2]
        return ret

    # [TODO] make it faster
    def function_to_J_higher(self, fn):
        """
        Generate a spatial coupling tensor from a function.

        Only values of the function within the specified localities are considered.

        Parameters
        ----------
        fn : callable
            Input function.

        Returns
        -------
        np.ndarray
            Spatial coupling tensor with shape (L1, L2, ..., LD).
        """
        diff_list = make_diff_list(self.R_list)

        def _fn(x):
            return fn(*x)

        x = np.array(list(map(_fn, diff_list)), dtype="f")
        sh = self.R_list*2+1
        x = x.reshape(sh)
        return self.distribute_to_corners(x)

    def check_J_tensor(self, Jtensor):
        """
        Validate a spatial coupling tensor.

        Checks if the tensor is symmetric, zero at the origin, and zero outside
        the specified region.

        Parameters
        ----------
        Jtensor : np.ndarray
            Tensor to be checked.

        Raises
        ------
        ValueError
            If the tensor fails any of the checks.
        """
        idx_origin = tuple([0] * self.D)
        try:
            assert (np.isclose(Jtensor[idx_origin], 0))
        except Exception as e:
            raise ValueError("J_tensor at origin must be zero") from e

        mask = np.ones(tuple(self.R_list * 2 + 1))
        mask = self.distribute_to_corners(mask)

        mask = 1 - mask
        try:
            assert (np.all(mask * Jtensor == 0))
        except Exception as e:
            raise ValueError(
                "nonzero value outside the specified region") from e

        Jtensor_expanded = Jtensor
        for k in np.arange(self.D):
            Jtensor_expanded = np.concatenate(
                [Jtensor_expanded, Jtensor_expanded], axis=k)
        sli = tuple((slice(1, None) for k in np.arange(self.D)))
        Jtensor_expanded = Jtensor_expanded[sli]

        sli_reverse = tuple((slice(None, None, -1)
                            for k in np.arange(self.D)))
        try:
            assert (np.all(Jtensor_expanded == Jtensor_expanded[sli_reverse]))
        except Exception as e:
            raise ValueError("not symmetric") from e
