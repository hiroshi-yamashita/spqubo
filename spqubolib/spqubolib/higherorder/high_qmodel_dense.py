from .rearrange2D import rearrange_helper_2D
from ..qmodel.constructor import spatial_qmodel_from_masked_J_and_h
from ..interaction import cxx_spin_mapping as csm


class high_qmodel_dense:
    def __init__(self, W_list, pad_list, J_tensor, h_tensor, xi_tensor, idx_y, idx_x, const=0, mode="naive"):
        """
        The quadratic objective function with D-dimensional dense spatial coupling matrix, 
        without the function of matrix operation.
        It supports obtaining the 2-dim qmodel from the D-dimensional model.
        with various conversion of positions depending on the specified dimension sets.

        Parameters
        ----------
        W_list : Sequential[int]
            the shape of spin array.
        pad_list : Sequential[int]
            the localities of the interaction function
        J_tensor : np.ndarray
            the tensor representing spatial interaction function
        h_tensor : np.ndarray
            1st-order coefficient 2-dim array (the bias), which has the shape of `W_list`
        xi_tensor : np.ndarray
            coefficient 2-dim array in spCM, which has the shape of `W_list`
        idx_y : np.ndarray
        idx_x : np.ndarray
            The indices of the dimension set for each axis in the obtained 2-dim qmodel.
        const : float, optional
            The constant term in the function, by default 0
        mode : Literal["naive", "fourier", "cxx_fourier"], optional
            The mode for matrix operation in the obtained 2-dim qmodel, by default "naive"
        """
        self.rh = rearrange_helper_2D(W_list, pad_list, idx_y, idx_x)
        J_arr = self.rh.Jtensor_to_Jarr(J_tensor)
        h_arr = self.rh.spintensor_to_spinarr(h_tensor)
        xi_arr = self.rh.spintensor_to_spinarr(xi_tensor)
        self.mask = self.rh.get_spinmask()
        self.q = spatial_qmodel_from_masked_J_and_h(
            self.mask, J_arr, h_arr, xi_arr, const=const, mode=mode)
        self.Ly = self.q.Ly
        self.Lx = self.q.Lx
        self.pos = self.q.pos

    def get_qmodel(self):
        """
        Retrieve the quadratic objective function with a 2-dimensional spatial coupling matrix
        converting the current high-dimensional model.

        Returns
        -------
        spatial_qmodel_sparse
            The constructed 2-dimensional sparse spatial quadratic model.
        """
        return self.q

    def spintensor_to_spinarr(self, x_tensor):
        """
        Rearrange a spin tensor to the array.

        Parameters
        ----------
        x_tensor : np.ndarray
            The input D-dimensional spin tensor
            The 1st axis in the result corresponds to `idx_y`,
            and the 2nd axis in the result corresponds to `idx_x`.

        Returns
        -------
        np.ndarray
            The rearranged spin array with shape (Ly, Lx), where Ly and Lx are the dimensions of the spatial coupling matrix.
        """
        return self.rh.spintensor_to_spinarr(x_tensor)

    def spinarr_to_spintensor(self, x_arr):
        """
        Rearrange a spin array to the tensor.

        Parameters
        ----------
        x_arr : np.ndarray
            The input spin array with shape (Ly, Lx).
            The 1st axis in the result corresponds to `group_idx_y`,
            and the 2nd axis in the result corresponds to `group_idx_x`.

        Returns
        -------
        np.ndarray
            The rearranged D-dimensional spin tensor.
        """
        return self.rh.spinarr_to_spintensor(x_arr)

    def spinarr_to_spinvector(self, x_arr):
        """
        Rearrange a spin array to the vector.
        Only the positions corresponding to the spin positions are taken into account.

        Parameters
        ----------
        x_arr : np.ndarray
            input 2-dim array

        Returns
        -------
        np.ndarray
            the rearranged spin vector
        """
        return csm.spinarr_to_spinvector(self.Ly, self.Lx, self.pos, x_arr)

    def spinvector_to_spinarr(self, x_vector):
        """
        Rearrange a spin vector to the array.

        Parameters
        ----------
        x_vector : np.ndarray
            input vector

        Returns
        -------
        np.ndarray
            Rearranged 2-dim array
        """
        return csm.spinvector_to_spinarr(self.Ly, self.Lx, self.pos, x_vector)

    def spintensor_to_spinvector(self, x_tensor):
        """
        Rearrange a spin tensor to the vector.
        Only the positions corresponding to the spin positions are taken into account.

        Parameters
        ----------
        x_tensor : np.ndarray
            The input D-dimensional spin tensor

        Returns
        -------
        np.ndarray
            The rearranged spin vector.
        """
        x_arr = self.spintensor_to_spinarr(x_tensor)
        return self.spinarr_to_spinvector(x_arr)

    def spinvector_to_spintensor(self, x_vector):
        """
        Rearrange a spin vector to the tensor.

        Parameters
        ----------
        x_vector : np.ndarray
            The input spin vector

        Returns
        -------
        np.ndarray
            The rearranged D-dimensional spin tensor.
        """
        x_arr = self.spinvector_to_spinarr(x_vector)
        return self.spinarr_to_spintensor(x_arr)
