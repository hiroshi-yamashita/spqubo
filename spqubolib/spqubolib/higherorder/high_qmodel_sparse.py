from .rearrange2D import rearrange_helper_2D
from ..qmodel.spatial_qmodel_sparse import spatial_qmodel_sparse


class high_qmodel_sparse:
    def __init__(self, W_list, pad_list, J_tensor, h, xi, pos_high, idx_y, idx_x, const=0, mode="naive"):
        """
        The quadratic objective function with D-dimensional sparse spatial coupling matrix (spCM), 
        without the function of matrix operation.
        It supports obtaining the 2-dim qmodel from the D-dimensional model.
        with various conversion of positions depending on the specified dimension sets.

        Parameters
        ----------
        W_list : Sequential[int]
            the shape of spin array with length D
        pad_list : Sequential[int]
            the localities of the interaction function with length D
        J_tensor : np.ndarray
            the D-dimensional tensor representing spatial interaction function
        h : np.ndarray
            1st-order coefficient vector (the bias)
        xi : np.ndarray
            coefficient vector in the spCM
        pos_high : np.ndarray
            (N, D) array of spin positions
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

        Ly, Lx = J_arr.shape
        pos = self.rh.py_poshigh_to_pos2D(pos_high).astype("i")
        self.q = spatial_qmodel_sparse(
            Ly, Lx, J_arr, h, xi, pos, const=const, mode=mode
        )

    def get_qmodel(self):
        """
        Retrieve the quadratic objective function with a 2-dimensional sparse spatial coupling matrix
        converting the current high-dimensional model.
        Although the current model is dense model, the obtained model is sparse model 
        because of the conversion of the spin positions.

        Returns
        -------
        spatial_qmodel_sparse
            The constructed 2-dimensional sparse spatial quadratic model.
        """
        return self.q
