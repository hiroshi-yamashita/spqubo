import numpy as np


def Jh_to_ijc(J, h):
    """
    Converts a quadratic model from matrix representation (J, h) to list representation (i, j, c).
    The matrix representation uses a symmetric coupling matrix J and bias vector h, while the list representation uses lists of indices i, j and corresponding coefficients c.

    Parameters
    ----------
    J : np.ndarray
        A symmetric NxN coupling matrix J[i,j].
        Only the upper triangular part (i < j) is used.
    h : np.ndarray
        A vector of length N containing the bias terms for each spin.

    Returns
    -------
    tuple
        A tuple (N, i_list, j_list, c_list) where:
        - N : int
            The number of spins (dimension of the system)
        - i_list : list
            List of row indices for non-zero couplings and biases
        - j_list : list 
            List of column indices for non-zero couplings (-1 for bias terms)
        - c_list : list
            List of corresponding coupling/bias coefficients
    """
    N = J.shape[0]
    i_list = []
    j_list = []
    c_list = []
    for i in range(N):
        for j in range(N):
            if i < j and J[i, j] != 0:
                i_list.append(i)
                j_list.append(j)
                c_list.append(J[i, j])
    for i in range(N):
        if h[i] != 0:
            i_list.append(i)
            j_list.append(-1)
            c_list.append(h[i])
    return N, i_list, j_list, c_list


def ijc_to_Jh(N, i_list, j_list, c_list):
    """
    Converts a quadratic model from list representation (i, j, c) to matrix representation (J, h).
    The list representation uses lists of indices i, j and corresponding coefficients c, while the matrix representation uses a symmetric coupling matrix J and bias vector h.

    Parameters
    ----------
    N : int
        The number of spins (dimension of the system)
    i_list : list
        List of row indices for non-zero couplings and biases
    j_list : list 
        List of column indices for non-zero couplings (-1 for bias terms)
    c_list : list
        List of corresponding coupling/bias coefficients

    Returns
    -------
    tuple
        A tuple (J, h) where:
        - J : np.ndarray
            A symmetric NxN coupling matrix where J[i,j] represents the coupling strength between spins i and j
        - h : np.ndarray
            A vector of length N containing the bias terms for each spin
    """
    J = np.zeros((N, N), dtype="f")
    h = np.zeros((N, ), dtype="f")
    for i, j, c in zip(i_list, j_list, c_list):
        if i < 0 and j >= 0:
            h[j] = c
        elif i >= 0 and j < 0:
            h[i] = c
        elif i >= 0 and j >= 0:
            J[i, j] = c
            J[j, i] = c
        else:
            raise ValueError
    return J, h
