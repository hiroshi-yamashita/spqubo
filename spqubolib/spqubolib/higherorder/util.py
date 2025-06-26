import numpy as np


def make_diff_list(R_list):
    """
    Enumerate all vectors in the Cartesian product of intervals [-R_k, R_k].

    Parameters
    ----------
    R_list : Sequential[int]
        The localities of the interaction function with length D.

    Returns
    -------
    np.ndarray
        An (N, D)-shaped array of enumerated vectors, where N is the total number
        of combinations and D is the length of R_list.
    """
    D = len(R_list)
    R2 = R_list * 2 + 1
    diff_list = []
    for k in np.arange(D):
        newdim = tuple((i for i in np.arange(D) if i != k))
        diff = np.arange(R2[k])-R_list[k]
        diff = np.broadcast_to(
            np.expand_dims(diff, newdim),
            R2
        )
        diff = diff.ravel()
        diff_list.append(diff)
    diff_list = np.array(diff_list).T
    return diff_list


def make_dense_pos_list(N_list):
    """
    Enumerate all vectors in the Cartesian product of intervals [0, N_k).

    Parameters
    ----------
    N_list : List[int]
        The shape of the spin array with length D.

    Returns
    -------
    np.ndarray
        An (N, D)-shaped array of enumerated vectors, where N is the total number
        of combinations and D is the length of N_list.
    """

    D = len(N_list)
    pos_list = []
    for k in np.arange(D):
        newdim = tuple((i for i in np.arange(D) if i != k))
        pos = np.arange(N_list[k])
        pos = np.broadcast_to(
            np.expand_dims(pos, newdim),
            N_list
        )
        pos = pos.ravel()
        pos_list.append(pos)
    pos_list = np.array(pos_list).T
    return pos_list
