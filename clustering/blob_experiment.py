import sys
import time
import contextlib
from typing import Sequence

import numpy as np
nax = np.newaxis
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm

from pathlib import Path
path = Path("../..").resolve().as_posix()
sys.path.append(path)
if True:
    from spqubolib.higherorder.high_qmodel_sparse import high_qmodel_sparse
    from spqubolib.qmodel.spatial_qmodel_sparse import spatial_qmodel_sparse
    from spqubolib.higherorder.rearrange2D import rearrange_helper_2D


class timer_class:
    def __init__(self):
        self.t = None


@contextlib.contextmanager
def timer():
    start_time = time.process_time()
    t = timer_class()
    try:
        yield t
    finally:
        end_time = time.process_time()
        elapsed_time = end_time - start_time
        t.t = elapsed_time
    return t


### generate QUBO from position array ###

def build_q(B: int, D: int, K: int, pos_list: np.ndarray, Ainte: float, Acnt: float) -> spatial_qmodel_sparse:
    """
    Build a sparse qmodel representing the clustering problem with the given data points.

    Parameters
    ----------
    B : int
        The size of the grid, identical for all dimensions.
    D : int
        The dimension of the data space.
    K : int
        The number of clusters.
    pos_list : np.ndarray
        The array of data point positions with shape (N, D).
    Ainte : float
        Coefficient for the clustering metric.
    Acnt : float
        Coefficient for the penalty of exact-one constraints.

    Returns
    -------
    spatial_qmodel_sparse
        A sparse qmodel representing the clustering problem.
    """
    assert (D == 2)

    idx_y = [0]
    idx_x = [2, 1]
    # space dimension + group dimension(=1)
    assert (len(idx_y) + len(idx_x) == (D + 1))

    N_spins = pos_list.shape[0]

    N_list = [B] * D + [K]
    R_list = [B-1] * D + [K-1]

    ### interaction function ###

    # The coefficients are halved from the paper,
    # since spqubolib assumes H = 1 * xWx + hx instead of H = 1/2 * xWx + hx.

    def fn2(diff_a, diff_b, cls):
        Linte = (cls == 0).astype("f") * np.sqrt(diff_a**2 + diff_b**2) / (B-1)
        Lcnt = (diff_a == 0).astype("f") * \
            (diff_b == 0).astype("f") * \
            (cls != 0).astype("f")
        ### ### merge two components ### ###
        return Ainte * Linte + Acnt * Lcnt

    def fn_inte(diff_a, diff_b, cls):
        Linte = (cls == 0).astype("f") * np.sqrt(diff_a**2 + diff_b**2) / (B-1)
        ### ### merge two components ### ###
        return Linte

    def fn_cnt(diff_a, diff_b, cls):
        Lcnt = (diff_a == 0).astype("f") * \
            (diff_b == 0).astype("f") * \
            (cls != 0).astype("f")
        ### ### merge two components ### ###
        return Lcnt

    print("function_to_J_higher")

    ### high qmodel parameters ###

    W_list = N_list
    pad_list = R_list

    ### ### spatial interaction matrix ### ###
    rh = rearrange_helper_2D(N_list, R_list, idx_y, idx_x)
    J_tensor = rh.function_to_J_higher(fn2)
    J_tensor_inte = rh.function_to_J_higher(fn_inte)
    J_tensor_cnt = rh.function_to_J_higher(fn_cnt)

    ### ### get h and const by merging two components ### ###
    h_inte = np.zeros((N_spins, ), dtype="f")
    h_cnt = np.zeros((N_spins, ), dtype="f")
    # (x1 + x2 - 1)^2 = x1x2 + x2x1 + (x1^2 + x2^2) - 2x1 - 2x2 + 1 = x1x2 + x2x1 + (-2 + 1)(x1 + x2) + 1 =
    h_cnt[:] = -2 + 1
    h = Ainte * h_inte + Acnt * h_cnt

    const_inte = 0
    const_cnt = N_spins // K
    const = Ainte * const_inte + Acnt * const_cnt

    ### ### spins have the equal weights ### ###
    xi = np.ones((N_spins, ), dtype="f")

    ### build qmodel ###

    print("build high_qmodel_sparse")
    hq = high_qmodel_sparse(W_list,
                                pad_list,
                                J_tensor.astype("f"),
                                h.astype("f"),
                                xi.astype("f"),
                                pos_list.astype("int32"),
                                idx_y, idx_x,
                                const=const)
    q = hq.get_qmodel()

    hq_inte = high_qmodel_sparse(W_list,
                                     pad_list,
                                     J_tensor_inte.astype("f"),
                                     h_inte.astype("f"),
                                     xi.astype("f"),
                                     pos_list.astype("int32"),
                                     idx_y, idx_x,
                                     const=const_inte)
    q_inte = hq_inte.get_qmodel()

    hq_cnt = high_qmodel_sparse(W_list,
                                    pad_list,
                                    J_tensor_cnt.astype("f"),
                                    h_cnt.astype("f"),
                                    xi.astype("f"),
                                    pos_list.astype("int32"),
                                    idx_y, idx_x,
                                    const=const_cnt)
    q_cnt = hq_cnt.get_qmodel()

    return q, q_inte, q_cnt


# spqubolib.highorder.util.make_dense_pos_list can be used,
# but we keep this source self-contained

def make_dense_pos_list(N_list):
    """
    Enumerate all vectors in the Cartesian product of ranges.

    Parameters
    ----------
    N_list : List[int]
        The shape of the spin array with length D.

    Returns
    -------
    np.ndarray
        An (N, D)-shaped array of enumerated vectors.
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


### generate clustering problem with known clusters ###

def generate_problem(rng: np.random.RandomState, K: int, B_list: Sequence[int], points_for_class: int) -> pd.DataFrame:
    """
    Generate a clustering problem with known clusters.

    Parameters
    ----------
    rng : np.random.RandomState
        A random number generator for reproducibility.
    K : int
        The number of clusters.
    B_list : Sequence[int]
        The problem sizes for each dimension.
    points_for_class : int
        The number of points for each cluster.

    Returns
    -------
    pd.DataFrame
        A dataframe containing data points and their cluster indices.

    Index
    -------
    idx : int
        The index of the data point from all possible data points.

    Columns
    -------
    cls : int
        The cluster index.
    (Other columns) : int
        The i-th dimension value of the position.
    """
    ### report problem size information ###
    possible_points = np.prod(B_list)
    possible_spins = np.prod(B_list) * K
    arrsize = np.prod(np.array(B_list) * 2 - 1) * (K*2 - 1)
    print(f"possible_points: {possible_points}")
    print(f"possible_spins: {possible_spins}")
    print(f"arr size: {arrsize}")

    D = len(B_list)
    pos_list = make_dense_pos_list(B_list)
    nax = np.newaxis
    rel_pos_list = pos_list / (B_list-1)[nax, :]

    ### determine cluster centers and prepare for drawing data points ###

    ### fixed positions of cluster center ###
    assert (K == 7)
    mu = np.array([
        [0.1, 0.1], # (1, 1)
        [0.1, 0.9], # (1, 9)
        [0.9, 0.1], # (9, 1)
        [0.9, 0.9], # (9, 9)
        [0.3, 0.5], # (3, 5)
        [0.6, 0.3], # (6, 3)
        [0.6, 0.7], # (6, 7)
    ])

    ### blob size ###
    scale = np.ones((K, D)) * 0.1 # σ = 0.1

    ### density function for each blob ###
    logpdf = np.sum(norm.logpdf(
        rel_pos_list[:, nax, :], # (N, K, D)
        loc=mu[nax, :, :],
        scale=scale[nax, :, :]
    ), axis=2) # (N, K)

    ### the blob with the highest density is used for each position ###
    cls = np.argmax(logpdf, axis=1) # (N, )

    ### draw data points ###
    drawn = []
    for k in range(K):
        ### take candidates ###
        idx = np.arange(len(cls))[cls == k] # (N_k, )

        ### normalize probability ###
        assert (len(idx) > 0)
        logpdf_k = (logpdf[:, k])[idx] # (N_k, )
        prob = np.exp(logpdf_k - logsumexp(logpdf_k)) # (N_k, )

        ### draw ###
        x = rng.choice(idx, size=points_for_class, replace=False, p=prob) # (M, )

        ### collect result ###
        df = pd.DataFrame({"idx": x})
        df["cls"] = k
        drawn.append(df)
    drawn = pd.concat(drawn, axis=0)

    ### format the result ###
    df_pos = pd.DataFrame(pos_list)  # [TODO] (refactoring) specify column names explicitly
    df_pos_drawn = df_pos.reindex(drawn["idx"].values)
    df_pos_drawn["cls"] = drawn["cls"].values

    return df_pos_drawn
