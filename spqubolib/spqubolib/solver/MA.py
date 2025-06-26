from ..utils import fixedseed, save_log
import numpy as np
import scipy.linalg as slin
import sys
nax = np.newaxis

# [TODO] maximum value of c (coupl. s.)


def MA_main(h,
            p_const1=0.5,
            p_const2=2000,
            beta0=(0.1*10**-1),
            c_const1=1000,
            p_generator=None,
            c_generator=None,
            T_generator=None,
            w_generator=None,
            K=2000,
            J=None,
            J_sp=None,
            w=None,
            seed=0,
            mode="fast",
            filename_log=None,
            save_ignore=[],
            **kwargs):
    """
    The main function of the Momentum Annealing (MA) algorithm.

    This maximizes $H = (1/2)x^\top Jx+hx$ for spins $x_i\in\{-1, +1\}$ for given $J$ and $h$. 

    See the docstrings of the generator functions for the parameters used in each step.
    Their parameters or themselves can be overridden by arguments.

    Parameters
    ----------
    h : np.ndarray
        1st-order coefficient vector (the bias vector).
    p_const1 : float, optional
        The initial value of the mask probability, by default 0.5.
    p_const2 : float, optional
        The time constant for the decrease of the mask probability, by default 2000.
    beta0 : float, optional
        The initial value of the inverse temperature, by default 0.01.
    c_const1 : float, optional
        The time constant for the increase of the inter-layer coupling strength, by default 1000.
    p_generator : Iterator[float], optional
    c_generator : Iterator[float], optional
    T_generator : Iterator[float], optional
        The generator functions for the mask probability, the strength of the inter-layer coupling,
        and temperature, respectively. 
        If these are specified, they are used instead of the parameters `p_const1`, `p_const2`, `c_const1`, and `beta0`.
    w_generator : Iterator[float], optional
        The generator function for the inter-layer coupling strength for each spin.
        If this is specified, the `mode` must be "generator".
    K : int, optional
        The length of the iteration, by default 2000.
    J : np.ndarray, optional
        2nd-order coefficient matrix (the interaction matrix), by default None.
    J_sp : np.ndarray, optional
        The spatial interaction matrix, by default None.
        This is for logging purposes only and directly passed to the output.
    w : np.ndarray, optional
        Inter-layer coupling strength for each spin, by default None.
    seed : int, optional
        The seed for the random number generator, by default 0.
    mode : Literal["naive", "fast", "spatial", "generator"], optional
        The mode for the calculation of the inter-layer coupling strength for each spin,
        by default "fast".
        - "naive": use the formula in the original paper (Okuyama et al., 2019).
        - "fast": compute the fast approximation value of the original method.
        - "spatial": compute the fast approximation value using the spatial interaction matrix.
        - "generator": use the generator function given as `w_generator`.
    filename_log : str, optional
        The filename for saving the log of the solving process, by default None.
        If this is specified, the function saves the log of the solving process.
        The log is saved as a numpy file whose keys are described in the `MA_sub` function.
    save_ignore : list, optional
        The list of keys to be ignored when saving the log, by default [].

    Returns
    -------
    s: np.ndarray
        The final spin configuration of the solver.
    """
    # maximize xJ_0x+h_0x
    # minimize -(1/2) xJ_1x　- h_1x
    #  maximize (1/2) x J_1 x + h_1 x
    # J_1 = J_0 * 2
    # h_1 = h_0

    if w_generator is not None and mode != "generator":
        raise ValueError(
            "`mode` must be \"generator\" when `w_generator` is specified")

    with fixedseed(seed):
        N,  = h.shape
        if w == None:
            if mode == "naive":
                w = calcuate_w(J)
            elif mode == "fast":
                w = calcuate_fast(J)
            elif mode == "spatial":
                w = calculate_w_spatial(N, J_sp)
            elif mode == "generator":
                w = w_generator
            else:
                raise ValueError

        if p_generator is None:
            p_generator = generate_p(p_const1=p_const1, p_const2=p_const2)
        if c_generator is None:
            c_generator = generate_c(c_const1=c_const1)  # strength of the link
        if T_generator is None:
            T_generator = generate_T(beta0)  # temperature
        p = p_generator
        c = c_generator
        T = T_generator

        init_s_prev = np.random.randint(2, size=(N, )) * 2 - 1
        # [TODO] better initialization?
        init_s_oppo = np.random.randint(2, size=(N, )) * 2 - 1
        s_init = (init_s_prev, init_s_oppo)

        s, log = MA_sub(h, w, s_init, K, p, c, T, J=J, J_sp=J_sp, **kwargs)
        if not filename_log is None:
            save_log(filename_log,
                     log,
                     keys_ignore=save_ignore,
                     keys_timeseries=["pk", "ck", "Tk", "s_log", "I_log", "t_log"])
        return s


def MA_sub(h, w, s_init, K, p, c, T, J=None, J_sp=None, fn_Jx=None, verbose=True, save_skip=1):
    """
    The working function of the Momentum Annealing (MA) algorithm.
    This function is called by the `MA_main` function with specified parameters.

    Parameters
    ----------
    h : np.ndarray
        1st-order coefficient vector (the bias vector).
    w : np.ndarray
        Inter-layer coupling strength for each spin.
    s_init : np.ndarray
        Initial spin configuration of the solver.
    K : int
        The number of iterations.
    p : Iterator[float]
    c : Iterator[float]
    T : Iterator[float]
        Iterators for the mask probability, the strength of the inter-layer coupling, 
        and temperature, respectively.
        They must generate at least `K` numbers.
    J : np.ndarray, optional
        2nd-order coefficient matrix (the interaction matrix), by default None.
        If `fn_Jx` is specified, this is ignored in the computation,
        but passed to the output for logging purposes.
    J_sp : np.ndarray, optional
        The spatial interaction matrix, by default None.
        This is for logging purposes only and directly passed to the output.
    fn_Jx : Callable[np.ndarray, np.ndarray], optional
        The function to calculate Jx, by default None, for the computing efficiency.
        If this is specified, the function is used instead of `J`.
    verbose : bool, optional
        Verbosity level, by default True.
        If True, the function prints the progress of the solving process.
    save_skip : int, optional
        The interval of the saving process, by default 1.
        If `save_skip` is 1, the function saves every step.
        If `save_skip` is greater than 1, the function saves every `save_skip` steps,
        starting from 0.

    Returns
    -------
    s: np.ndarray
        The final spin configuration of the solver.
    log: dict
        The dictionary containing the log of the solving process.
        The keys are:
        - "pk": the series of the mask probability.
        - "ck": the series of the strength of the inter-layer coupling.
        - "Tk": the series of the temperature.
        - "s_init": the initial spin configuration.
        - "s_log": the series of the spin configuration.
        - "I_log": the series of the force component values.
        - "t_log": the series of the step numbers.
        - "J": the 2nd-order coefficient matrix (the interaction matrix).
        - "J_sp": the spatial interaction matrix.
        - "h": the 1st-order coefficient vector (the bias vector).
        - "w": the inter-layer coupling strength for each spin.
    """
    s_prev, s_oppo = s_init
    # s_prev = s_{k-2}
    # s_oppo = s_{k-1}

    if fn_Jx is None:
        def fn_Jx_(s):
            return J @ s
        fn_Jx = fn_Jx_

    if verbose:
        print("minimizing -1/2 xJx - hx ...")
    N = s_prev.shape[0]
    log_length = (K-1) // save_skip + 1

    s_log = np.empty((log_length, N))
    I_log = np.empty((log_length, 5, N))
    params_log = np.empty((log_length, 3))
    t_log = np.empty((log_length, ))
    for k, pk, ck, Tk in zip(np.arange(K), p, c, T):
        if verbose and np.mod(k, (K//10)) == 0:
            print(
                f"step: {k}, p (mask prob.): {pk:.4f}, c (coupling s.): {ck:.4f}, T: {Tk:.4f}")
        # print("".join(["-+"[int((s+1)/2)] for s in s_oppo]),
        #       f"pk:{pk:.4f} ck:{ck:.4f} Tk:{Tk:.4f}")
        mask = np.random.random(size=w.shape) > pk
        w_tmp = ck * mask * w
        gammak = \
            np.random.gamma(1, scale=1, size=s_prev.shape)
        I1 = h
        I2 = fn_Jx(s_oppo)

        I3 = w_tmp * s_oppo
        I4 = -(Tk / 2) * gammak * s_prev
        I = I1 + I2 + I3 + I4

        s_new = np.sign(I)
        s_prev = s_oppo
        s_oppo = s_new
        if k % save_skip == 0:
            _k = k // save_skip
            s_log[_k, :] = s_new
            I_log[_k, :, :] = np.stack([I1, I2, I3, I4, I])
            params_log[_k, :] = (pk, ck, Tk)
            t_log[_k] = k
    if verbose:
        print("finished:")
        print(s_oppo[:5], (s_oppo[nax, :] @ fn_Jx(s_oppo)[:, nax])[0, 0])

    log = {
        "pk": params_log[:, 0],
        "ck": params_log[:, 1],
        "Tk": params_log[:, 2],
        "s_init": s_init,
        "s_log": s_log,
        "I_log": I_log,
        "t_log": t_log,
        "J": J,
        "J_sp": J_sp,
        "h": h,
        "w": w,
    }
    return s_oppo, log


def generate_T(beta0):
    """
    A generator function for the temperature.
    The temperature is generated by the formula:
    T = 1 / (beta0 * log(1 + k)),
    where k is the iteration number.

    Parameters
    ----------
    beta0 : float
        The initial value of the inverse temperature.

    Yields
    ------
    float
        The temperature at the current iteration.
    """
    k = 1
    while True:
        yield 1 / (beta0 * np.log(1+k))
        k = k + 1


def generate_p(p_const1=0.5, p_const2=2000):
    """
    A generator function for the mask probability.
    The mask probability is generated by the formula:
    p = max(0, p_const1 - k / p_const2),
    where k is the iteration number.

    Parameters
    ----------
    p_const1 : float, optional
        The initial value of the mask probability, by default 0.5.
    p_const2 : float, optional
        The time constant for the decrease of the mask probability, by default 2000.

    Yields
    ------
    float
        The mask probability at the current iteration.
    """
    k = 1
    while True:
        yield np.maximum(0, p_const1 - k / p_const2)
        k = k + 1


def generate_c(c_const1=1000):
    """
    A generator function for the strength of the inter-layer coupling.
    The strength of the inter-layer coupling is generated by the formula:
    c = min(1, sqrt(k / c_const1)),
    where k is the iteration number.

    Parameters
    ----------
    c_const1 : float, optional
        The time constant for the increase of the strength of the inter-layer coupling, by default 1000.

    Yields
    ------
    float
        The strength of the inter-layer coupling at the current iteration.
    """
    k = 1
    while True:
        yield np.minimum(1, np.sqrt(k/c_const1))
        k = k + 1


def calcuate_w(J):
    """
    Compute the inter-layer coupling strength for each spin,
    following the formula in the original paper (Okuyama et al., 2019).

    Parameters
    ----------
    J : np.ndarray
        2nd-order coefficient matrix (the interaction matrix).

    Returns
    -------
    np.ndarray
        The inter-layer coupling strength for each spin.
    """
    N = J.shape[0]
    lamb = slin.eigh(-J,
                     subset_by_index=[N-1, N-1],
                     eigvals_only=True)
    print("lamb:", lamb)
    absj = np.abs(J)
    sum1 = np.sum(absj, axis=1)

    flag_c = lamb >= sum1

    sum2 = np.sum(absj[:, flag_c], axis=1)

    w = sum1 - sum2 / 2
    w[~flag_c] = lamb / 2
    return w


def calcuate_fast(J):
    """
    Compute the inter-layer coupling strength for each spin,
    using a fast approximation of the original work.
    The coupling strength is set to be the half of the row-wise sum of the absolute value of the coupling matrix,
    which are less than the original work.
    
    Parameters
    ----------
    J : np.ndarray
        2nd-order coefficient matrix (the interaction matrix).

    Returns
    -------
    np.ndarray
        The inter-layer coupling strength for each spin.
    """
    N = J.shape[0]
    absj = np.abs(J)
    sum1 = absj @ np.ones((N, ))
    w = np.full((N, ), sum1 / 2)
    return w


def calculate_w_spatial(N, J):
    """
    Compute the inter-layer coupling strength for each spin using the spatial interaction matrix 
    and the fast approximation of the original work.
    The coupling strength is set to be the half of the row-wise sum of the absolute value of the coupling matrix,
    which are less than the original work.

    Parameters
    ----------
    N : int
        The number of spins.
    J : np.ndarray
        2nd-order coefficient matrix (the interaction matrix).

    Returns
    -------
    np.ndarray
        The inter-layer coupling strength for each spin.
    """
    absj = np.abs(J)
    sum1 = np.sum(absj)

    w = np.full((N, ), sum1 / 2)
    return w

