import contextlib

import numpy as np
from typing import Dict, Optional, List, Sequence


@contextlib.contextmanager
def fixedseed(seed):
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        yield
    finally:
        np.random.set_state(state)


@contextlib.contextmanager
def printoption(**kwargs):
    state = np.get_printoptions()
    try:
        np.set_printoptions(**kwargs)
        yield
    finally:
        np.set_printoptions(**state)


def save_log(filename: str,
             data_dict: Dict[str, np.ndarray],
             keys: Optional[Sequence[str]] = None,
             keys_ignore: Optional[Sequence[str]] = None,
             keys_timeseries: Sequence[int] = [],
             timepoints: Optional[Sequence[int]] = None,
             key_timepoints: Optional[str] = "time",
             keep_timepoints: bool = False) -> None:
    """
    save specified data of the log dictionary. 
    Timeseries data are sampled at the specified timepoints 
    and the sampled index are added to the saved dictionary.    

    Parameters
    ----------
    filename : str
        destination file name
    data_dict : Dict[str, np.ndarray]
        log dictionary to be saved
    keys : Optional[List[str]], optional
        the collection of keys to be included in the saved file, by default None 
    keys_ignore : Optional[List[str]], optional
        the collection of keys to be ignored in the saved file, by default None
        `keys` and `keys_ignore` can not be used at once. If both are None, all items are saved.
    keys_timeseries : Sequence[int], optional
        the collection of keys regarded as timeseries and sampling is applied, by default []
    timepoints : Optional[Sequence[int]], optional
        the collection of timepoints to be included in the saved file, by default None 
        specifying that no sampling is applied
    key_timepoints : Optional[str], optional
        the key of added item of sampled index, by default "time"
        If None, it is not added.
    keep_timepoints : bool, optional
        If True, specified timepoints are used and the exception is raised when the data is not included in the time window.
        If False, only valid timepoints are used and the saved timepoints may be different from the specified.
        By default False
    """
    ### resolve keys ###
    if keys:
        if keys_ignore:
            raise ValueError(
                "`keys` and `keys_ignore` can not be used at once")
    else:
        keys = list(data_dict.keys())
        if keys_ignore:
            keys = [k for k in keys if k not in keys_ignore]
    keys_timeseries = list(set(keys_timeseries) & set(keys))

    if len(keys_timeseries) > 0:
        ### check timeseries data ###
        for k in keys_timeseries:
            if not (1 <= len(data_dict[k].shape) <= 2):
                raise ValueError(
                    f"invalid shape in {k}: {data_dict[k].shape}, expected: (*, *)")

        ### check timeseries length ###
        L = data_dict[keys_timeseries[0]].shape[0]
        for k in keys_timeseries:
            if not data_dict[k].shape[0] == L:
                raise ValueError(
                    f"invalid shape in {k}: {data_dict[k].shape}, expected: ({L}, *)")

        ### resolve timepoints ###
        if timepoints is None:
            timepoints = np.arange(L)
        if keep_timepoints:
            assert(np.all(0 <= timepoints))
            assert(np.all(timepoints < L))
        else:
            timepoints = np.array([
                t for t in timepoints if (0 <= t < L)
            ])
    else:
        timepoints = None

    data = {}
    for k in keys:
        v = data_dict[k]
        if k in keys_timeseries:
            data[k] = v[list(timepoints)]
        else:
            data[k] = v
    if key_timepoints and timepoints is not None:
        assert(key_timepoints not in set(data.keys()))
        data[key_timepoints] = timepoints

    np.savez(filename, **data)
