import numpy as np
cimport numpy as cnp

cdef void call_spinvector_to_spinarr(
    int Ly,
    int Lx,
    int N,
    float[:] x,
    int[:, :] pos,
    float[:, :] arr
):
    _spinvector_to_spinarr(
            Ly, 
            Lx, 
            N, 
            &x[0],
            &pos[0, 0],
            &arr[0, 0]
        )



cdef void call_spinarr_to_spinvector(
    int Ly,
    int Lx,
    int N,
    float[:] x,
    int[:, :] pos,
    float[:, :] arr
):
    _spinarr_to_spinvector(
            Ly, 
            Lx, 
            N, 
            &x[0],
            &pos[0, 0],
            &arr[0, 0]
        )



def spinvector_to_spinarr(Ly, Lx, pos, x):
    arr = np.zeros((Ly, Lx), dtype="f")
    call_spinvector_to_spinarr(Ly, Lx, pos.shape[0], 
    x, 
    np.ascontiguousarray(pos), 
    arr)
    return arr
    
def spinarr_to_spinvector(Ly, Lx, pos, arr):
    x = np.zeros((pos.shape[0], ), dtype="f")
    call_spinarr_to_spinvector(Ly, Lx, pos.shape[0], 
    x, 
    np.ascontiguousarray(pos), 
    np.ascontiguousarray(arr))
    return x
