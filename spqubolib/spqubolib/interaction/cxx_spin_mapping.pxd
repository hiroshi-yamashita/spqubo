cimport numpy as cnp

cdef extern from "_cxx_spin_mapping.h" namespace "SPIN_MAPPING":
    cdef void _spinvector_to_spinarr(
        int Ly, 
        int Lx, 
        int N, 
        float *x,
        int *pos, 
        float *arr
    )
    cdef void _spinarr_to_spinvector(
        int Ly, 
        int Lx, 
        int N, 
        float *x,
        int *pos, 
        float *arr
    )

