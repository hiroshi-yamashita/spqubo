from . import fourier_interaction as sff
from . import cxx_spin_mapping as csm

import numpy as np

# L: int

# x, y:
#   spins: (N, ), dtype="f"

# pos_x, pos_y:
#    pos: (N, 2), dtype="i"
#    pos[i]: (y, x)

# f_J: (L, L), dtype="f"

# pad_x, pad_y: dtype="i"


def spatial_xJy_sparse(Ly, Lx, x, pos_x, f_J, y, pos_y):
    assert (pos_x.dtype == np.int32)
    assert (pos_y.dtype == np.int32)
    assert (f_J.dtype == np.float32)
    assert (x.dtype == np.float32)
    assert (y.dtype == np.float32)

    spins_x = csm.spinvector_to_spinarr(Ly, Lx, pos_x, x)
    spins_y = csm.spinvector_to_spinarr(Ly, Lx, pos_y, y)
    return sff._spatial_xJy(Ly*Lx, spins_x, f_J, spins_y)


def spatial_Jx_sparse(Ly, Lx, f_J, x, pos_x):
    assert (pos_x.dtype == np.int32)
    assert (f_J.dtype == np.float32)
    assert (x.dtype == np.float32)

    spins_x = csm.spinvector_to_spinarr(Ly, Lx, pos_x, x)
    a_arr = sff._spatial_Jx(f_J, spins_x)
    a = csm.spinarr_to_spinvector(Ly, Lx, pos_x, a_arr)
    return a


def spatial_xJy_dense(Ly, Lx, x, pad_x_y, pad_x_x, f_J, y, pad_y_y, pad_y_x):
    # same as "fou" mode
    return sff.spatial_xJy_dense(Ly, Lx, x, pad_x_y, pad_x_x, f_J, y, pad_y_y, pad_y_x)


def spatial_Jx_dense(Ly, Lx, f_J, x, pad_x_y, pad_x_x):
    # same as "fou" mode
    return sff.spatial_Jx_dense(Ly, Lx, f_J, x, pad_x_y, pad_x_x)
