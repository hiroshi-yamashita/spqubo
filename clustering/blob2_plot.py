import numpy as np
nax = np.newaxis
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from spqubolib.interaction import spin_mapping_py as psm


def savefig(filename, fig):
    """
    Save a matplotlib figure to a file.

    Parameters
    ----------
    filename : str
        The file path to save the figure.
    fig : matplotlib.figure.Figure
        The figure to save.
    """
    if str.endswith(filename, ".pdf"):
        fig.savefig(filename,  metadata={"CreationDate": None})
    else:
        fig.savefig(filename, dpi=360)

def _plot_c(q, fig, ax):
    handle = ax.pcolormesh(psm.spinvector_to_spinarr(
        q.Ly, q.Lx, q.pos, np.ones(q.N)))
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="7%", pad="2%")
    cb = fig.colorbar(handle, cax=cax)
    cax.tick_params(labelsize=15)
    ax.tick_params(labelsize=15)
    ax.set_ylabel("$d_2$", fontsize=15)
    ax.set_aspect(1)


def _plot_d(q, fig, ax):
    q_arr = -q.J
    x_axis = np.fft.fftfreq(q_arr.shape[1], d=1.0/q_arr.shape[1])
    y_axis = np.fft.fftfreq(q_arr.shape[0], d=1.0/q_arr.shape[0])
    q_arr = np.fft.fftshift(q_arr)
    x_axis = np.fft.fftshift(x_axis)
    y_axis = np.fft.fftshift(y_axis)

    handle = ax.pcolormesh(x_axis, y_axis, q_arr, vmin=-1, shading='nearest')
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="7%", pad="2%")
    cb = fig.colorbar(handle, cax=cax)
    cax.tick_params(labelsize=15)
    ax.tick_params(labelsize=15)
    ax.set_ylabel("$d_2$", fontsize=15)
    ax.set_aspect(1)



def _plot_e(q, fig, ax):
    q_arr = -q.get_f_J()
    x_axis = np.fft.fftfreq(q_arr.shape[1], d=1.0/q_arr.shape[1])
    y_axis = np.fft.fftfreq(q_arr.shape[0], d=1.0/q_arr.shape[0])
    q_arr = np.fft.fftshift(q_arr)
    x_axis = np.fft.fftshift(x_axis)
    y_axis = np.fft.fftshift(y_axis)

    handle = ax.pcolormesh(x_axis, y_axis, q_arr, vmin=-
                           1000, vmax=1000, shading='nearest')
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="7%", pad="2%")
    cb = fig.colorbar(handle, cax=cax)
    cax.tick_params(labelsize=15)
    ax.tick_params(labelsize=15)
    ax.set_xlabel("$d_1$", fontsize=15)
    ax.set_ylabel("$d_2$", fontsize=15)
    ax.set_aspect(1)



def plot(q, filename):
    """
    Plot the spin positions and the interaction matrix of an spQUBO.

    Parameters
    ----------
    q : qmodel
        The spatial quadratic function of the problem to be plotted.
    filename : str
        The file path to save the plot.
    """
    tx, ty = -0.15, 1.1
    fig, axes = plt.subplots(3, 1, figsize=(12, 4))
    axc = 0

    #### ####

    ax = axes[axc]
    _plot_c(q, fig, ax)
    # add title on top-left on the subplot
    ax.text(tx, ty,
            'c', fontsize=18,
            transform=ax.transAxes, va='bottom', ha='left', weight="bold")
    axc += 1

    #### ####

    ax = axes[axc]
    _plot_d(q, fig, ax)
    # add title on top-left on the subplot
    ax.text(tx, ty,
            'd', fontsize=18,
            transform=ax.transAxes, va='bottom', ha='left', weight="bold")
    axc += 1

    #### ####

    ax = axes[axc]
    _plot_e(q, fig, ax)
    # add title on top-left on the subplot
    ax.text(tx, ty,
            'e', fontsize=18,
            transform=ax.transAxes, va='bottom', ha='left', weight="bold")
    axc += 1

    for ax in axes:
        ax.yaxis.set_label_coords(-0.06, 0.5)

    #### ####

    # plt.tight_layout()
    savefig(filename, fig)
