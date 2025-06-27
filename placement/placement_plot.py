import numpy as np
nax = np.newaxis
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


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

def plot_problem(q, filename):
    """
    Plot the spin positions and interaction matrix of an spQUBO problem.

    Parameters
    ----------
    q : qmodel
        The spatial  quadratic function of the problem to be plotted.
    filename : str
        The file path to save the plot.
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axc = 0

    #### ####

    ax = axes[axc]

    q_arr = q.J
    x_axis = np.fft.fftfreq(q_arr.shape[0], d=1.0/q_arr.shape[0])
    y_axis = np.fft.fftfreq(q_arr.shape[1], d=1.0/q_arr.shape[1])    
    q_arr = np.fft.fftshift(q_arr)
    x_axis = np.fft.fftshift(x_axis)
    y_axis = np.fft.fftshift(y_axis)

    handle = ax.pcolormesh(x_axis, y_axis, q_arr, vmin=-0.3, vmax=0.3)
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="7%", pad="5%")
    cb = fig.colorbar(handle, cax=cax)
    cax.tick_params(labelsize=15)
    ax.tick_params(labelsize=15)
    ax.set_aspect(1)
    ax.set_xlabel("$d_1$", fontsize=15)
    ax.set_ylabel("$d_2$", fontsize=15)

    tx, ty = -0.20, 1.1
    ax.text(tx, ty,
            "a", fontsize=18,
            transform=ax.transAxes, va='bottom', ha='right', weight="bold")

    axc += 1

    #### ####

    ax = axes[axc]
    q_arr = q.get_f_J()
    x_axis = np.fft.fftfreq(q_arr.shape[0], d=1.0/q_arr.shape[0])
    y_axis = np.fft.fftfreq(q_arr.shape[1], d=1.0/q_arr.shape[1])    
    q_arr = np.fft.fftshift(q_arr)
    x_axis = np.fft.fftshift(x_axis)
    y_axis = np.fft.fftshift(y_axis)

    handle = ax.pcolormesh(x_axis, y_axis, q_arr, vmin=-100, vmax=100)

    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="7%", pad="5%")
    cb = fig.colorbar(handle, cax=cax)
    cax.tick_params(labelsize=15)
    ax.tick_params(labelsize=15)
    ax.set_aspect(1)
    ax.set_xlabel("$d_1$", fontsize=15)
    ax.set_ylabel("$d_2$", fontsize=15)

    tx, ty = -0.20, 1.1
    ax.text(tx, ty,
            "b", fontsize=18,
            transform=ax.transAxes, va='bottom', ha='right', weight="bold")

    axc += 1

    #### ####

    fig.tight_layout()
    fig.subplots_adjust(left=0.10, right=0.93)
    savefig(filename, fig)


def plot_answer(spins, settings, filename):
    """
    Plot the solution of a placement problem as a color mesh plot.

    Parameters
    ----------
    spins : np.ndarray
        The spin array containing the results of the QUBO optimization.
    settings : Tuple[int, int, float, float, np.ndarray]
        The settings of the placement problem, including:
            - Wy: Number of rows in the positions grid.
            - Wx: Number of columns in the positions grid.
            - skip_y: Step size in the y direction.
            - skip_x: Step size in the x direction.
            - placement_cost: Cost of placement at each position.
    filename : str
        The file path to save the plot.
    """
    Wy, Wx, skip_y, skip_x, placement_cost = settings

    x = np.zeros((Wy, Wx))
    y = np.zeros((Wy, Wx))
    x[:, :] = (np.arange(Wx) * skip_x)[nax, :]
    y[:, :] = (np.arange(Wy) * skip_y)[:, nax]

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    #### ####

    vmin, vmax = np.min(placement_cost), np.max(placement_cost)
    vrange = vmax - vmin
    expand = 0.3
    vmax = vmax + vrange * expand
    vmin = vmin - vrange * expand

    x_ = x.ravel()[spins > 0]
    y_ = y.ravel()[spins > 0]

    handle = ax.pcolormesh(x[0, :], y[:, 0],
                             placement_cost,
                             shading="nearest",
                             vmin=vmin,
                             vmax=vmax
                             )
    handle.set_cmap("viridis")

    ax.scatter(x_, y_, color="white", marker="x", s=24)

    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="7%", pad="5%")
    cb = fig.colorbar(handle, cax=cax)
    cax.tick_params(labelsize=15)
    ax.tick_params(labelsize=15)
    ax.set_aspect(1)
    ax.set_xlabel("$p_1$", fontsize=15)
    ax.set_ylabel("$p_2$", fontsize=15)

    tx, ty = -0.15, 1.1
    ax.text(tx, ty,
            "c", fontsize=18,
            transform=ax.transAxes, va='bottom', ha='right', weight="bold")


    #### ####

    fig.tight_layout()
    fig.subplots_adjust(left=0.16, right=0.87)
    savefig(filename, fig)
    plt.close(fig)


def plot_runtime(ret, filename):
    """
    Plot the runtime of the solver for different problem sizes and modes.

    Parameters
    ----------
    ret : pd.DataFrame
        Dataframe containing runtime data with columns:
            - size: Problem size.
            - time: Time taken to solve the problem.
            - mode: Solver mode (e.g., "fourier", "naive").
    filename : str
        The file path to save the plot.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    #### ####

    label = {
        "fourier": "Fourier",
        "naive": "direct"
    }
    for g, df_g in ret.groupby("mode"):
        ax.scatter(df_g["size"], df_g["time"], label=label[g])
        df_mean = df_g.groupby("size").mean()
        df_mean.sort_index(inplace=True)
        ax.plot(df_mean.index, df_mean["time"])
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("size", fontsize=15)
    ax.set_ylabel("time [s]", fontsize=15)
    ax.tick_params(labelsize=15)
    ax.legend(fontsize=15)

    #### ####

    fig.tight_layout()
    savefig(filename, fig)
    plt.close()
