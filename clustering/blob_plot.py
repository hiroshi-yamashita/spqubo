import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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


def get_color_seq(c):
    if np.isscalar(c):
        c = np.arange(c)
    _min = np.min(c)
    _max = np.max(c)
    if _min != _max:
        colors = [(_c - _min) / (_max - _min) for _c in c]
        ret = cm.nipy_spectral([0.05 + 0.9 * (_c) for _c in colors])
    else:
        ret = ["black"] * len(c)

    return ret


def plot_problem(df, L, filename, subcaption=None):
    """
    Plot a scatter plot for a clustering problem.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the points to be plotted.
    L : int
        The size of the grid where the points are located.
    filename : str
        The file path to save the plot.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

    #### ####

    if L > 20:
        s, pad = 18, 5
    else:
        s, pad = 72, 1

    for g, df_g in df.groupby("cls"):
        ax.scatter(df_g[0],
                   df_g[1],
                   label=g, marker="ox+ox+o"[g],
                   s=s)

    ax.set_xlim([-pad, L + pad - 1])
    ax.set_ylim([-pad, L + pad - 1])
    ax.legend([f"C{k}" for k in range(7)], fontsize=15,
              loc='upper left', bbox_to_anchor=(1, 1))
    ax.tick_params(labelsize=15)
    ax.set_aspect(1)
    ax.set_xlabel("$d_1$", fontsize=15)
    ax.set_ylabel("$d_2$", fontsize=15)

    if subcaption:
        tx, ty = -0.20, 1.1
        ax.text(tx, ty,
                subcaption, fontsize=18,
                transform=ax.transAxes, va='bottom', ha='right')

    #### ####

    fig.tight_layout()
    fig.subplots_adjust(left=0, right=0.85)

    savefig(filename, fig)
    plt.close()


def plot_answer(ans, K, Nk, filename):
    """
    Plot the clustering result as a color mesh.

    Parameters
    ----------
    ans : np.ndarray
        The answer array containing the results of the QUBO optimization.
    K : int
        The number of clusters.
    Nk : int
        The number of points in each cluster, which is 
        used to compute the number of points in the grid.
    filename : str
        The file path to save the plot.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

    #### ####

    ax.pcolormesh(ans.reshape(K, K*Nk))

    ax.set_xlabel("$d_1$", fontsize=15)
    ax.set_ylabel("$d_2$", fontsize=15)

    #### ####

    fig.tight_layout()
    fig.subplots_adjust(left=0, right=0.85)

    savefig(filename, fig)
    plt.close()


def plot_answer_2_scatter(df_prob, L, ans, K, Nk, filename, subcaption=None):
    """
    Plot the answer of a clustering problem in a scatter plot.

    Parameters
    ----------
    df_prob : pd.DataFrame
        The dataframe containing the points to be plotted.
    L : int
        The size of the grid where the points are located.
    ans : np.ndarray
        The answer array containing the clustering results.
    K : int
        The number of clusters.
    Nk : int
        The number of points in each cluster, which is
        used to compute the number of points in the grid.
    filename : str
        The file path to save the plot.
    """

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

    #### ####

    df_prob = df_prob.copy()

    ans_mat = ans.reshape(K, K*Nk)
    satisfying_exact_one = (np.sum(ans_mat, axis=0) == 1)

    cls = np.where(satisfying_exact_one,
                   np.sum(np.arange(K)[:, np.newaxis] * ans_mat, axis=0),
                   K
                   ).astype("i")

    df_prob["cls"] = cls

    if L > 20:
        s, pad = 18, 5
    else:
        s, pad = 72, 1

    colors = list(get_color_seq(7)[np.arange(7)]) + ["black"]

    for g, df_g in df_prob.groupby("cls"):
        ax.scatter(df_g[0],
                   df_g[1],
                   label=g,
                   marker="ooooxxxv"[g],
                   s=s,
                   color=colors[g])

    ax.set_xlim([-pad, L + pad - 1])
    ax.set_ylim([-pad, L + pad - 1])
    ax.legend([f"C{k}" for k in range(7)] + ["Invalid"], fontsize=15,
              loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_xticks(np.arange(0, 51, 10))
    ax.set_yticks(np.arange(0, 51, 10))
    ax.tick_params(labelsize=15)
    ax.set_xlabel("$d_1$", fontsize=15)
    ax.set_ylabel("$d_2$", fontsize=15)
    ax.set_aspect(1)

    if subcaption:
        tx, ty = -0.20, 1.1
        ax.text(tx, ty,
                subcaption, fontsize=18,
                transform=ax.transAxes, va='bottom', ha='right')

    #### ####

    fig.tight_layout()
    fig.subplots_adjust(left=0, right=0.85)

    savefig(filename, fig)
    plt.close()
