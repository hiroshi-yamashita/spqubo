from blob_plot import _plot_problem_a, _plot_answer_2_scatter
from blob2_plot import _plot_c, _plot_d, _plot_e
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


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


def composed_plot(kwargs_1, kwargs_2, filename):
    fig = plt.figure(figsize=(13, 9))


    gs = gridspec.GridSpec(2, 2, height_ratios=[4, 3],
                           left=0.1, right=0.85, top=0.95, bottom=0.05, 
                           hspace=0.2, wspace=0.6)
    ax1 = fig.add_subplot(gs[0, 0])  
    ax2 = fig.add_subplot(gs[0, 1])  
    gs_ax3to5 = gs[1, :]  
    inner_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_ax3to5, height_ratios=[1, 1, 1],
                                                
                                                hspace=0.2, wspace=0.2)
    ax3 = fig.add_subplot(inner_gs[0, 0])  
    ax4 = fig.add_subplot(inner_gs[1, 0])  
    ax5 = fig.add_subplot(inner_gs[2, 0])  

    axes = [ax1, ax2, ax3, ax4, ax5]

    axc = 0
    tx, ty = -0.20, 1.05

    df_prob, B, ans, K, Nk = kwargs_1['df_prob'], kwargs_1['B'], kwargs_1['ans'], kwargs_1['K'], kwargs_1['Nk']

    ax = axes[axc]
    _plot_problem_a(df_prob, B, fig, ax1)
    ax.text(tx, ty,
            "a", fontsize=18,
            transform=ax.transAxes, va='bottom', ha='right', weight="bold")
    axc += 1

    ax = axes[axc]
    _plot_answer_2_scatter(df_prob, B, ans, K, Nk, fig, ax)
    ax.text(tx, ty,
            "b", fontsize=18,
            transform=ax.transAxes, va='bottom', ha='right', weight="bold")
    axc += 1

    q = kwargs_2['q']
    tx, ty = -0.1, 1.05

    ax = axes[axc]
    _plot_c(q, fig, ax)
    ax.text(tx, ty,
            'c', fontsize=18,
            transform=ax.transAxes, va='bottom', ha='left', weight="bold")
    axc += 1

    ax = axes[axc]
    _plot_d(q, fig, ax)
    ax.text(tx, ty,
            'd', fontsize=18,
            transform=ax.transAxes, va='bottom', ha='left', weight="bold")
    axc += 1

    ax = axes[axc]
    _plot_e(q, fig, ax)
    ax.text(tx, ty,
            'e', fontsize=18,
            transform=ax.transAxes, va='bottom', ha='left', weight="bold")
    axc += 1

    for ax in [ax3, ax4, ax5]:
        ax.yaxis.set_label_coords(-0.06, 0.5)

    savefig(filename, fig)
