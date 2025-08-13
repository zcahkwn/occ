import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.graph_objects as go
from matplotlib.colors import LogNorm
from occenv.utils_bivariate import Gaussian2D


def plot_heatmap(x, y, z, xlabel, ylabel, title, cmap, vmin, vmax, outpath=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    c = ax.pcolormesh(x, y, z, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(c, ax=ax)
    if outpath:
        plt.savefig(outpath)
    plt.show()


def plot_heatmap_ellipse(U, V, Z, mu, Sigma, outpath=None, title=None):
    """
    Heatmap of P(U=u, V=v) with log colour scale (zeros shown as darkest blue),
    plus Gaussian mean marker and confidence ellipses derived from the grid.
    """
    Z = np.asarray(Z, float)
    pos = Z[Z > 0]
    if pos.size == 0:
        raise ValueError("All probabilities are zero; nothing to plot on log scale.")
    vmin, vmax = pos.min(), pos.max()

    # Set up the colormap (zeros are marked as darkest blue)
    Z_plot = np.ma.masked_where(Z <= 0, Z)  # Mask out zeros
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(cmap(0.0))
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    im = ax.imshow(
        Z_plot,
        origin="lower",
        aspect="auto",
        extent=[U[0], U[-1], V[0], V[-1]],
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap=cmap,
    )
    ax.plot(*mu, marker="+", ms=12, mew=2, color="white")  # mean marker

    # Confidence ellipses
    gauss = Gaussian2D(mu, Sigma)
    for p, ls, lw in [(0.68, "-", 1.8), (0.95, "--", 1.4), (0.997, ":", 1.2)]:
        gauss.ellipse(ax, p=p, edgecolor="white", linestyle=ls, linewidth=lw)

    fig.colorbar(im, ax=ax, label="Probability  P(U=u, V=v)")
    ax.set(xlabel="Union size  u", ylabel="Intersection size  v", title=title)
    if outpath:
        fig.savefig(outpath, dpi=200)
    plt.show()


def plot_surface_3d(U, V, Z, title, log_colors=True):
    """
    3D surface of Z on (U,V).
    """
    U = np.asarray(U)
    V = np.asarray(V)
    Z = np.asarray(Z, float)
    UU, VV = np.meshgrid(U, V)

    pos = Z[Z > 0]
    if pos.size == 0:
        raise ValueError("All probabilities are zero; nothing to plot.")
    vmin, vmax = pos.min(), pos.max()

    eps = vmin * 0.5
    cmap = plt.get_cmap("viridis")
    if log_colors:
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        facecolors = cmap(norm(np.maximum(Z, eps)))
    else:
        norm = mpl.colors.Normalize(vmin=Z.min(), vmax=Z.max())
        facecolors = cmap(norm(Z))

    fig = plt.figure(figsize=(9, 6), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        UU,
        VV,
        Z,
        facecolors=facecolors,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
    )

    m = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    m.set_array([])
    fig.colorbar(m, ax=ax, shrink=0.7, pad=0.05, label=r"$P(U=u,V=v)$")

    ax.set(
        xlabel="Union size  u",
        ylabel="Intersection size  v",
        zlabel=r"$P(U=u,V=v)$",
        title=title,
    )
    ax.view_init(elev=35, azim=-135)
    plt.show()


def plot_surface_plotly(U, V, Z, title):
    U = np.asarray(U)
    V = np.asarray(V)
    Z = np.asarray(Z, float)
    UU, VV = np.meshgrid(U, V)

    fig = go.Figure(
        data=[
            go.Surface(
                x=UU,
                y=VV,
                z=Z,
                colorscale="Viridis",
                colorbar=dict(title="P(U=u,V=v)"),
                hovertemplate="u=%{x}<br>v=%{y}<br>P=%{z:.3e}<extra></extra>",
            )
        ]
    )
    fig.update_scenes(
        xaxis_title="Union size u",
        yaxis_title="Intersection size v",
        zaxis_title="P(U=u,V=v)",
    )
    fig.update_layout(title=title, width=800, height=600)
    fig.show()
