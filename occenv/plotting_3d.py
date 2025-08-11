# env/plotting_3d.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm

from occenv.utils_bivariate import BivariateGrid, Gaussian2D

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def create_heatmap(x, y, z, xlabel, ylabel, title, cmap, vmin, vmax, outpath=False):
    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.pcolormesh(x, y, z, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(c, ax=ax)
    if outpath:
        plt.savefig(outpath)
    plt.show()


def plot_option_a(U, V, Z, outpath, title):
    """
    Heatmap of P(U=u, V=v) with log colour scale (zeros shown as darkest blue),
    plus Gaussian mean marker and confidence ellipses derived from the grid.
    """
    Z = np.asarray(Z, float)
    pos = Z[Z > 0]
    if pos.size == 0:
        raise ValueError("All probabilities are zero; nothing to plot on log scale.")
    vmin, vmax = pos.min(), pos.max()

    Z_plot = np.ma.masked_where(Z <= 0, Z)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(cmap(0.0))  # zeros -> darkest blue

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    im = ax.imshow(
        Z_plot,
        origin="lower",
        aspect="auto",
        extent=[U[0], U[-1], V[0], V[-1]],
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap=cmap,
    )

    # --- Use classes: fit Gaussian from the discrete distribution and overlay ---
    grid = BivariateGrid(U, V, Z)
    mu, Sigma, evals, evecs = grid.gaussian_from_grid(return_eigendecomp=True)
    gauss = Gaussian2D(mu, Sigma)

    # Mean
    ax.plot(mu[0], mu[1], marker="+", ms=12, mew=2, color="white")

    # Confidence ellipses
    gauss.ellipse(ax, p=0.68, edgecolor="white", linewidth=1.8)
    gauss.ellipse(ax, p=0.95, edgecolor="white", linestyle="--", linewidth=1.4)
    gauss.ellipse(ax, p=0.997, edgecolor="white", linestyle=":", linewidth=1.2)

    # Principal axes (1-sigma length)
    v1 = evecs[:, 0]
    v2 = evecs[:, 1]
    scale1 = np.sqrt(evals[0])
    scale2 = np.sqrt(evals[1])
    ax.plot(
        [mu[0] - scale1 * v1[0], mu[0] + scale1 * v1[0]],
        [mu[1] - scale1 * v1[1], mu[1] + scale1 * v1[1]],
        color="white",
        alpha=0.6,
        linewidth=1.2,
    )
    ax.plot(
        [mu[0] - scale2 * v2[0], mu[0] + scale2 * v2[0]],
        [mu[1] - scale2 * v2[0], mu[1] + scale2 * v2[0]],
        color="white",
        alpha=0.6,
        linewidth=1.2,
    )

    # (Optional) diagnostics
    print("mu =", mu)
    print("Sigma =\n", Sigma)
    print("Eigenvalues =", evals)
    print("Eigenvectors (columns) =\n", evecs)

    cbar = fig.colorbar(im, ax=ax, label="Probability  P(U=u, V=v)")
    ax.set_xlabel("Union size  u")
    ax.set_ylabel("Intersection size  v")
    ax.set_title(title + " — Option A (LogNorm, zeros as dark blue)")
    fig.savefig(outpath, dpi=200)
    plt.show()


def plot_surface_3d(
    U,
    V,
    Z,
    title="Bivariate distribution — 3D surface",
    log_colors=True,
    log_height=False,
):
    """
    3D surface of Z on (U,V). By default: linear heights, log-coloured faces.
    Set log_height=True to raise log10(Z) (with epsilon) instead.
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

    # Heights
    if log_height:
        Zheight = np.log10(np.maximum(Z, eps))
        zlabel = r"$\log_{10}\,P(U=u,V=v)$"
    else:
        Zheight = Z
        zlabel = r"$P(U=u,V=v)$"

    # Colours for faces (log or linear)
    cmap = plt.get_cmap("viridis")
    if log_colors:
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = mpl.colors.Normalize(vmin=Z.min(), vmax=Z.max())
    facecolors = cmap(norm(np.maximum(Z, eps)))

    # Plot
    fig = plt.figure(figsize=(9, 6), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        UU,
        VV,
        Zheight,
        facecolors=facecolors,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
    )

    # Colourbar keyed to probabilities, not heights
    m = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    m.set_array([])
    fig.colorbar(m, ax=ax, shrink=0.7, pad=0.05, label=r"$P(U=u,V=v)$")

    ax.set_xlabel("Union size  u")
    ax.set_ylabel("Intersection size  v")
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    ax.view_init(elev=35, azim=-135)  # nice default view

    return fig, ax


def add_floor_contours(ax, U, V, Z, levels=8, offset=0.0, **contour_kw):
    U = np.asarray(U)
    V = np.asarray(V)
    Z = np.asarray(Z, float)
    UU, VV = np.meshgrid(U, V)
    return ax.contour(UU, VV, Z, levels=levels, offset=offset, zdir="z", **contour_kw)


def plot_surface_plotly(U, V, Z, title="Bivariate distribution — 3D (interactive)"):
    if not HAS_PLOTLY:
        print("Plotly not available. Install with: pip install plotly")
        return

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
