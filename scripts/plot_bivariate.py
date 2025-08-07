import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm, SymLogNorm
from math import comb, prod
from pathlib import Path

from occenv.analytical import AnalyticalResult
from occenv.constants import FIGURE_DIR

from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


import numpy as np


def gaussian_from_grid(U, V, Z):
    """
    Given 1D grids U, V and a probability table Z (shape |V| x |U|),
    return mean vector mu, covariance matrix Sigma, eigenvalues/vecs.
    """
    Z = np.asarray(Z, dtype=float)
    p = Z / Z.sum()  # normalize to a distribution

    UU, VV = np.meshgrid(U, V)  # same shape as Z
    mu_u = (p * UU).sum()
    mu_v = (p * VV).sum()
    mu = np.array([mu_u, mu_v])

    du = UU - mu_u
    dv = VV - mu_v
    cov_uu = (p * du * du).sum()
    cov_vv = (p * dv * dv).sum()
    cov_uv = (p * du * dv).sum()
    Sigma = np.array([[cov_uu, cov_uv], [cov_uv, cov_vv]])

    # Eigendecomposition (principal axes). eigh -> symmetric matrix, sorted ascending
    evals, evecs = np.linalg.eigh(Sigma)
    order = evals.argsort()[::-1]  # largest first
    evals = evals[order]
    evecs = evecs[:, order]

    return mu, Sigma, evals, evecs


def gaussian_from_grid_new(U, V, Z):
    U = np.asarray(U)
    V = np.asarray(V)
    Z = np.asarray(Z, float)
    p = Z / Z.sum()
    UU, VV = np.meshgrid(U, V)  # shape = Z.shape

    mu_u = (p * UU).sum()
    mu_v = (p * VV).sum()
    mu = np.array([mu_u, mu_v])

    du = UU - mu_u
    dv = VV - mu_v
    cov_uu = (p * du * du).sum()
    cov_vv = (p * dv * dv).sum()
    cov_uv = (p * du * dv).sum()
    Sigma = np.array([[cov_uu, cov_uv], [cov_uv, cov_vv]])
    return mu, Sigma


def draw_cov_ellipse(ax, mu, Sigma, p=0.95, **kw):
    """
    Add a confidence ellipse for a bivariate normal N(mu, Sigma)
    that contains probability mass p (e.g. p=0.68, 0.95, 0.997).
    """
    try:
        from scipy.stats import chi2

        c = chi2.ppf(p, df=2)
    except Exception:
        # Fallback for common levels if SciPy isn't available
        lookup = {
            0.50: 1.386,
            0.68: 2.279,
            0.90: 4.605,
            0.95: 5.991,
            0.99: 9.210,
            0.997: 11.829,
        }
        c = lookup.get(p, 5.991)  # default ~95%

    # Eigen decomposition
    evals, evecs = np.linalg.eigh(Sigma)
    order = evals.argsort()[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    # Ellipse params: semi-axes lengths = sqrt(c * eigenvalues)
    width, height = 2 * np.sqrt(c * evals)  # diameters (2a, 2b)
    angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))  # angle of largest axis

    e = Ellipse(xy=mu, width=width, height=height, angle=angle, fill=False, **kw)
    ax.add_patch(e)
    return e


def add_mahalanobis_contours(ax, mu, Sigma, xlim, ylim, ps=(0.68, 0.95, 0.997), **kw):
    try:
        from scipy.stats import chi2

        levels = [chi2.ppf(p, df=2) for p in ps]
    except Exception:
        fallback = {0.68: 2.279, 0.95: 5.991, 0.997: 11.829}
        levels = [fallback[p] for p in ps]

    xx = np.linspace(xlim[0], xlim[1], 400)
    yy = np.linspace(ylim[0], ylim[1], 400)
    XX, YY = np.meshgrid(xx, yy)

    Sinv = np.linalg.inv(Sigma + 1e-12 * np.eye(2))  # small jitter for safety
    dx = XX - mu[0]
    dy = YY - mu[1]
    M2 = Sinv[0, 0] * dx * dx + 2 * Sinv[0, 1] * dx * dy + Sinv[1, 1] * dy * dy

    cs = ax.contour(XX, YY, M2, levels=levels, **kw)
    # Optional: label with probabilities
    fmt = {lvl: f"{int(100*p)}%" for lvl, p in zip(cs.levels, ps)}
    ax.clabel(cs, inline=1, fontsize=9, fmt=fmt)
    return cs


# -----------------------
# Inputs
# -----------------------
N = 100
shard_sizes = [50, 40, 60]


# -----------------------
# Helpers
# -----------------------
def compute_grid(ar: AnalyticalResult, N: int, sizes: list[int]):
    """Return (U, V, Z, sum_probs, exact_check). Z contains probabilities; infeasible cells are 0."""
    sizes = list(sizes)
    m, S = len(sizes), sum(sizes)
    nmax, nmin = max(sizes), min(sizes)

    # Feasible 1D ranges
    U = np.arange(nmax, min(N, S) + 1)  # union
    V = np.arange(0, nmin + 1)  # intersection

    # Exact big-int denominator once
    den = prod(comb(N, n) for n in sizes)

    Z = np.zeros((len(V), len(U)), dtype=float)
    total_cases = 0

    # Fill only feasible (u,v) pairs; others stay 0
    for iv, v in enumerate(V):
        # ceil((S - v) / (m - 1)) using integer arithmetic
        u_lo = max(nmax, (S - v + (m - 2)) // (m - 1))
        u_hi = min(N, S - (m - 1) * v)
        if u_lo > u_hi:
            continue
        for u in range(u_lo, u_hi + 1):
            cases = ar.bivariate_cases(u, v)
            total_cases += cases
            if cases:
                Z[iv, u - U[0]] = cases / den

    return U, V, Z, float(Z.sum()), (total_cases == den)


def plot_option_a(U, V, Z, outpath, title):
    """
    LogNorm for positive probabilities; zeros are masked and colored as the
    darkest blue (cmap[0.0]), i.e., visually like 10^-inf.
    """
    pos = Z[Z > 0]
    if pos.size == 0:
        raise ValueError("All probabilities are zero; nothing to plot on log scale.")
    vmin, vmax = pos.min(), pos.max()

    Z_plot = np.ma.masked_where(Z <= 0, Z)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(cmap(0.0))  # zeros -> darkest blue instead of white

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    im = ax.imshow(
        Z_plot,
        origin="lower",
        aspect="auto",
        extent=[U[0], U[-1], V[0], V[-1]],
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap=cmap,
    )

    # --- fit Gaussian from the discrete distribution and overlay ---
    mu, Sigma, evals, evecs = gaussian_from_grid(U, V, Z)

    # Mark the mean
    ax.plot(mu[0], mu[1], marker="+", ms=12, mew=2, color="white")

    # Option A: confidence ellipses
    draw_cov_ellipse(ax, mu, Sigma, p=0.68, edgecolor="white", linewidth=1.8)
    draw_cov_ellipse(
        ax, mu, Sigma, p=0.95, edgecolor="white", linestyle="--", linewidth=1.4
    )
    draw_cov_ellipse(
        ax, mu, Sigma, p=0.997, edgecolor="white", linestyle=":", linewidth=1.2
    )

    # Option B (alternative): Mahalanobis distance contours
    # add_mahalanobis_contours(ax, mu, Sigma, xlim=(U[0], U[-1]), ylim=(V[0], V[-1]),
    #                          ps=(0.68, 0.95, 0.997), colors="white", linewidths=1.0)

    # If you want to see the principal axes themselves:
    v1 = evecs[:, 0]  # axis for the largest eigenvalue
    v2 = evecs[:, 1]
    scale1 = np.sqrt(evals[0])  # length ~ 1-sigma along axis
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

    # (Optional) Print the eigendecomposition
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
    UU, VV = np.meshgrid(U, V)  # same shape as Z
    Z = np.asarray(Z, dtype=float)

    # Mask/epsilon for zeros so log colouring won't explode
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
    facecolors = cmap(norm(np.maximum(Z, eps)))  # avoid nan in colours

    # Plot
    fig = plt.figure(figsize=(9, 6), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
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


def gaussian_on_grid(U, V, mu, Sigma, restrict_to_support=True, integrate_cells=False):
    """
    Return Q[v,u] ~ discretized Gaussian over the same feasible cells as Z.
    - restrict_to_support: set Q=0 outside Z's support and renormalize.
    - integrate_cells=False: use pdf(u,v) as a proxy for cell mass and renormalize.
      (If you have SciPy, you could integrate each unit cell for higher fidelity.)
    """
    U = np.asarray(U)
    V = np.asarray(V)
    UU, VV = np.meshgrid(U, V)
    X = np.stack([UU, VV], axis=-1)  # (*,2)

    Sinv = np.linalg.inv(Sigma)
    dX = X - mu
    # pdf up to the normalizing constant
    expo = np.einsum("...i,ij,...j", dX, Sinv, dX)  # Mahalanobis^2
    det = np.linalg.det(Sigma)
    const = 1.0 / (2 * np.pi * np.sqrt(det))
    Q = const * np.exp(-0.5 * expo)

    if restrict_to_support:
        # If your Z has infeasible cells set to exactly 0, match that support
        mask = np.isfinite(Q) & (Q > 0)
        # keep all (feasible) cells; outside support we set to zero
        mask &= ~np.isclose(Q, 0.0)
        # You may also apply the same feasibility rule you used to build Z.
        # Here we simply mirror the zero pattern in Z if you pass it in.
    Q = Q / Q.sum()
    return Q


def compare_distributions(P, Q, eps=1e-300):
    P = P / P.sum()
    Q = Q / Q.sum()
    # avoid log(0) by clipping Q (P zeros are harmless)
    Qc = np.clip(Q, eps, None)
    # Total variation
    TV = 0.5 * np.abs(P - Q).sum()
    # KL divergences
    maskP = P > 0
    KL_PQ = np.sum(P[maskP] * (np.log(P[maskP]) - np.log(Qc[maskP])))
    maskQ = Q > 0
    KL_QP = np.sum(Q[maskQ] * (np.log(Q[maskQ]) - np.log(np.clip(P[maskQ], eps, None))))
    # Jensen–Shannon
    M = 0.5 * (P + Q)
    JSD = 0.5 * (
        np.sum(P[maskP] * (np.log(P[maskP]) - np.log(np.clip(M[maskP], eps, None))))
        + np.sum(Q[maskQ] * (np.log(Q[maskQ]) - np.log(np.clip(M[maskQ], eps, None))))
    )
    # Chi-square distance (Pearson)
    CHI2 = np.sum((P - Q) ** 2 / np.clip(Q, eps, None))
    return dict(TV=TV, KL_PQ=KL_PQ, KL_QP=KL_QP, JSD=JSD, CHI2=CHI2)


def mahalanobis_ks(U, V, Z, mu, Sigma):
    U = np.asarray(U)
    V = np.asarray(V)
    Z = np.asarray(Z, float)
    P = Z / Z.sum()
    UU, VV = np.meshgrid(U, V)
    X = np.stack([UU, VV], axis=-1).reshape(-1, 2)
    w = P.reshape(-1)

    dX = X - mu
    Sinv = np.linalg.inv(Sigma)
    M2 = np.einsum("ni,ij,nj->n", dX, Sinv, dX)  # length Ncells

    # Ensure M2 is always an array (safety measure)
    M2 = np.atleast_1d(M2)

    # Weighted CDF of M2
    order = np.argsort(M2)
    m2_sorted = M2[order]
    w_sorted = w[order]
    cdf_hat = np.cumsum(w_sorted)

    # Theoretical CDF for chi^2_2: F(t) = 1 - exp(-t/2)
    F = 1.0 - np.exp(-0.5 * m2_sorted)

    KS = np.max(np.abs(cdf_hat - F))
    return KS, m2_sorted, cdf_hat, F


def mardia_moments(U, V, Z, mu, Sigma):
    U = np.asarray(U)
    V = np.asarray(V)
    Z = np.asarray(Z, float)
    P = Z / Z.sum()
    UU, VV = np.meshgrid(U, V)
    X = np.stack([UU, VV], axis=-1).reshape(-1, 2)
    w = P.reshape(-1)
    dX = X - mu
    Sinv = np.linalg.inv(Sigma)
    M2 = np.einsum("ni,ij,nj->n", dX, Sinv, dX)
    E_M2 = np.sum(w * M2)
    E_M4 = np.sum(w * M2**2)  # this is Mardia's kurtosis β2,p in population terms
    return E_M2, E_M4


def angle_uniformity(U, V, Z, mu, Sigma):
    U = np.asarray(U)
    V = np.asarray(V)
    Z = np.asarray(Z, float)
    P = Z / Z.sum()
    UU, VV = np.meshgrid(U, V)
    X = np.stack([UU, VV], axis=-1).reshape(-1, 2)
    w = P.reshape(-1)

    # Whiten: z = L^{-1}(x - mu) with LL^T = Sigma
    L = np.linalg.cholesky(Sigma)
    Zw = np.linalg.solve(L, (X - mu).T).T  # (Ncells, 2)

    theta = np.arctan2(Zw[:, 1], Zw[:, 0])
    c = np.sum(w * np.cos(theta))
    s = np.sum(w * np.sin(theta))
    R = np.sqrt(c * c + s * s)  # 0 is ideal
    return R


def conditional_linearity(U, V, Z):
    U = np.asarray(U)
    V = np.asarray(V)
    Z = np.asarray(Z, float)
    P = Z / Z.sum()
    pV = P.sum(axis=1)  # over U (columns) -> shape |V|
    EV = (pV * V).sum()

    # E[U|V=v] and Var[U|V=v]
    EU_cond = np.zeros_like(V, dtype=float)
    VarU_cond = np.zeros_like(V, dtype=float)
    for i, v in enumerate(V):
        if pV[i] > 0:
            pU_given_v = P[i, :] / pV[i]
            mu_u_v = (pU_given_v * U).sum()
            EU_cond[i] = mu_u_v
            VarU_cond[i] = (pU_given_v * (U - mu_u_v) ** 2).sum()

    # Weighted linear regression of E[U|V] on v with weights p(V=v)
    w = pV
    X = np.vstack([np.ones_like(V, float), V]).T
    W = np.diag(w)
    beta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ EU_cond)
    intercept, slope = beta

    # Weighted R^2
    y = EU_cond
    yhat = X @ beta
    ybar = (w * y).sum() / w.sum()
    SS_res = (w * (y - yhat) ** 2).sum()
    SS_tot = (w * (y - ybar) ** 2).sum()
    R2 = 1.0 - SS_res / SS_tot if SS_tot > 0 else 1.0

    # Constancy of Var(U|V=v): coefficient of variation across v
    mean_var = np.average(VarU_cond, weights=w)
    std_var = np.sqrt(np.average((VarU_cond - mean_var) ** 2, weights=w))
    cv_var = std_var / mean_var if mean_var > 0 else 0.0

    return dict(
        intercept=intercept,
        slope=slope,
        R2=R2,
        mean_cond_var=mean_var,
        cv_cond_var=cv_var,
        EU_cond=EU_cond,
        VarU_cond=VarU_cond,
        weights=w,
    )


def mardia_skewness(U, V, Z, mu, Sigma):
    # Weighted population version: β1,p = E[(z_i^T z_j)^3], i,j iid
    U = np.asarray(U)
    V = np.asarray(V)
    Z = np.asarray(Z, float)
    P = Z / Z.sum()
    UU, VV = np.meshgrid(U, V)
    X = np.stack([UU, VV], axis=-1).reshape(-1, 2)
    w = P.reshape(-1)

    L = np.linalg.cholesky(Sigma)
    Zw = np.linalg.solve(L, (X - mu).T).T  # (Ncells, 2)

    # pairwise inner products (z_i · z_j)
    G = Zw @ Zw.T
    W = np.outer(w, w)
    beta1p = np.sum(W * (G**3))
    return beta1p


def plot_bars_3d(
    U, V, Z, stride_u=1, stride_v=1, title="Bivariate distribution — 3D bars"
):
    """
    3D bar chart. Uses index coordinates for bar positions with ticks mapped to U,V.
    Use stride_* to thin the grid if there are many bins.
    """
    Z = np.asarray(Z, dtype=float)
    mV, mU = Z.shape
    uu_idx, vv_idx = np.meshgrid(np.arange(mU), np.arange(mV))
    x = uu_idx.ravel()[::stride_u]
    y = vv_idx.ravel()[::stride_v]
    dz = Z.ravel()[:: (stride_u * stride_v)]

    keep = dz > 0
    x, y, dz = x[keep], y[keep], dz[keep]
    z0 = np.zeros_like(dz)

    dx = dy = 0.9  # bar footprint

    fig = plt.figure(figsize=(10, 6), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.bar3d(x, y, z0, dx, dy, dz, shade=True)

    # Map index ticks back to the actual U,V values
    xticks = np.linspace(0, mU - 1, min(8, mU)).astype(int)
    yticks = np.linspace(0, mV - 1, min(8, mV)).astype(int)
    ax.set_xticks(xticks)
    ax.set_xticklabels(np.array(U)[xticks])
    ax.set_yticks(yticks)
    ax.set_yticklabels(np.array(V)[yticks])

    ax.set_xlabel("Union size  u")
    ax.set_ylabel("Intersection size  v")
    ax.set_zlabel(r"$P(U=u,V=v)$")
    ax.set_title(title)
    ax.view_init(elev=30, azim=-125)
    return fig, ax


def add_floor_contours(ax, U, V, Z, levels=8, offset=0.0, **contour_kw):
    UU, VV = np.meshgrid(U, V)
    cs = ax.contour(UU, VV, Z, levels=levels, offset=offset, zdir="z", **contour_kw)
    return cs


# Example usage right after plot_surface_3d(...):
# fig, ax = plot_surface_3d(U, V, Z, title="Bivariate — surface with floor contours")
# add_floor_contours(ax, U, V, Z, levels=10, offset=0.0, cmap="viridis")


def plot_surface_plotly(U, V, Z, title="Bivariate distribution — 3D (interactive)"):
    if not HAS_PLOTLY:
        print("Plotly not available. Install with: pip install plotly")
        return

    UU, VV = np.meshgrid(U, V)
    Z = np.asarray(Z, dtype=float)
    pos = Z[Z > 0]
    eps = pos.min() * 0.5 if pos.size else 1e-16

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


# -----------------------
# Run
# -----------------------
ar = AnalyticalResult(total_number=N, shard_sizes=shard_sizes)
U, V, Z, sum_probs, exact = compute_grid(ar, N, shard_sizes)

print(f"Sum of probabilities over grid: {sum_probs:.15f}")
print(f"Exact check (sum cases == denominator): {exact}")

outdir = Path(FIGURE_DIR)
outdir.mkdir(parents=True, exist_ok=True)
base = f"bivariate_heatmap_N{N}_sizes{'-'.join(map(str, shard_sizes))}"

plot_option_a(
    U,
    V,
    Z,
    outdir / f"{base}_log_zeros-darkblue.png",
    title=f"Bivariate distribution for N={N}, sizes={shard_sizes}",
)

# 3D surface (recommended)
fig3d, ax3d = plot_surface_3d(
    U, V, Z, title=f"Bivariate distribution for N={N}, sizes={shard_sizes} — 3D"
)
# Optional floor contours:
add_floor_contours(ax3d, U, V, Z, levels=12, offset=0.0, cmap="viridis")
plt.show()

plot_surface_plotly(U, V, Z)

mu, Sigma = gaussian_from_grid_new(U, V, Z)
Q = gaussian_on_grid(U, V, mu, Sigma, restrict_to_support=True)

metrics = compare_distributions(Z / Z.sum(), Q)
KS, m2, cdf_hat, F = mahalanobis_ks(U, V, Z, mu, Sigma)
E_M2, E_M4 = mardia_moments(U, V, Z, mu, Sigma)
R_angle = angle_uniformity(U, V, Z, mu, Sigma)
cond = conditional_linearity(U, V, Z)

print("Global distances:", metrics)
print(f"Mahalanobis KS (vs χ²₂): {KS:.4f}   E[M²]={E_M2:.3f}   E[M⁴]={E_M4:.3f}")
print(f"Angle uniformity R={R_angle:.4f}")
print(
    f"E[U|V]=a+bv: a={cond['intercept']:.3f}, b={cond['slope']:.3f}, R²={cond['R2']:.5f}, "
    f"Var(U|V) CV={cond['cv_cond_var']:.3f}"
)
# Optional Mardia skewness:
print("Mardia β1,2 (skewness):", mardia_skewness(U, V, Z, mu, Sigma))
