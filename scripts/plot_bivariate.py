from occenv.analytical_bivariate import AnalyticalBivariate
from occenv.approximated import ApproximatedResult
from occenv.constants import FIGURE_DIR
from occenv.plotting_3d import (
    plot_heatmap_ellipse,
    plot_surface_3d,
    plot_surface_plotly,
)
from occenv.utils_bivariate import Gaussian2D

N = 200
shard_sizes = [150, 140, 160]

# Build grid (U, V, Z) from the analytical model
ar = AnalyticalBivariate(total_number=N, shard_sizes=shard_sizes)
U_vals, V_vals, Z_vals = ar.bivariate_grid()
mu = ar.bivariate_mu()
Sigma = ar.bivariate_matrix()


# Plot 2D heatmap + ellipses overlays indicating confidence levels
plot_heatmap_ellipse(
    U_vals,
    V_vals,
    Z_vals,
    mu,
    Sigma,
    FIGURE_DIR / f"bivariate_heatmap_N{N}_sizes{'-'.join(map(str, shard_sizes))}.png",
    title=f"Bivariate distribution for N={N}, sizes={shard_sizes}",
)

# Plot 3D surface
plot_surface_3d(
    U_vals,
    V_vals,
    Z_vals,
    title=f"Bivariate distribution for N={N}, sizes={shard_sizes} — 3D",
)

# Plot 3D surface (interactive)
plot_surface_plotly(
    U_vals,
    V_vals,
    Z_vals,
    title=f"Bivariate distribution for N={N}, sizes={shard_sizes} — 3D (interactive)",
)

# Compare analytical and approximated mu and Sigma
approx = ApproximatedResult(N, shard_sizes)
approx_mu = approx.bivariate_mu_approx()
approx_Sigma = approx.bivariate_matrix_approx()

print("mu =", mu, "\napprox_mu =", approx_mu)
print("Sigma =\n", Sigma, "\napprox_Sigma =\n", approx_Sigma)
print(
    "Eigenvalues =",
    Gaussian2D(mu, Sigma).evals,
    "\napprox_Eigenvalues =",
    Gaussian2D(approx_mu, approx_Sigma).evals,
)
print(
    "Eigenvectors (columns) =\n",
    Gaussian2D(mu, Sigma).evecs,
    "\napprox_Eigenvectors (columns) =\n",
    Gaussian2D(approx_mu, approx_Sigma).evecs,
)
