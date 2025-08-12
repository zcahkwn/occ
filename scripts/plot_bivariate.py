# plot.py
from pathlib import Path
import matplotlib.pyplot as plt

from occenv.analytical_bivariate import AnalyticalBivariate
from occenv.constants import FIGURE_DIR

from occenv.utils_bivariate import (
    BivariateGrid,
    Gaussian2D,
    compare_distributions,
)
from occenv.plotting_3d import (
    plot_option_a,
    plot_surface_3d,
    add_floor_contours,
    plot_surface_plotly,
)

N = 200
shard_sizes = [150, 140, 160]

# Build grid (U, V, Z) from the analytical model using the classmethod
ar = AnalyticalBivariate(total_number=N, shard_sizes=shard_sizes)
grid, sum_probs, exact = BivariateGrid.from_analytical(ar, N, shard_sizes)

print(f"Sum of probabilities over grid: {sum_probs:.15f}")
print(f"Exact check (sum cases == denominator): {exact}")

outdir = Path(FIGURE_DIR)
outdir.mkdir(parents=True, exist_ok=True)
base = f"bivariate_heatmap_N{N}_sizes{'-'.join(map(str, shard_sizes))}"

# 2D heatmap + Gaussian overlays (plotting function builds/uses classes internally too)
plot_option_a(
    grid.U,
    grid.V,
    grid.Z,
    outdir / f"{base}_log_zeros-darkblue.png",
    title=f"Bivariate distribution for N={N}, sizes={shard_sizes}",
)

# 3D surface + optional floor contours
fig3d, ax3d = plot_surface_3d(
    grid.U,
    grid.V,
    grid.Z,
    title=f"Bivariate distribution for N={N}, sizes={shard_sizes} — 3D",
)
add_floor_contours(ax3d, grid.U, grid.V, grid.Z, levels=12, offset=0.0, cmap="viridis")
plt.show()

plot_surface_plotly(grid.U, grid.V, grid.Z)

# ---- Fit Gaussian and compute diagnostics using classes ----
mu, Sigma = grid.mean_cov()
gauss = Gaussian2D(mu, Sigma)

# Discretize the fitted Gaussian onto the same support for comparison
Q = grid.discretize_gaussian(gauss, restrict_to_support=True)

# Global distances (kept as a tiny helper function)
metrics = compare_distributions(grid.P, Q)

# Mahalanobis KS against χ²₂
KS, m2, cdf_hat, F = grid.mahalanobis_ks(gauss)

# Mardia moments (E[M²], E[M⁴]), angle uniformity, conditional linearity
E_M2, E_M4 = grid.mardia_moments(gauss)
R_angle = grid.angle_uniformity(gauss)
cond = grid.conditional_linearity()

# Mardia skewness β1,2
beta1p = grid.mardia_skewness(gauss)

# ---- Print diagnostics ----
print("Global distances:", metrics)
print(f"Mahalanobis KS (vs χ²₂): {KS:.4f}   E[M²]={E_M2:.3f}   E[M⁴]={E_M4:.3f}")
print(f"Angle uniformity R={R_angle:.4f}")
print(
    f"E[U|V]=a+bv: a={cond['intercept']:.3f}, b={cond['slope']:.3f}, "
    f"R²={cond['R2']:.5f}, Var(U|V) CV={cond['cv_cond_var']:.3f}"
)
print("Mardia β1,2 (skewness):", beta1p)
