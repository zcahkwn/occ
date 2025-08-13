from occenv.constants import FIGURE_DIR
from occenv.analytical_univariate import AnalyticalUnivariate
from occenv.analytical_bivariate import AnalyticalBivariate
from occenv.approximated import ApproximatedResult
from occenv.plotting_3d import plot_heatmap
import numpy as np

N = 10
n_vals = np.arange(1, N)
x, y = np.meshgrid(n_vals, n_vals)

plot_dict = {
    "collusion_prob": {
        "title": f"Probability of Collusion with $N={N},m=2$",
        "z": np.vectorize(
            lambda n1, n2: AnalyticalUnivariate(N, [n1, n2]).union_prob(N)
        )(x, y),
    },
    "sigma": {
        "title": f"$\sigma$ with $N={N},m=2$",
        "z": np.vectorize(lambda n1, n2: ApproximatedResult(N, [n1, n2]).sigma_value())(
            x, y
        ),
    },
    "occ": {
        "title": f"OCC with $N={N},m=2$",
        "z": np.vectorize(lambda n1, n2: ApproximatedResult(N, [n1, n2]).occ_value())(
            x, y
        ),
    },
    "expected_jaccard": {
        "title": f"Expected Jaccard index with $N={N}$",
        "z": np.vectorize(lambda n1, n2: AnalyticalBivariate(N, [n1, n2]).jaccard_mu())(
            x, y
        ),
    },
    "estimated_jaccard": {
        "title": f"Estimated Jaccard index with $N={N}$",
        "z": np.vectorize(
            lambda n1, n2: ApproximatedResult(
                N, [n1, n2]
            ).jaccard_mu_approx_simplified()
        )(x, y),
    },
    "jaccard_difference": {
        "title": f"Difference between Expected and Estimated Jaccard index with $N={N}$",
        "z": np.vectorize(lambda n1, n2: AnalyticalBivariate(N, [n1, n2]).jaccard_mu())(
            x, y
        )
        - np.vectorize(
            lambda n1, n2: ApproximatedResult(
                N, [n1, n2]
            ).jaccard_mu_approx_simplified()
        )(x, y),
        "vmax": 0.05,
    },
}

for key, value in plot_dict.items():
    plot_heatmap(
        x=x,
        y=y,
        z=value["z"],
        xlabel="$n_1$",
        ylabel="$n_2$",
        title=value["title"],
        cmap="Blues",
        vmin=0,
        vmax=value.get("vmax", 1),
    )
