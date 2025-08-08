"""
This script plots the PDF of the union and the intersection of the parties, together with the approximated normal distribution.

The script also plots the reverse cumulative distribution and the cumulative distribution.
"""

import numpy as np
from occenv.analytical import AnalyticalResult
from occenv.utils import (
    mu_calculation,
    sigma_calculation,
    mae_calculation,
    sse_calculation,
    norm_pdf,
)
from occenv.plotting_2d import plot_line_graph, plot_pmf_with_normals
from occenv.approximated import ApproximatedResult

N = 200
shard_sizes = [150, 140]
m = len(shard_sizes)
alpha = [n / N for n in shard_sizes]

analytical = AnalyticalResult(N, shard_sizes)
approx = ApproximatedResult(N, shard_sizes)

numbers_range = np.arange(0, N + 1, 1)
x_continuous = np.linspace(min(numbers_range) - 1, max(numbers_range) + 1, 500)

problems = ["union", "intersect"]

for problem in problems:
    # Calculate the PMF for the problem
    globals()[f"{problem}_pmf"] = []
    prob_method = getattr(analytical, f"{problem}_prob")
    for number_in_range in numbers_range:
        probability = prob_method(number_in_range)
        globals()[f"{problem}_pmf"].append(probability)

    pmf = globals()[f"{problem}_pmf"]
    mu_method = getattr(approx, f"{problem}_mu_approx")
    sd_method = getattr(approx, f"{problem}_sd_approx")

    # Calculate the analytical mean and standard deviation of the discrete PMF
    mu_analytical = mu_calculation(numbers_range, pmf)
    sigma_analytical = sigma_calculation(numbers_range, pmf)
    normal_analytical = norm_pdf(x_continuous, mu_analytical, sigma_analytical)

    # Calculate the approximated mean and standard deviation of the discrete PMF (using CLT)
    mu_approx = mu_method()
    sd_approx = sd_method()
    normal_approx = norm_pdf(x_continuous, mu_approx, sd_approx)

    # Error between the two normal distributions
    mae_norm = mae_calculation(normal_analytical, normal_approx)
    sse_norm = sse_calculation(normal_analytical, normal_approx)

    # Plot the bar chart of the discrete PMF and plot the normal approximation on top of the PMF
    plot_pmf_with_normals(
        numbers_range,
        pmf,
        x_continuous,
        mu_analytical,
        sigma_analytical,
        mu_approx,
        sd_approx,
        title=f"PMF for N={N},$S_{len(shard_sizes)}$={shard_sizes}",
    )

    # Compare the approximated mean and standard deviation with the analytical mean and standard deviation
    print(
        f"Analytical mean for {problem} (μ): {mu_analytical} \nApproximated mean for {problem} (μ): {mu_approx}"
    )
    print(
        f"Analytical standard deviation for {problem} (σ): {sigma_analytical} \nApproximated standard deviation for {problem} (σ): {sd_approx}"
    )
    print(f"Error between the two normals: MAE = {mae_norm:.5f}, SSE = {sse_norm:.5f}")

    # Plot the complementary cumulative distribution function and the cumulative distribution function
    cdf = np.cumsum(pmf)
    ccdf = np.cumsum(pmf[::-1])[::-1]
    plot_line_graph(
        numbers_range,
        ccdf,
        f"{problem} Complementary Cumulative Distribution Function",
        f"{problem} size ($N')",
    )
    plot_line_graph(
        numbers_range,
        cdf,
        f"{problem} Cumulative Distribution Function",
        f"{problem} size ($N')",
    )
