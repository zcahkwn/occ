"""
This script plots the PDF of the union and the intersection of the parties.

The union probability is the probability that at least one party has a number in the union of the numbers covered by the parties;
The intersection probability is the probability that at least one party has a number in the intersection of the numbers covered by the parties;

These values are calculated using the analytical result in occenv.analytical.AnalyticalResult.

The script also plots the approximated normal distribution using theoretical mean and standard deviation.

The script also plots the reverse cumulative distribution and the cumulative distribution.

"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from occenv.analytical import AnalyticalResult
from occenv.approximated import ApproximatedResult
from occenv.constants import FIGURE_DIR

N = 50

shard_sizes = [25, 36, 24]
m = len(shard_sizes)
alpha = [n / N for n in shard_sizes]

numbers_covered = np.arange(0, N + 1, 1)
numbers_intersected = np.arange(0, N + 1, 1)
union_pmf = []
intersect_pmf = []

analytical = AnalyticalResult(N, shard_sizes)
approx = ApproximatedResult(N, shard_sizes)

for number_covered in numbers_covered:
    union_probability = analytical.union_prob(number_covered)
    union_pmf.append(union_probability)

for intersection in numbers_intersected:
    intersect_probability = analytical.intersect_prob(intersection)
    intersect_pmf.append(intersect_probability)


def mu_calculation(pmf: list[float]) -> float:
    return np.sum([i * pmf[i] for i in range(1, N + 1)])


def sigma_calculation(pmf: list[float]) -> float:
    return np.sqrt(
        np.sum([(i - mu_calculation(pmf)) ** 2 * pmf[i] for i in range(1, N + 1)])
    )


def mae_calculation(normal_pdf: list[float], approx_pdf: list[float]) -> float:
    return np.mean(np.abs(normal_pdf - approx_pdf))


def sse_calculation(normal_pdf: list[float], approx_pdf: list[float]) -> float:
    return np.sum((normal_pdf - approx_pdf) ** 2)


x_continuous = np.linspace(min(numbers_covered) - 1, max(numbers_covered) + 1, 500)

"""
union
"""

# Plot the bar chart of the discrete PMF for union and plot the normal approximation on top of the PMF
plt.bar(numbers_covered, union_pmf, alpha=0.6, label="Probability Mass Function")
# Calculate the analytical mean and standard deviation of the discrete union PMF
mu_union = mu_calculation(union_pmf)
sigma_union = sigma_calculation(union_pmf)
normal_union = stats.norm.pdf(x_continuous, mu_union, sigma_union)
plt.plot(x_continuous, normal_union, "r-", label="Analytical normal distribution")
plt.legend()
plt.xlabel("Total union size ($N'$)")
plt.ylabel("$P(X=N')$")
plt.title("PMF for N={N},$S_{len(shard_sizes)}$={shard_sizes}")
plt.grid(True)
# plt.savefig(FIGURE_DIR / f"union_pmf{shard_sizes}.pdf")
plt.show()


# Calculate the approximated mean and standard deviation for the union PMF
prob_union_approx = 1 - np.prod([1 - alpha_i for alpha_i in alpha])
mu_union_approx = N * prob_union_approx

var_union_approx = N * prob_union_approx * (1 - prob_union_approx) + N * (N - 1) * (
    -np.prod([1 - alpha_i for alpha_i in alpha]) ** 2
    + np.prod([(N - shard) * (N - shard - 1) / (N * (N - 1)) for shard in shard_sizes])
)
sd_union_approx = np.sqrt(var_union_approx)


# Plot the two normal distributions in one figure
normal_union_approx = stats.norm.pdf(x_continuous, mu_union_approx, sd_union_approx)
normal_union = stats.norm.pdf(x_continuous, mu_union, sigma_union)

plt.plot(x_continuous, normal_union, "b", label="Analytical Normal Distribution")
plt.plot(
    x_continuous, normal_union_approx, "r--", label="Approximated Normal Distribution"
)
plt.legend()
plt.xlabel("Total union size ($N'$)")
plt.ylabel("$P(X=N')$")
plt.title("PMF for N={N},$S_{len(shard_sizes)}$={shard_sizes}")
plt.grid(True)
plt.savefig(FIGURE_DIR / f"union_pmf_approx{shard_sizes}.pdf")
plt.show()


# Compare the approximated mean and standard deviation with the analytical mean and standard deviation for union
print(f"Analytical mean for union (μ): {mu_union}")
print(f"Analytical standard deviation for union (σ): {sigma_union}")
print(f"Approximated mean for union (μ): {mu_union_approx}")
print(f"Approximated standard deviation for union (σ): {sd_union_approx}")

# Error between the two normal distributions
mae_union_norm = mae_calculation(normal_union, normal_union_approx)
sse_union_norm = sse_calculation(normal_union, normal_union_approx)

print(
    f"Error between the two normals: MAE = {mae_union_norm:.5f}, SSE = {sse_union_norm:.5f}"
)


# Plot the complementary cumulative distribution function:
complementary_cumulative_probabilities_union = np.cumsum(union_pmf[::-1])[::-1]
plt.plot(
    numbers_covered,
    complementary_cumulative_probabilities_union,
    marker="o",
    linestyle="-",
)
plt.xlabel("Total union size ($N'$)")
plt.ylabel("$P(X\geq N')$")
plt.title(
    f"Union Complementary Cumulative Distribution Function for N={N},$S_{len(shard_sizes)}$={shard_sizes}"
)
plt.grid(True)
plt.savefig(FIGURE_DIR / f"union_ccdf_{shard_sizes}.pdf")
plt.show()


# Plot the cumulative distribution function:
cumulative_probabilities_union = np.cumsum(union_pmf)
plt.plot(numbers_covered, cumulative_probabilities_union, marker="o", linestyle="-")
plt.xlabel("Total union size ($N'$)")
plt.ylabel("$P(X\leq N')$")
plt.title(
    f"UnionCumulative Distribution Function for N={N},$S_{len(shard_sizes)}$={shard_sizes}"
)
plt.grid(True)
plt.show()


"""
intersect
"""
# repeat the same for the intersection

# Plot the bar chart of the discrete PMF for intersection and plot the normal approximation on top of the PMF
plt.bar(
    numbers_intersected, intersect_pmf, alpha=0.6, label="Probability Mass Function"
)
mu_intersect = mu_calculation(intersect_pmf)
sigma_intersect = sigma_calculation(intersect_pmf)
normal_intersect = stats.norm.pdf(x_continuous, mu_intersect, sigma_intersect)
plt.plot(x_continuous, normal_intersect, "r-", label="Analytical normal distribution")
plt.legend()
plt.xlabel("Intersection size ($N'$)")
plt.ylabel("$P(X=N')$")
plt.title(f"Intersection PMF for N={N},$S_{len(shard_sizes)}$={shard_sizes}")
plt.grid(True)
# plt.savefig(FIGURE_DIR / f"intersect_pmf_{shard_sizes}.pdf")
plt.show()


# Calculate the approximated mean and standard deviation of the intersection PMF
prob_intersect_approx = np.prod([alpha_i for alpha_i in alpha])
mu_intersect_approx = N * prob_intersect_approx

var_intersect_approx = N * prob_intersect_approx * (1 - prob_intersect_approx) + N * (
    N - 1
) * (
    -np.prod([alpha_i for alpha_i in alpha]) ** 2
    + np.prod([(alpha_i**2 + (alpha_i**2 - alpha_i) / (N - 1)) for alpha_i in alpha])
)
sd_intersect_approx = np.sqrt(var_intersect_approx)


# Plot the two normal distributions in one figure
normal_intersect_approx = stats.norm.pdf(
    x_continuous, mu_intersect_approx, sd_intersect_approx
)
normal_intersect = stats.norm.pdf(x_continuous, mu_intersect, sigma_intersect)

plt.plot(x_continuous, normal_intersect, "b", label="Analytical Normal Distribution")
plt.plot(
    x_continuous,
    normal_intersect_approx,
    "r--",
    label="Approximated Normal Distribution",
)
plt.legend()
plt.xlabel("Intersection size ($N'$)")
plt.ylabel("$P(X=N')$")
plt.title("PMF for N={N},$S_{len(shard_sizes)}$={shard_sizes}")
plt.grid(True)
plt.savefig(FIGURE_DIR / f"intersect_pmf_approx{shard_sizes}.pdf")
plt.show()

# Compare the approximated mean and standard deviation with the analytical mean and standard deviation for intersection
print(f"Analytical Mean for intersection (μ): {mu_intersect}")
print(f"Analytical Standard Deviation for intersection (σ): {sigma_intersect}")
print(f"Approximated Mean for intersection (μ): {mu_intersect_approx}")
print(f"Approximated Standard Deviation for intersection (σ): {sd_intersect_approx}")

# Error between the two normal distributions
mae_intersect_norm = mae_calculation(normal_intersect, normal_intersect_approx)
sse_intersect_norm = sse_calculation(normal_intersect, normal_intersect_approx)
print(
    f"Error between the two normals: MAE = {mae_intersect_norm:.5f}, SSE = {sse_intersect_norm:.5f}"
)


# Plot the complementary cumulative distribution function:
complementary_cumulative_probabilities_intersect = np.cumsum(intersect_pmf[::-1])[::-1]
plt.plot(
    numbers_intersected,
    complementary_cumulative_probabilities_intersect,
    marker="o",
    linestyle="-",
)
plt.xlabel("Intersection size ($N'$)")
plt.ylabel("$P(X\geq N')$")
plt.title(
    f"Intersection Complementary Cumulative Distribution Function for N={N},$S_{len(shard_sizes)}$={shard_sizes}"
)
plt.grid(True)
# plt.savefig(FIGURE_DIR / f"intersect_ccdf_{shard_sizes}.pdf")
plt.show()


# Plot the cumulative distribution function:
cumulative_probabilities_intersect = np.cumsum(intersect_pmf)
plt.plot(
    numbers_intersected, cumulative_probabilities_intersect, marker="o", linestyle="-"
)
plt.xlabel("Intersection size ($N'$)")
plt.ylabel("$P(X\leq N')$")
plt.title(
    f"Intersection Cumulative Distribution Function for N={N},$S_{len(shard_sizes)}$={shard_sizes}"
)
plt.grid(True)
plt.show()
