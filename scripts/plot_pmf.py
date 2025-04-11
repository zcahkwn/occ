import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from occenv.analytical import AnalyticalResult
from occenv.constants import FIGURE_DIR

N = 10

shard_sizes = [5, 6, 4]
m = len(shard_sizes)
alpha = [n / N for n in shard_sizes]

numbers_covered = np.arange(1, N + 1, 1)
numbers_intersected = np.arange(0, N + 1, 1)
union_pmf = []
intersect_pmf = []

for number_covered in numbers_covered:
    parties_list = AnalyticalResult(N, shard_sizes)
    union_probability = parties_list.union_prob(number_covered)
    union_pmf.append(union_probability)

for intersection in numbers_intersected:
    intersect_probability = parties_list.intersect_prob(intersection)
    intersect_pmf.append(intersect_probability)


"""
union
"""
plt.plot(numbers_covered, union_pmf, marker="o", linestyle="-")
plt.xlabel("Total numbers covered ($N'$)")
plt.ylabel("$P(X=N')$")
plt.title(f"Probability Mass Function for N={N},$S_{len(shard_sizes)}$={shard_sizes}")
plt.grid(True)

plt.savefig(FIGURE_DIR / f"union_pmf_{shard_sizes}.pdf")
plt.show()


mu_union = np.sum([i * union_pmf[i - 1] for i in range(1, N + 1)])
sigma_union = np.sqrt(
    np.sum([(i - mu_union) ** 2 * union_pmf[i - 1] for i in range(1, N + 1)])
)
print(f"Mean (μ): {mu_union}")
print(f"Standard Deviation (σ): {sigma_union}")
plt.bar(numbers_covered, union_pmf, alpha=0.6, label="Probability Mass Function")

# Plot the normal approximation on top of the PMF
x_continuous = np.linspace(min(numbers_covered) - 1, max(numbers_covered) + 1, 500)
normal_pdf = stats.norm.pdf(x_continuous, mu_union, sigma_union)
plt.plot(x_continuous, normal_pdf, "r-", label="Normal Approximation")

plt.legend()
plt.title("Discrete Distribution vs. Normal Approximation")
plt.savefig(FIGURE_DIR / f"union_normal_approx_{shard_sizes}.pdf")
plt.show()

# Plot the complementary cumulative distribution function:
complementary_cumulative_probabilities = np.cumsum(union_pmf[::-1])[::-1]
plt.plot(
    numbers_covered, complementary_cumulative_probabilities, marker="o", linestyle="-"
)
plt.xlabel("Total numbers covered ($N'$)")
plt.ylabel("$P(X\geq N')$")
plt.title(
    f"Complementary Cumulative Distribution Function for N={N},$S_{len(shard_sizes)}$={shard_sizes}"
)
plt.grid(True)
plt.savefig(FIGURE_DIR / f"union_ccdf_{shard_sizes}.pdf")
plt.show()


"""
intersect
"""

# Plot the intersection PMF
plt.plot(numbers_intersected, intersect_pmf, marker="o", linestyle="-")
plt.xlabel("Intersection")
plt.ylabel("Probability")
plt.title(f"Intersection PMF for N={N},$S_{len(shard_sizes)}$={shard_sizes}")
plt.grid(True)
plt.savefig(FIGURE_DIR / f"intersect_pmf_{shard_sizes}.pdf")
plt.show()

# Calculate the mean and standard deviation of the intersection PMF
mu_intersect = np.sum([i * intersect_pmf[i] for i in numbers_intersected])
sigma_intersect = np.sqrt(
    np.sum([(i - mu_intersect) ** 2 * intersect_pmf[i] for i in numbers_intersected])
)
print(f"Analytical Mean (μ): {mu_intersect}")
print(f"Analytical Standard Deviation (σ): {sigma_intersect}")
plt.bar(numbers_intersected, intersect_pmf, alpha=0.6, label="PMF")


# Plot the normal approximation (using mean and sd of PMF)
intersect_continuous = np.linspace(
    min(numbers_intersected) - 1, max(numbers_intersected) + 1, 500
)
normal_intersect = stats.norm.pdf(intersect_continuous, mu_intersect, sigma_intersect)
plt.plot(intersect_continuous, normal_intersect, "r-", label="Normal Approximation")

plt.legend()
plt.grid(True)
plt.title(
    f"PMF vs. Normal Approximation for N={N},$S_{len(shard_sizes)}$={shard_sizes}"
)
plt.savefig(FIGURE_DIR / f"intersect_normal_approx_{shard_sizes}.pdf")
plt.show()


# Calculate an error metric between the PMF normal approximation and the real PMF.
normal_intersect_at_points = stats.norm.pdf(
    numbers_intersected, mu_intersect, sigma_intersect
)
# Mean Absolute Error:
mae = np.mean(np.abs(normal_intersect_at_points - intersect_pmf))
print(f"Mean Absolute Error (MAE) between the PMF and normal approximation: {mae:.5f}")
# Sum of squared errors:
sse = np.sum((normal_intersect_at_points - intersect_pmf) ** 2)
print(
    f"Sum of Squared Errors (SSE) between the PMF and normal approximation: {sse:.5f}"
)

# CLT Normal approximation
p = np.prod([n / N for n in shard_sizes])
mu_intersect_clt = N * p
sigma_intersect_clt = np.sqrt(
    N * p * (1 - p)
    + N
    * (N - 1)
    * (
        np.prod([a**2 + (a**2 - a) / (N - 1) for a in alpha])
        - np.prod([a**2 for a in alpha])
    )
)

print("CLT Approximation Parameters:")
print("Mean (mu_clt) =", mu_intersect_clt)
print("Standard deviation (sigma_clt) =", sigma_intersect_clt)


# Compare the PMF and the CLT Normal Density
x_int = np.arange(0, N + 1)
intersect_pmf = stats.norm.pdf(x_int, loc=mu_intersect, scale=sigma_intersect)

# Evaluate the CLT normal density at those same integer points.
pdf_clt_int = stats.norm.pdf(x_int, loc=mu_intersect_clt, scale=sigma_intersect_clt)

# Compute errors between the PMF and CLT normal density
mae_pmf_vs_clt = np.mean(np.abs(pdf_clt_int - intersect_pmf))
sse_pmf_vs_clt = np.sum((pdf_clt_int - intersect_pmf) ** 2)
print(
    f"Error between PMF and CLT normal density: MAE = {mae_pmf_vs_clt:.5f}, SSE = {sse_pmf_vs_clt:.5f}"
)

# Plotting PMF and CLT Normal approximation in one figure
plt.bar(numbers_intersected, intersect_pmf, alpha=0.6, label="PMF")

x_cont = np.linspace(0, N, 500)
pdf_clt_smooth = stats.norm.pdf(x_cont, loc=mu_intersect_clt, scale=sigma_intersect_clt)

plt.plot(x_cont, pdf_clt_smooth, "r-", lw=2, label="CLT Normal Approximation")

plt.xlabel("Intersection Size")
plt.ylabel("Probability / Density")
plt.title(
    f"Discrete PMF vs. CLT Normal Approximation for N={N},$S_{len(shard_sizes)}$={shard_sizes}"
)
plt.legend()
plt.grid(True)
plt.show()


# Plot the Two Normal Approximations in one figure
pdf_intersect_clt = stats.norm.pdf(
    x_cont, loc=mu_intersect_clt, scale=sigma_intersect_clt
)
pdf_intersect_pmf = stats.norm.pdf(x_cont, loc=mu_intersect, scale=sigma_intersect)

plt.plot(x_cont, pdf_intersect_clt, label="CLT Normal Approximation", lw=2)
plt.plot(x_cont, pdf_intersect_pmf, "r--", label="PMF Normal Approximation", lw=2)
plt.xlabel("Intersection Size")
plt.ylabel("Density")
plt.title(
    f"Comparison of Two Normal Approximations for N={N},$S_{len(shard_sizes)}$={shard_sizes}"
)
plt.legend()
plt.grid(True)
plt.show()

# Error between the Two Normal Approximations
mae_norm = np.mean(np.abs(pdf_intersect_clt - pdf_intersect_pmf))
sse_norm = np.sum((pdf_intersect_clt - pdf_intersect_pmf) ** 2)
print(f"Error between the two normals: MAE = {mae_norm:.5f}, SSE = {sse_norm:.5f}")
