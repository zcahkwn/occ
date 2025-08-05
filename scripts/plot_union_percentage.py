"""
This script plots the PDF of the union of the parties.

The union probability is the probability that at least one party has a number in the union of the numbers covered by the parties, whichis calculated using the analytical result in occenv.analytical.AnalyticalResult.

The script also plots the approximated normal distribution using theoretical mean and standard deviation.

The script also plots the reverse cumulative distribution and the cumulative distribution.

"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from occenv.analytical import AnalyticalResult
from occenv.constants import FIGURE_DIR

N = 500
shard_sizes = [400, 300, 250, 350]
numbers_covered = np.arange(1, N + 1, 1)

parties_list = AnalyticalResult(N, shard_sizes)
probabilities = [
    parties_list.union_prob(number_covered) for number_covered in numbers_covered
]

# Plot the discrete distribution
plt.plot(numbers_covered, probabilities, marker="o", linestyle="-")
plt.xlabel("Total numbers covered")
plt.ylabel("Probability")
plt.title(f"Collusion Probability for N={N},$S_{len(shard_sizes)}$={shard_sizes}")
plt.grid(True)

# plt.savefig(FIGURE_DIR / f"collusion_probability_{shard_sizes}.pdf")
plt.show()


# Calculate the discrete mean and standard deviation
mu = np.sum([i * probabilities[i - 1] for i in numbers_covered])
sigma = np.sqrt(np.sum([(i - mu) ** 2 * probabilities[i - 1] for i in numbers_covered]))
print(f"Mean (μ): {mu}")
print(f"Standard Deviation (σ): {sigma}")

# Calculate the theoretical mean and standard deviation
prob_rv = 1 - np.prod([1 - shard / N for shard in shard_sizes])
mu_theoretical = N * prob_rv

var_theoretical = N * prob_rv * (1 - prob_rv) + N * (N - 1) * (
    -np.prod([1 - shard / N for shard in shard_sizes]) ** 2
    + np.prod([(N - shard) * (N - shard - 1) / (N * (N - 1)) for shard in shard_sizes])
)
sigma_theoretical = np.sqrt(var_theoretical)
print(f"Theoretical Mean (μ): {mu_theoretical}")
print(f"Theoretical Standard Deviation (σ): {sigma_theoretical}")


# Plot the discrete distribution and the approximated normal distribution
plt.bar(numbers_covered, probabilities, alpha=0.6, label="Discrete PDF")
x_continuous = np.linspace(0, N + 1, 1000)
normal_pdf = stats.norm.pdf(
    x_continuous, mu, sigma
)  # approximated normal distribution using discrete mean and standard deviation
plt.plot(x_continuous, normal_pdf, "r-", label="Normal Approximation")

plt.legend()
plt.title("Discrete Distribution vs. Normal Approximation")
plt.show()


# Plot the reverse cumulative distribution
reverse_cumulative_probabilities = np.cumsum(probabilities[::-1])[::-1]
plt.plot(numbers_covered, reverse_cumulative_probabilities, marker="o", linestyle="-")
plt.xlabel("Total numbers covered")
plt.ylabel("Cumulative Probability")
plt.title(
    f"Reverse Cumulative Collusion Probability for N={N},$S_{len(shard_sizes)}$={shard_sizes}"
)
plt.grid(True)
# plt.savefig(FIGURE_DIR / f"reverse_cumulative_collusion_probability_{shard_sizes}.pdf")
plt.show()


# Plot the cumulative distribution
cumulative_probabilities = np.cumsum(probabilities)
plt.plot(numbers_covered, cumulative_probabilities, marker="o", linestyle="-")
plt.xlabel("Total numbers covered")
plt.ylabel("Cumulative Probability")
plt.title(
    f"Cumulative Collusion Probability for N={N},$S_{len(shard_sizes)}$={shard_sizes}"
)
plt.grid(True)
# plt.savefig(FIGURE_DIR / f"cumulative_collusion_probability_{shard_sizes}.pdf")
plt.show()
