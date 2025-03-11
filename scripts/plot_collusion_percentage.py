import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from occenv.analytical import AnalyticalResult
from occenv.constants import FIGURE_DIR

N = 100
shard_sizes = [70, 60]
numbers_covered = np.arange(1, N + 1, 1)
probabilities = []

for number_covered in numbers_covered:
    parties_list = AnalyticalResult(N, shard_sizes)
    probability = parties_list.collude_prob(number_covered)
    probabilities.append(probability)

plt.plot(numbers_covered, probabilities, marker="o", linestyle="-")
plt.xlabel("Total numbers covered")
plt.ylabel("Probability")
plt.title(f"Collusion Probability for N={N},$S_{len(shard_sizes)}$={shard_sizes}")
plt.grid(True)

# plt.savefig(FIGURE_DIR / f"collusion_probability_{shard_sizes}.pdf")
plt.show()


mu = np.sum([i * probabilities[i - 1] for i in range(1, N + 1)])
sigma = np.sqrt(np.sum([(i - mu) ** 2 * probabilities[i - 1] for i in range(1, N + 1)]))
print(f"Mean (μ): {mu}")
print(f"Standard Deviation (σ): {sigma}")
plt.bar(numbers_covered, probabilities, alpha=0.6, label="Discrete PDF")

# Plot the normal approximation
x_continuous = np.linspace(min(numbers_covered) - 1, max(numbers_covered) + 1, 100)
normal_pdf = stats.norm.pdf(x_continuous, mu, sigma)
plt.plot(x_continuous, normal_pdf, "r-", label="Normal Approximation")

plt.legend()
plt.title("Discrete Distribution vs. Normal Approximation")
plt.show()

# p = 0.88
# binom_pdf = stats.binom.pmf(numbers_covered, N, p)

# plt.figure(figsize=(10, 5))
# plt.bar(numbers_covered, binom_pdf, alpha=0.6, color='green', label="Binomial Distribution (p=0.88)")
# plt.xlabel("Number of successes")
# plt.ylabel("Probability")
# plt.title(f"Binomial Distribution: n={N}, p={p}")
# plt.legend()
# plt.grid(True)
# plt.show()

# reverse_cumulative_probabilities = np.cumsum(probabilities[::-1])[::-1]
# plt.plot(numbers_covered, reverse_cumulative_probabilities, marker="o", linestyle="-")
# plt.xlabel("Total numbers covered")
# plt.ylabel("Cumulative Probability")
# plt.title(
#     f"Reverse Cumulative Collusion Probability for N={N},$S_{len(shard_sizes)}$={shard_sizes}"
# )
# plt.grid(True)

# plt.savefig(FIGURE_DIR / f"reverse_cumulative_collusion_probability_{shard_sizes}.pdf")
# plt.show()

# cumulative_probabilities = np.cumsum(probabilities)
# plt.plot(numbers_covered, cumulative_probabilities, marker="o", linestyle="-")
# plt.xlabel("Total numbers covered")
# plt.ylabel("Cumulative Probability")
# plt.title(
#     f"Cumulative Collusion Probability for N={N},$S_{len(shard_sizes)}$={shard_sizes}"
# )
# plt.grid(True)

# plt.savefig(FIGURE_DIR / f"cumulative_collusion_probability_{shard_sizes}.pdf")
# plt.show()
