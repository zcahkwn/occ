import matplotlib.pyplot as plt
import numpy as np
from occenv.constants import FIGURE_DIR
from occenv.analytical import Analytical_result

N = 10
shard_sizes = [5, 6, 4]
numbers_covered = np.arange(1, N + 1, 1)
probabilities = []

for number_covered in numbers_covered:
    parties_list = Analytical_result(N, shard_sizes)
    probability = parties_list.collude_prob(number_covered)
    probabilities.append(probability)

plt.plot(numbers_covered, probabilities, marker="o", linestyle="-")
plt.xlabel("Total numbers covered")
plt.ylabel("Probability")
plt.title(f"Collusion Probability for N={N},$S_{len(shard_sizes)}$={shard_sizes}")
plt.grid(True)

plt.savefig(FIGURE_DIR / f"collusion_probability_{shard_sizes}.pdf")
plt.show()


cumulative_probabilities = cumulative_probabilities = np.cumsum(probabilities[::-1])[
    ::-1
]
plt.plot(numbers_covered, cumulative_probabilities, marker="o", linestyle="-")
plt.xlabel("Total numbers covered")
plt.ylabel("Cumulative Probability")
plt.title(
    f"Cumulative Collusion Probability for N={N},$S_{len(shard_sizes)}$={shard_sizes}"
)
plt.grid(True)

plt.savefig(FIGURE_DIR / f"cumulative_collusion_probability_{shard_sizes}.pdf")
plt.show()
