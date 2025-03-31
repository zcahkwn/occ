from math import comb, prod, floor
import numpy as np
import matplotlib.pyplot as plt
from occenv.constants import FIGURE_DIR 

class Collusion: 
    def __init__(self, total_number: int, shard_sizes: list[int], percentage: float):
        self.total_number = total_number
        self.shard_sizes = shard_sizes
        self.percentage = percentage
        self.number_coverage = floor(total_number * self.percentage)

    def collude_cases(self, total_number: int, shard_sizes: list[int]) -> int:
        """
        Calculate the number of cases of colluding to reconstruct the secret set.
        """

        last_shard = shard_sizes[-1]
        rest_shard = shard_sizes[:-1]
        return sum(
            comb(k, k + last_shard - total_number)
            * comb(total_number, k)
            * (self.collude_cases(k, rest_shard) if rest_shard else 1)
            for k in np.arange(
                start=max(rest_shard + [total_number - last_shard]),
                stop=min(sum(rest_shard), total_number) + 1,
                step=1,
            )
        )

    def collude_prob(self) -> float:
        """
        Calculate the probability of colluding to reconstruct the secret set.
        """

        if sum(self.shard_sizes) < self.number_coverage or any(shard > self.number_coverage for shard in self.shard_sizes):
            return 0
        
        numerator = comb(self.total_number, self.number_coverage) * self.collude_cases(self.number_coverage, self.shard_sizes)
        denominator = prod(comb(self.total_number, n) for n in self.shard_sizes)
        return numerator / denominator

if __name__ == "__main__":
    percentages = np.arange(0.1, 1.1, 0.1)
    probabilities = []

    for percentage in percentages:
        parties_list = Collusion(10, [4,7], percentage)
        probability = parties_list.collude_prob()
        probabilities.append(probability)

    plt.plot(percentages, probabilities, marker='o', linestyle='-')
    plt.xlabel("Percentage")
    plt.ylabel("Probability")
    plt.title("Collusion Probability vs Percentage")
    plt.grid(True)

    plt.savefig(FIGURE_DIR/"collusion_probability_4_7.pdf")
    plt.show()


    cumulative_probabilities = cumulative_probabilities = np.cumsum(probabilities[::-1])[::-1]
    plt.plot(percentages, cumulative_probabilities, marker='o', linestyle='-')
    plt.xlabel("Percentage")
    plt.ylabel("Cumulative Probability")
    plt.title("Cumulative Collusion Probability vs Percentage")
    plt.grid(True)

    plt.savefig(FIGURE_DIR/"cumulative_collusion_probability_4_7.pdf")
    plt.show()

