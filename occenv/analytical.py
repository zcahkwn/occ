from math import comb, prod
from typing import Iterable
import numpy as np
import itertools


class AnalyticalResult:
    def __init__(self, total_number: int, shard_sizes: list[int]):
        self.total_number = total_number
        self.shard_sizes = shard_sizes
        self.party_number = len(shard_sizes)

    def collude_cases(self, number_covered) -> int:
        """
        Calculate the number of cases of colluding to reconstruct the secret set.
        """

        if sum(self.shard_sizes) < number_covered:
            return 0

        def collude_cases_recursive(total_number: int, shard_sizes: list[int]) -> int:

            last_shard = shard_sizes[-1]
            rest_shard = shard_sizes[:-1]
            return sum(
                comb(k, k + last_shard - total_number)
                * comb(total_number, k)
                * (collude_cases_recursive(k, rest_shard) if rest_shard else 1)
                for k in np.arange(
                    start=max(rest_shard + [total_number - last_shard]),
                    stop=min(sum(rest_shard), total_number) + 1,
                    step=1,
                )
            )

        return comb(self.total_number, number_covered) * collude_cases_recursive(
            self.total_number, self.shard_sizes
        )

    def collude_prob(self, number_covered: int) -> float:
        """
        Calculate the probability of colluding to reconstruct the secret set.
        """
        return self.collude_cases(number_covered) / prod(
            comb(self.total_number, n) for n in self.shard_sizes
        )

    def rho(self, indices: Iterable[int]) -> float:
        product = prod(self.shard_sizes[i] for i in indices)
        k = len(indices)
        result = product / (self.total_number**k)
        print(
            f"When the list is {self.shard_sizes} and the combination is {indices}, rho = {result}"
        )
        return result

    def compute_sigma(self):
        sigma = 0.0
        for k in range(
            1, self.party_number + 1
        ):  # the outer summation - loop over k from 1 to m
            sum_k = 0.0
            for combo in itertools.combinations(
                range(self.party_number), k
            ):  # the inner summation - loop over all combinations of m choose k
                sum_k += self.rho(combo)
            sigma += ((-1) ** (k + 1)) * sum_k
        return sigma

    def occ_value(self):
        return self.rho(list(range(self.party_number)))


if __name__ == "__main__":
    compute = AnalyticalResult(10, [5, 6, 9])
    collution_probability = compute.collude_prob(10)
    sigma_value = compute.compute_sigma()
    occ_value = compute.occ_value()

    print("sigma =", sigma_value)
    print("occ =", occ_value)
    print("probability of collusion =", collution_probability)
