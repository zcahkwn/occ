from math import comb, prod, lgamma, log, exp
from typing import Iterable
import numpy as np
import itertools


class AnalyticalResult:
    def __init__(self, total_number: int, shard_sizes: list[int]):
        self.total_number = total_number
        self.shard_sizes = shard_sizes
        self.party_number = len(shard_sizes)

    def union_cases(self, number_covered) -> int:
        """
        Calculate the number of cases when the union of the shards have size number_covered.
        """

        if sum(self.shard_sizes) < number_covered:
            return 0

        def union_cases_recursive(number_covered: int, shard_sizes: list[int]) -> int:

            last_shard = shard_sizes[-1]
            rest_shard = shard_sizes[:-1]
            return sum(
                comb(k, k + last_shard - number_covered)
                * comb(number_covered, k)
                * (union_cases_recursive(k, rest_shard) if rest_shard else 1)
                for k in np.arange(
                    start=max(rest_shard + [number_covered - last_shard]),
                    stop=min(sum(rest_shard), number_covered) + 1,
                    step=1,
                )
            )

        return comb(self.total_number, number_covered) * union_cases_recursive(
            number_covered, self.shard_sizes
        )

    def union_prob(self, number_covered: int) -> float:
        """
        Calculate the probability of colluding to reconstruct the secret set.
        """
        return self.union_cases(number_covered) / prod(
            comb(self.total_number, n) for n in self.shard_sizes
        )

    def intersect_cases(self, overall_intersect: int) -> float:
        """
        Calculate the number of cases when the intersection of the shards have size intersect.
        """
        if not (
            max(0, sum(self.shard_sizes) - (self.party_number - 1) * self.total_number)
            <= overall_intersect
            <= min(self.shard_sizes)
        ):
            return 0.0

        def intersect_cases_recursive(
            number_intersect: int, remaining_shards: list[int]
        ) -> int:
            if not remaining_shards:
                # base case: intersection of zero parties is N (the universe)
                return 1 if number_intersect == self.total_number else 0

            last_shard = remaining_shards[-1]
            rest_shard = remaining_shards[:-1]
            return sum(
                comb(k, number_intersect)
                * comb(self.total_number - k, last_shard - number_intersect)
                * (intersect_cases_recursive(k, rest_shard))
                for k in np.arange(
                    start=max(
                        number_intersect,
                        sum(rest_shard)
                        - self.total_number * (len(remaining_shards) - 2),
                    ),
                    stop=min(rest_shard) + 1 if rest_shard else self.total_number + 1,
                    step=1,
                )
            )

        return intersect_cases_recursive(overall_intersect, self.shard_sizes)

    def intersect_prob(self, overall_intersect: int) -> float:
        """
        Calculate the probability that the intersection of the shards have size overall_intersect.
        """
        return self.intersect_cases(overall_intersect) / prod(
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

    def expected_jaccard(self):
        """
        Compute the expected Jaccard index for two sets of sizes n1 and n2
        using log-space math to avoid overflow.
        """
        n1, n2 = self.shard_sizes
        expected_jaccard_index = 0

        for i in range(max(0, n1 + n2 - self.total_number), min(n1, n2) + 1):
            expected_jaccard_index += (
                i
                / (n1 + n2 - i)
                * comb(n1, i)
                * comb(self.total_number - n1, n2 - i)
                / comb(self.total_number, n2)
            )
        return expected_jaccard_index

    def estimated_jaccard(self):
        """
        Estimate Jaccard index using a simplified formula.
        """
        n1, n2 = self.shard_sizes
        return n1 * n2 / (self.total_number * (n1 + n2) - n1 * n2)


if __name__ == "__main__":
    compute = AnalyticalResult(100, [1, 1])
    union_size = 94
    union_pmf = compute.union_prob(union_size)
    sigma_value = compute.compute_sigma()
    occ_value = compute.occ_value()

    intersect_size = 34
    intersect_pmf = compute.intersect_prob(intersect_size)

    print("sigma =", sigma_value)
    print("occ =", occ_value)
    print(f"probability that union size is {union_size} =", union_pmf)
    print(f"probability that intersect size is {intersect_size} =", intersect_pmf)

    # Calculate Jaccard index for two parties
    if compute.party_number == 2:
        expected_jaccard = compute.expected_jaccard()
        estimated_jaccard = compute.estimated_jaccard()

        print(f"Expected Jaccard index: {expected_jaccard}")
        print(f"Estimated Jaccard index: {estimated_jaccard}")
        print(
            f"Percentage difference: {(expected_jaccard - estimated_jaccard)*100/expected_jaccard}%"
        )
