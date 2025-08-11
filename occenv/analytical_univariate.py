from math import comb, prod
import numpy as np
from occenv.utils import mu_calculation


class AnalyticalUnivariate:
    def __init__(self, total_number: int, shard_sizes: list[int]):
        self.total_number = total_number
        self.shard_sizes = shard_sizes
        self.party_number = len(shard_sizes)

    # ---- Analytical result for union pmf ----
    def union_cases(self, number_covered) -> int:
        """
        Calculate the number of cases when the union of the shards have size number_covered.
        """
        if sum(self.shard_sizes) < number_covered:
            return 0

        def union_cases_recursive(number_covered: int, shard_sizes_m: list[int]) -> int:

            last_shard = shard_sizes_m[-1]
            rest_shard = shard_sizes_m[:-1]
            return sum(
                comb(k, k + last_shard - number_covered)
                * comb(number_covered, k)
                * (union_cases_recursive(k, rest_shard) if rest_shard else 1)
                for k in np.arange(
                    start=max(rest_shard + [number_covered - last_shard, 0]),
                    stop=min(sum(rest_shard), number_covered) + 1,
                    step=1,
                )
            )

        return comb(self.total_number, number_covered) * union_cases_recursive(
            number_covered, self.shard_sizes
        )

    def union_prob(self, number_covered: int) -> float:
        return self.union_cases(number_covered) / prod(
            comb(self.total_number, n) for n in self.shard_sizes
        )

    def union_mu(self) -> float:
        return mu_calculation(
            range(self.total_number + 1),
            [self.union_prob(u) for u in range(self.total_number + 1)],
        )

    # ---- Analytical result for intersect pmf ----
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
            number_intersect: int, shard_sizes_m: list[int]
        ) -> int:
            if not shard_sizes_m:
                # base case: intersection of zero parties is N
                return 1 if number_intersect == self.total_number else 0

            last_shard = shard_sizes_m[-1]
            rest_shard = shard_sizes_m[:-1]
            return sum(
                comb(k, number_intersect)
                * comb(self.total_number - k, last_shard - number_intersect)
                * (intersect_cases_recursive(k, rest_shard))
                for k in np.arange(
                    start=max(
                        number_intersect,
                        sum(rest_shard) - self.total_number * (len(shard_sizes_m) - 2),
                    ),
                    stop=min(rest_shard) + 1 if rest_shard else self.total_number + 1,
                    step=1,
                )
            )

        return intersect_cases_recursive(overall_intersect, self.shard_sizes)

    def intersection_prob(self, overall_intersect: int) -> float:
        return self.intersect_cases(overall_intersect) / prod(
            comb(self.total_number, n) for n in self.shard_sizes
        )

    def intersection_mu(self) -> float:
        return mu_calculation(
            range(self.total_number + 1),
            [self.intersection_prob(v) for v in range(self.total_number + 1)],
        )
