from math import comb, prod, ceil, gcd
from typing import Iterable
from functools import lru_cache
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

        def union_cases_recursive(number_covered: int, shard_sizes_m: list[int]) -> int:

            last_shard = shard_sizes_m[-1]
            rest_shard = shard_sizes_m[:-1]
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
        # if not (
        #     max(0, sum(self.shard_sizes) - (self.party_number - 1) * self.total_number)
        #     <= overall_intersect
        #     <= min(self.shard_sizes)
        # ):
        #     return 0.0

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

    def intersect_prob(self, overall_intersect: int) -> float:
        """
        Calculate the probability that the intersection of the shards have size overall_intersect.
        """
        return self.intersect_cases(overall_intersect) / prod(
            comb(self.total_number, n) for n in self.shard_sizes
        )

    def bivariate_cases(self, number_covered: int, number_intersect: int) -> int:
        """
        Exact count of ordered m-tuples with |⋃P_i|=u and |⋂P_i|=v.
        """
        # m=1 case
        # if len(self.shard_sizes) == 1:
        #     return (
        #         comb(self.total_number, n)
        #         if (
        #             number_covered == self.shard_sizes[0]
        #             and number_intersect == self.shard_sizes[0]
        #         )
        #         else 0
        #     )

        # if not (
        #     max(
        #         *self.shard_sizes,
        #         ceil(
        #             (sum(self.shard_sizes) - number_intersect)
        #             / (len(self.shard_sizes) - 1)
        #         ),
        #     )
        #     <= number_covered
        #     <= min(
        #         self.total_number,
        #         sum(self.shard_sizes) - (len(self.shard_sizes) - 1) * number_intersect,
        #     )
        # ):
        #     return 0

        @lru_cache(None)
        def bivariate_cases_recursive(
            u_m: int, v_m: int, shard_sizes_m: tuple[int, ...]
        ) -> int:
            m = len(shard_sizes_m)
            last_shard = shard_sizes_m[-1]
            rest_shard = shard_sizes_m[:-1]
            if m == 1:
                return (
                    comb(self.total_number, last_shard)
                    if (u_m == last_shard and v_m == last_shard)
                    else 0
                )

            v_min = max(v_m, sum(rest_shard) - (m - 2) * self.total_number, 0)
            v_max = min(rest_shard)
            total = 0

            for v_prev in range(v_min, v_max + 1):
                # if m == 2:
                #     u_min = u_max = sum(rest_shard)
                # else:
                u_min = (
                    max(
                        *rest_shard,
                        ceil((sum(rest_shard) - v_prev) / (m - 2)),
                    )
                    if m > 2
                    else max(rest_shard)
                )
                u_max = sum(rest_shard) - (m - 2) * v_prev

                u_min = max(u_min, v_prev, u_m - last_shard, 0)
                u_max = min(u_max, sum(rest_shard), u_m)

                for u_prev in range(u_min, u_max + 1):
                    if not (
                        0 <= u_m - u_prev <= min(last_shard, self.total_number - u_prev)
                    ):
                        continue
                    if not (0 <= last_shard - v_m - u_m + u_prev <= (u_prev - v_prev)):
                        continue

                    total += (
                        comb(v_prev, v_m)
                        * comb(u_prev - v_prev, last_shard + u_prev - u_m - v_m)
                        * comb(self.total_number - u_prev, u_m - u_prev)
                        * bivariate_cases_recursive(u_prev, v_prev, rest_shard)
                    )
            return total

        return bivariate_cases_recursive(
            number_covered, number_intersect, tuple(self.shard_sizes)
        )

    def bivariate_prob(self, number_covered: int, number_intersect: int) -> float:
        """Probability for (|⋃P_i|,|⋂P_i|)=(u,v)."""
        return self.bivariate_cases(number_covered, number_intersect) / prod(
            comb(self.total_number, n) for n in self.shard_sizes
        )

    # jaccard probability for a given numerator and denominator is bivariate_prob(numerator, denominator)
    def jaccard_prob(self, numerator: int, denominator: int) -> float:
        gcd_jaccard = gcd(numerator, denominator)
        simplified = (numerator // gcd_jaccard, denominator // gcd_jaccard)
        possible_numerators = [
            k * simplified[0] for k in range(1, self.party_number + 1)
        ]
        possible_denominators = [
            k * simplified[1] for k in range(1, self.party_number + 1)
        ]
        possible_combinations = list(zip(possible_numerators, possible_denominators))
        jaccard_prob = sum(self.bivariate_prob(n, d) for n, d in possible_combinations)
        return jaccard_prob

    # def jaccard_prob(self, indices: Iterable[int]) -> float:
    #     return self.jaccard(indices) * self.union_prob(sum(self.shard_sizes[i] for i in indices))

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

    def occ_value(self):  # N * OCC value is the expected total intersection
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
    compute = AnalyticalResult(100, [50])
    collusion_probability = compute.union_prob(100)
    union_size = 60
    union_pmf = compute.union_prob(union_size)
    sigma_value = compute.compute_sigma()
    occ_value = compute.occ_value()

    intersect_size = 50
    intersect_pmf = compute.intersect_prob(intersect_size)

    print("sigma =", sigma_value)
    print("expected total intersection =", occ_value)
    print("probability of collusion =", collusion_probability)
    print(f"probability that union size is {union_size} =", union_pmf)
    print(f"probability that intersect size is {intersect_size} =", intersect_pmf)

    # Calculate Jaccard index for two parties
    # if compute.party_number == 2:
    #     expected_jaccard = compute.expected_jaccard()
    #     estimated_jaccard = compute.estimated_jaccard()

    #     print(f"Expected Jaccard index: {expected_jaccard}")
    #     print(f"Estimated Jaccard index: {estimated_jaccard}")
    #     print(
    #         f"Percentage difference: {(expected_jaccard - estimated_jaccard)*100/expected_jaccard}%"
    #     )
    pair = (50, 40)
    bivariate_prob = compute.bivariate_prob(*pair)
    print(f"bivariate probability for {pair} = {bivariate_prob}")
