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
                    start=max(rest_shard + [number_covered - last_shard, 0]),
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
        # if len(self.shard_sizes) == 1:
        #     if not (
        #         number_covered == self.shard_sizes[0]
        #         and number_intersect == self.shard_sizes[0]
        #     ):
        #         return 0
        # else:
        #     if not (
        #         max(
        #             *self.shard_sizes,
        #             ceil(
        #                 (sum(self.shard_sizes) - number_intersect)
        #                 / (len(self.shard_sizes) - 1)
        #             ),
        #         )
        #         <= number_covered
        #         <= min(
        #             self.total_number,
        #             sum(self.shard_sizes)
        #             - (len(self.shard_sizes) - 1) * number_intersect,
        #         )
        #     ):
        #         return 0

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
                u_min = (
                    max(
                        *rest_shard,
                        ceil((sum(rest_shard) - v_prev) / (m - 2)),
                    )
                    if m > 2
                    else max(rest_shard)
                )
                u_max = sum(rest_shard) - (m - 2) * v_prev

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

    def jaccard_prob(self, numerator: int, denominator: int) -> float:
        """
        Probability that the Jaccard index equals intersection/union.
        """
        if not (0 < numerator <= denominator):
            return 0.0

        g = gcd(numerator, denominator)
        a, b = numerator // g, denominator // g  # reduced ratio v/u

        prob = 0.0
        k = 1
        while True:
            v = k * a  # candidate intersection
            u = k * b  # candidate union

            # stop if the pair is impossible
            if u > self.total_number or v > min(self.shard_sizes):
                break

            prob += self.bivariate_prob(u, v)  # (union, intersection)
            k += 1

        return prob

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
    N = 100
    shard_sizes = [30, 40]
    compute = AnalyticalResult(N, shard_sizes)
    collusion_probability = compute.union_prob(N)

    sigma_value = compute.compute_sigma()
    occ_value = compute.occ_value()

    print("sigma =", sigma_value)
    print("Expected total intersection =", occ_value)
    print("Probability of collusion =", collusion_probability)

    union_test = 60
    intersect_test = 20
    union_pmf = compute.union_prob(union_test)
    intersect_pmf = compute.intersect_prob(intersect_test)

    print(
        f"\n------Test bivariate probability-----\nWhen the union size is {union_test} and the intersect size is {intersect_test}, the following results are obtained: \n"
    )
    bivariate_prob = compute.bivariate_prob(union_test, intersect_test)
    print(f"bivariate probability for {union_test, intersect_test} = {bivariate_prob}")

    # Check whether marginal probability for bivariate probability conditional on union size adds up to intersect_pmf when intersect_size is fixed
    marginal_prob_intersect = sum(
        compute.bivariate_prob(union_size, intersect_test)
        for union_size in range(1, N + 1)
    )
    print(
        f"When intersect_size = {intersect_test}, sum of bivariate probability conditional on union size =",
        marginal_prob_intersect,
    )
    print(f"Probability that intersect size is {intersect_test} =", intersect_pmf)
    if intersect_pmf - marginal_prob_intersect < 1e-6:
        print("=> The marginal probability conditional on union size is correct\n")
    else:
        print("=> The marginal probability conditional on union size is not correct\n")

    # Check whether marginal probability for bivariate probability conditional on intersect size adds up to union_pmf when union_size is fixed
    marginal_prob_union = sum(
        compute.bivariate_prob(union_test, intersect_size)
        for intersect_size in range(1, N + 1)
    )
    print(
        f"When union_size = {union_test}, sum of bivariate probability conditional on intersect size =",
        marginal_prob_union,
    )
    print(f"Probability that union size is {union_test} =", union_pmf)
    if union_pmf - marginal_prob_union < 1e-6:
        print("=> The marginal probability conditional on intersect size is correct\n")
    else:
        print(
            "=> The marginal probability conditional on intersect size is not correct\n"
        )

    # Calculate Jaccard index
    pair = (58, 12)
    jaccard_prob = compute.jaccard_prob(6, 29)
    print(
        f"------Test Jaccard index------\nJaccard probability for {6, 29} = {jaccard_prob}"
    )
