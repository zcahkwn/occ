import math
from math import comb, prod, ceil, gcd
from typing import Iterable
from functools import lru_cache
import numpy as np
import itertools


class AnalyticalBivariate:
    def __init__(self, total_number: int, shard_sizes: list[int]):
        self.total_number = total_number
        self.shard_sizes = shard_sizes
        self.party_number = len(shard_sizes)

    # ---- Analytical result for bivariate pmf ----
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
        return self.bivariate_cases(number_covered, number_intersect) / prod(
            comb(self.total_number, n) for n in self.shard_sizes
        )

    def bivariate_mu(self) -> np.ndarray:
        U = range(0, self.total_number + 1)
        V = range(0, min(self.shard_sizes) + 1)

        EU = sum(u * self.bivariate_prob(u, v) for u in U for v in V)
        EV = sum(v * self.bivariate_prob(u, v) for u in U for v in V)
        return np.array([EU, EV])

    def bivariate_var(self) -> np.ndarray:
        U = range(max(self.shard_sizes), self.total_number + 1)
        V = range(0, min(self.shard_sizes) + 1)

        EU = self.bivariate_mu()[0]
        EV = self.bivariate_mu()[1]
        EU2 = sum(
            (u * u) * self.bivariate_prob(u, v) for u, v in itertools.product(U, V)
        )
        EV2 = sum(
            (v * v) * self.bivariate_prob(u, v) for u, v in itertools.product(U, V)
        )
        return np.array([EU2 - EU * EU, EV2 - EV * EV])

    def bivariate_cov(self) -> float:
        U = range(0, self.total_number + 1)
        V = range(0, min(self.shard_sizes) + 1)

        EU = self.bivariate_mu()[0]
        EV = self.bivariate_mu()[1]
        EUV = sum((u * v) * self.bivariate_prob(u, v) for u in U for v in V)
        return EUV - EU * EV

    def bivariate_matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.bivariate_var()[0], self.bivariate_cov()],
                [self.bivariate_cov(), self.bivariate_var()[1]],
            ]
        )

    # ---- Analytical result for jaccard index pmf ----
    def jaccard_prob(self, numerator: int, denominator: int) -> float:
        """
        Probability that the Jaccard index equals numerator/denominator.
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

    def jaccard_mu(self) -> float:
        return sum(
            v / u * self.bivariate_prob(u, v)
            for u, v in itertools.product(
                range(1, self.total_number + 1), range(0, min(self.shard_sizes) + 1)
            )
        )

    def jaccard_cdf_analytical(self, t: float) -> float:
        """
        Compute P(J < t) analytically by:
        - adding P(J=0) = sum_u P(U=u, V=0)
        - plus sum of P(J = a/b) over all reduced ratios a/b < t
        """

        # P(J=0) when intersection is 0
        p_zero = sum(self.bivariate_prob(u, 0) for u in range(1, self.total_number + 1))

        total = p_zero
        # Sum over all reduced fractions v/u with v/u < t
        for u in range(1, self.total_number + 1):
            # v must be <= min(u, nmin) to be feasible
            upper_v = min([u] + self.shard_sizes)
            for v in range(1, upper_v + 1):
                if math.gcd(v, u) != 1:
                    continue
                if v / u >= t:
                    continue
                total += self.jaccard_prob(v, u)

        return total


if __name__ == "__main__":
    ana = AnalyticalBivariate(200, [150, 140, 160])
    print(ana.bivariate_mu())
    print(ana.bivariate_matrix())
