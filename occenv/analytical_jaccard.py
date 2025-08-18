from occenv.analytical_bivariate import AnalyticalBivariate
from math import gcd
import math
import itertools


class AnalyticalJaccard:
    def __init__(
        self, total_number: int, shard_sizes: list[int], ar: AnalyticalBivariate
    ):
        self.total_number = total_number
        self.shard_sizes = shard_sizes
        self.ar = ar

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

            prob += self.ar.bivariate_prob(u, v)  # (union, intersection)
            k += 1

        return prob

    def jaccard_mu(self) -> float:
        return sum(
            v / u * self.ar.bivariate_prob(u, v)
            for u, v in itertools.product(
                range(1, self.total_number + 1), range(0, min(self.shard_sizes) + 1)
            )
        )

    def jaccard_var(self) -> float:
        mu = self.jaccard_mu()
        return sum(
            (v / u - mu) ** 2 * self.ar.bivariate_prob(u, v)
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
        p_zero = sum(
            self.ar.bivariate_prob(u, 0) for u in range(1, self.total_number + 1)
        )

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
