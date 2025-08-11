"""
Approximated results for the univariate and bivariate distributions using CLT.
"""

import numpy as np
import itertools
from math import prod
from typing import Iterable
import itertools


class ApproximatedResult:
    def __init__(self, total_number: int, shard_sizes: list[int]):
        self.total_number = total_number
        self.shard_sizes = shard_sizes
        self.party_number = len(shard_sizes)
        self.alpha = np.array(shard_sizes) / self.total_number

    # --- Set Union approximated results ---

    def union_p_approx(self) -> float:
        return 1 - np.prod(1 - self.alpha)

    def union_mu_approx(self) -> float:
        return self.total_number * self.union_p_approx()

    def union_var_approx(self) -> float:
        p_union = self.union_p_approx()
        return self.total_number * p_union * (1 - p_union) + self.total_number * (
            self.total_number - 1
        ) * (
            -np.prod(1 - self.alpha) ** 2
            + np.prod(
                [
                    (self.total_number - shard)
                    * (self.total_number - shard - 1)
                    / (self.total_number * (self.total_number - 1))
                    for shard in self.shard_sizes
                ]
            )
        )

    def union_sd_approx(self) -> float:
        return np.sqrt(self.union_var_approx())

    def sigma_value(
        self,
    ):  # N * sigma value is the expected total union, using inclusion-exclusion principle
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

    # --- Set Intersect approximated results ---

    def intersection_p_approx(self) -> float:
        return np.prod(self.alpha)

    def intersection_mu_approx(self) -> float:
        return self.total_number * self.intersection_p_approx()

    def intersection_var_approx(self) -> float:
        p_intersect = self.intersection_p_approx()
        return self.total_number * p_intersect * (
            1 - p_intersect
        ) + self.total_number * (self.total_number - 1) * (
            np.prod(
                self.alpha**2 + (self.alpha**2 - self.alpha) / (self.total_number - 1)
            )
            - np.prod(self.alpha) ** 2
        )

    def intersection_sd_approx(self) -> float:
        return np.sqrt(self.intersection_var_approx())

    def rho(self, indices: Iterable[int]) -> float:
        product = prod(self.shard_sizes[i] for i in indices)
        k = len(indices)
        result = product / (self.total_number**k)
        return result

    def occ_value(self):  # N * OCC value is the expected total intersection
        return self.rho(list(range(self.party_number)))

    # --- Bivariate approximated results ---

    # --- Jaccard index approximated results ---

    def jaccard_mu_approx(self) -> float:
        return (
            self.intersection_p_approx() / self.union_p_approx()
            if self.union_p_approx() > 0
            else 0
        )
