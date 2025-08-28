"""
Test that the marginal probabilities of the bivariate distribution match the univariate probabilities.
(Purely analytical identity checks, no simulation involved)
"""

import pytest
from occenv.analytical_bivariate import AnalyticalBivariate
from occenv.analytical_univariate import AnalyticalUnivariate
import random


@pytest.mark.parametrize(
    "shard_sizes",
    [
        (50,),
        (50, 40),
        (50, 50, 40),
        (50, 50, 50, 50),
    ],
)
def test_bivariate_marginals_match_univariate(shard_sizes):
    total_number = 200
    analytical = AnalyticalBivariate(total_number, shard_sizes)
    analytical_univariate = AnalyticalUnivariate(total_number, shard_sizes)

    # check a few v values (marginal probability conditioned on union)
    intersection_values = [
        0,
        min(shard_sizes) // 2,
        min(shard_sizes),
        random.randint(0, min(shard_sizes)),
    ]
    for v in intersection_values:
        sum_marginal_prob = sum(
            analytical.bivariate_prob(u, v) for u in range(0, total_number + 1)
        )
        analytical_intersection_prob = analytical_univariate.intersection_prob(v)
        assert analytical_intersection_prob == pytest.approx(
            sum_marginal_prob, abs=1e-10
        )

    # check a few u values (marginal probability conditioned on intersection)
    union_values = [
        max(shard_sizes),
        (max(shard_sizes) + total_number) // 2,
        total_number,
        random.randint(0, max(shard_sizes)),
    ]
    for u in union_values:
        sum_marginal_prob = sum(
            analytical.bivariate_prob(u, v) for v in range(0, min(shard_sizes) + 1)
        )
        analytical_union_prob = analytical_univariate.union_prob(u)
        assert analytical_union_prob == pytest.approx(sum_marginal_prob, abs=1e-10)
