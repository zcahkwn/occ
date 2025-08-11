"""
Test the approximated results vs analytical results
"""

import pytest
from occenv.approximated import ApproximatedResult
from occenv.analytical_univariate import AnalyticalUnivariate
from occenv.analytical_bivariate import AnalyticalBivariate


@pytest.mark.parametrize(
    "total_number, shard_sizes",
    [
        (100, [10, 20, 30]),
        (100, [10, 20, 30, 40]),
        (100, [40, 50]),
    ],
)
def test_approx(total_number, shard_sizes):

    univ = AnalyticalUnivariate(total_number, shard_sizes)
    biv = AnalyticalBivariate(total_number, shard_sizes)
    approx = ApproximatedResult(total_number, shard_sizes)

    # For union distribution, test mean, variance
    assert approx.union_mu_approx() == pytest.approx(univ.union_mu(), abs=0.01)
    assert approx.union_var_approx() == pytest.approx(univ.union_var(), abs=0.01)

    # For intersection distribution, test mean, variance
    assert approx.intersection_mu_approx() == pytest.approx(
        univ.intersection_mu(), abs=0.01
    )
    assert approx.intersection_var_approx() == pytest.approx(
        univ.intersection_var(), abs=0.01
    )

    # For bivariate distribution, test mean vector, and variance matrix
    assert approx.bivariate_mu_approx() == pytest.approx(biv.bivariate_mu(), abs=0.01)
    assert approx.bivariate_matrix_approx() == pytest.approx(
        biv.bivariate_matrix(), abs=0.01
    )

    # For jaccard index, test mean
    assert approx.jaccard_mu_approx() == pytest.approx(biv.jaccard_mu(), abs=0.01)
