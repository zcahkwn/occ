"""
Test that the analytical Jaccard index CDF matches the empirical Jaccard index CDF.
"""

import pytest
from occenv.simulate import Simulate
from occenv.analytical_jaccard import AnalyticalJaccard
from occenv.analytical_bivariate import AnalyticalBivariate


@pytest.mark.parametrize(
    "total_number, shard_sizes, thresholds",
    [
        (100, (30, 35), [0.2, 0.3, 0.4, 0.5]),
        (80, (20, 25), [0.25, 0.4, 0.5]),
        (60, (15, 20), [0.2, 0.33, 0.5]),
    ],
)
def test_jaccard_cdf(total_number, shard_sizes, thresholds):
    repeats = int(1e6)
    sim = Simulate(total_number, shard_sizes)
    pmf = sim.simulate_bivariate_repeat(repeat=repeats)  # {(U,V): p}

    # Empirical CDF from simulated PMF
    def jaccard_cdf_emp(t: float) -> float:
        s = 0.0
        for (u, v), p in pmf.items():
            # u>0 in practice; include v==0 (J=0) and v/u < t
            if v == 0 or (u > 0 and v / u < t):
                s += p
        return s

    # Analytical CDF using analytical result
    ar = AnalyticalBivariate(total_number, shard_sizes)
    jaccard_ana = AnalyticalJaccard(total_number, shard_sizes, ar)

    for t in thresholds:
        jaccard_cdf_emp_t = jaccard_cdf_emp(t)
        jaccard_cdf_ana_t = jaccard_ana.jaccard_cdf_analytical(t)
        assert jaccard_cdf_ana_t == pytest.approx(jaccard_cdf_emp_t, abs=0.02)

    # Compare the mean of the empirical and analytical Jaccard indices
    jaccard_mu_emp = sum(v / u * p for (u, v), p in pmf.items())
    jaccard_mu_ana = jaccard_ana.jaccard_mu()
    assert jaccard_mu_ana == pytest.approx(jaccard_mu_emp, abs=0.02)
