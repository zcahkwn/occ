"""
Test that the simulated univariate probabilities match the analytical univariate probabilities.
"""

"""
Test that simulated union/intersection PMFs and means match analytical results.
"""

import pytest
from collections import Counter
from occenv.simulate import Simulate
from occenv.analytical_univariate import AnalyticalUnivariate
from occenv.utils import mu_calculation


@pytest.mark.parametrize(
    "shard_sizes",
    [
        # [7, 9],
        # [7],
        # [10],
        # [6, 3],
        # [5, 6],
        # [10, 6],
        # [3, 2, 4],
        [7, 6, 9],
        [3, 5, 7, 3],
    ],
)
def test_univariate_pmf_and_mean(shard_sizes):
    total_number = 10
    repeats = int(1e6)

    sim = Simulate(total_number, shard_sizes)
    ana = AnalyticalUnivariate(total_number, shard_sizes)

    # --- Empirical PMFs from samples ---
    union_samples = sim.simulate_union_repeat(repeat=repeats)
    inter_samples = sim.simulate_intersection_repeat(repeat=repeats)

    cU = Counter(union_samples)
    cV = Counter(inter_samples)

    x_u = list(range(0, total_number + 1))
    x_v = list(range(0, min(shard_sizes) + 1))

    pmf_u_emp = [cU.get(u, 0) / repeats for u in x_u]
    pmf_v_emp = [cV.get(v, 0) / repeats for v in x_v]

    # --- Analytical PMFs ---
    pmf_u_ana = [ana.union_prob(u) for u in x_u]
    pmf_v_ana = [ana.intersection_prob(v) for v in x_v]

    # --- Compare PMFs element-wise ---
    for p_emp, p_ana in zip(pmf_u_emp, pmf_u_ana):
        assert p_ana == pytest.approx(p_emp, abs=0.01)
    for p_emp, p_ana in zip(pmf_v_emp, pmf_v_ana):
        assert p_ana == pytest.approx(p_emp, abs=0.01)

    # --- Sanity checks ---
    assert sum(pmf_u_emp) == pytest.approx(1.0, abs=1e-3)
    assert sum(pmf_v_emp) == pytest.approx(1.0, abs=1e-3)
    assert sum(pmf_u_ana) == pytest.approx(1.0, abs=1e-9)
    assert sum(pmf_v_ana) == pytest.approx(1.0, abs=1e-9)

    # --- Compare means ---
    E_union_emp = mu_calculation(x_u, pmf_u_emp)
    E_union_ana = ana.union_mu()
    E_inter_emp = mu_calculation(x_v, pmf_v_emp)
    E_inter_ana = ana.intersection_mu()

    assert E_union_ana == pytest.approx(E_union_emp, abs=0.01)
    assert E_inter_ana == pytest.approx(E_inter_emp, abs=0.01)
