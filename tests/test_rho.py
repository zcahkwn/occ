import itertools
import pytest
from occenv.simulate import Simulate
from occenv.analytical import AnalyticalResult


@pytest.mark.parametrize(
    "shard_sizes",
    [[7], [10], [6, 3], [5, 6], [10, 6], [3, 2, 4], [7, 6, 9], [3, 5, 7, 3]],
)
def test_overlap_simulation(shard_sizes):
    total_number = 10
    shard_sizes = list(shard_sizes)

    simulation = Simulate(total_number, shard_sizes)
    analytical = AnalyticalResult(total_number, shard_sizes)

    repeat = int(1e5)
    simulation_results = simulation.simulate_overlap_repeat(repeat)

    m = len(shard_sizes)
    # For each combination of shards, compare simulated value with analytical value.
    for k in range(1, m + 1):
        for combo in itertools.combinations(range(m), k):
            analytic_val = analytical.rho(combo)
            simulation_val = simulation_results[combo]
            assert simulation_val == pytest.approx(analytic_val, abs=0.01), (
                f"Mismatch for combination {combo}: "
                f"simulated={simulation_val}, analytical={analytic_val}"
            )
