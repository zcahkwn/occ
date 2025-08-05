import pytest
from occenv.simulate import Simulate
from occenv.analytical import AnalyticalResult


@pytest.mark.parametrize(
    "shard_sizes",
    [[7], [10], [6, 3], [5, 6], [10, 6], [3, 2, 4], [7, 6, 9], [3, 5, 7, 3]],
)
def test_collusion_formula(shard_sizes):
    total_number = 10
    theoretical = AnalyticalResult(total_number, shard_sizes).union_prob(total_number)
    simulation = Simulate(total_number, shard_sizes)

    repeat = int(1e6)
    simulation_results = simulation.simulate_union_repeat(repeat=repeat)
    simulation_probability = (
        sum([w == total_number for w in simulation_results]) / repeat
    )

    assert theoretical == pytest.approx(simulation_probability, abs=0.01)
