import pytest
from occenv.collusion_simulate import simulate_repeat
from occenv.analytical import AnalyticalResult


@pytest.mark.parametrize(
    "shard_sizes",
    [[7], [10], [6, 3], [5, 6], [10, 6], [3, 2, 4], [7, 6, 9], [3, 5, 7, 3]],
)
def test_collusion_formula(shard_sizes):
    total_number = 10

    # result from analytical.py
    theoretical = AnalyticalResult(total_number, shard_sizes).collude_prob(total_number)

    # simulation result
    repeat = int(1e6)
    simulation_results = simulate_repeat(
        repeat=repeat, total_number=total_number, shard_sizes=shard_sizes
    )
    simulation_probability = (
        sum([w == total_number for w in simulation_results]) / repeat
    )

    assert theoretical == pytest.approx(simulation_probability, abs=0.01)
