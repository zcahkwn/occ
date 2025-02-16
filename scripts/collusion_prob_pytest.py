import pytest
from scripts.collusion_prob import collusion_m
from occenv.simulate import simulate_repeat

def test_collusion_formula():
    total_number = 10
    shard_sizes = [4, 4, 2, 3]
    
    # result from collusion_prob.py
    theoretical = collusion_m(total_number, shard_sizes)
    
    # simulation result
    repeat = int(1e7) 
    simulation_results = simulate_repeat(repeat=repeat, total_number=total_number, shard_sizes=shard_sizes)
    simulation_probability = sum(simulation_results) / repeat
    
    assert theoretical == pytest.approx(simulation_probability, abs=0.01)

if __name__ == "__main__":
    pytest.main([__file__])