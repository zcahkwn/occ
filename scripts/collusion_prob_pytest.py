import pytest
from occenv.simulate import simulate_repeat
from occenv.analytical import Collusion

def test_collusion_formula():
    total_number = 10
    shard_sizes = [6, 5, 3, 9]
    
    # result from analytical.py
    theoretical = Collusion(total_number, shard_sizes)
    
    # simulation result
    repeat = int(1e7) 
    simulation_results = simulate_repeat(repeat=repeat, total_number=total_number, shard_sizes=shard_sizes)
    simulation_probability = sum(simulation_results) / repeat
    
    assert theoretical == pytest.approx(simulation_probability, abs=0.01)

if __name__ == "__main__":
    pytest.main([__file__])