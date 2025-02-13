from joblib import Parallel, delayed

from occenv.env import DataShare


def simulate_once(
    total_number: int, shard_sizes: list[int]
    
) -> bool:
    new_mpc = DataShare(total_number)
    elements_covered = []
    for shard_size in shard_sizes:
        elements_covered.extend(new_mpc.create_shard(shard_size))
    
    return set(elements_covered) == set(new_mpc.secret_set)


def simulate_repeat(repeat: int, **kwargs) -> list[tuple[float, float | None]]:
    """
    Simulate repeat times and return the results using Joblib for parallel processing.
    """
    results = Parallel(n_jobs=-1)(
        delayed(simulate_once)(**kwargs) for _ in range(repeat)
    )
    return results  # type: ignore


if __name__ == "__main__":
    # Example usage
    results = simulate_repeat(repeat=500, total_number=100, shard_sizes=[30, 90, 99])
    print(results)
    print(sum(results) / len(results))