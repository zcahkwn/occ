from joblib import Parallel, delayed
from occenv.env import DataShare


def simulate_once(total_number: int, shard_sizes: list[int]) -> int:
    new_mpc = DataShare(total_number)
    elements_covered = []
    for shard_size in shard_sizes:
        elements_covered.extend(new_mpc.create_shard(shard_size))

    return len(set(elements_covered))


def simulate_repeat(repeat: int, **kwargs) -> list[int]:
    """
    Simulate repeat times and return the results using Joblib for parallel processing.
    """
    return Parallel(n_jobs=-1)(delayed(simulate_once)(**kwargs) for _ in range(repeat))


if __name__ == "__main__":
    n = int(5e7)
    # Example usage
    results = simulate_repeat(repeat=n, total_number=10, shard_sizes=[7, 5, 3])
    # print(sum(results) / n)
