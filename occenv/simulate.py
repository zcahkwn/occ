import itertools
from joblib import Parallel, delayed
from occenv.env import DataShare


class Simulate:
    def __init__(self, total_number: int, shard_sizes: list[int]):
        self.total_number = total_number
        self.shard_sizes = shard_sizes
        self.m = len(shard_sizes)

    def simulate_overlap_once(self) -> dict[tuple[int, ...], float]:
        """
        Compute the number of intersection for every combination of shards.
        Returns a dictionary where each key is a tuple of shard indices (e.g. (0,1)) and the value is the number of elements in the intersection of those shards.
        """
        new_mpc = DataShare(self.total_number)
        shards = [new_mpc.create_shard(shard_size) for shard_size in self.shard_sizes]
        overlaps = {}
        for k in range(1, self.m + 1):
            for combo in itertools.combinations(range(self.m), k):
                current_intersection = set(shards[combo[0]])
                for i in combo[1:]:
                    current_intersection &= set(shards[i])
                overlaps[combo] = len(current_intersection) / self.total_number
        return overlaps

    def simulate_overlap_repeat(self, repeat: int) -> dict[tuple[int, ...], float]:
        """
        Simulate repeat times and return the results using Joblib for parallel processing.
        """
        results = Parallel(n_jobs=-1)(
            delayed(self.simulate_overlap_once)() for _ in range(repeat)
        )

        aggregate: dict[tuple[int, ...], list[int]] = {}
        for result in results:
            for combo, value in result.items():
                aggregate.setdefault(combo, []).append(value)

        averages = {
            combo: sum(values) / len(values) for combo, values in aggregate.items()
        }
        return averages

    def simulate_union_once(self) -> int:
        new_mpc = DataShare(self.total_number)
        elements_covered = []
        for shard_size in self.shard_sizes:
            elements_covered.extend(new_mpc.create_shard(shard_size))

        return len(set(elements_covered))

    def simulate_union_repeat(self, repeat: int, **kwargs) -> list[int]:
        """
        Simulate repeat times and return the results using Joblib for parallel processing.
        """
        return Parallel(n_jobs=-1)(
            delayed(self.simulate_union_once)(**kwargs) for _ in range(repeat)
        )

    def simulate_intersection_once(self) -> int:
        new_mpc = DataShare(self.total_number)
        shards = [new_mpc.create_shard(shard_size) for shard_size in self.shard_sizes]
        intersection = set(shards[0])
        for shard in shards[1:]:
            intersection &= set(shard)
        return len(intersection)

    def simulate_intersection_repeat(self, repeat: int, **kwargs) -> list[int]:
        """
        Simulate repeat times and return the results using Joblib for parallel processing.
        """
        return Parallel(n_jobs=-1)(
            delayed(self.simulate_intersection_once)(**kwargs) for _ in range(repeat)
        )


if __name__ == "__main__":
    total_number = 10
    shard_sizes = [5, 6, 4]
    simulator = Simulate(total_number, shard_sizes)

    repeat = int(1e5)
    union = simulator.simulate_union_repeat(repeat)
    print("The average union is ", sum(union) / repeat)
    overlap_number = simulator.simulate_overlap_repeat(repeat)

    intersection = simulator.simulate_intersection_repeat(repeat)
    print("The average intersection is ", sum(intersection) / repeat)

    for combo, avg in sorted(overlap_number.items()):
        print(f"Combination {combo}: Average intersection = {avg}")
