import itertools
from joblib import Parallel, delayed
from occenv.env import DataShare


class Simulate:
    def __init__(self, total_number: int, shard_sizes: list[int]):
        self.total_number = total_number
        self.shard_sizes = shard_sizes
        self.m = len(shard_sizes)

    def simulate_combo_once(self) -> dict[tuple[int, ...], float]:
        """
        Compute the number of intersection for every combination of shards.
        Returns a dictionary where each key is a tuple of shard indices (e.g. (0,1)) and the value is the number of elements in the intersection of those shards.
        """
        new_mpc = DataShare(self.total_number)
        shards = [new_mpc.create_shard(shard_size) for shard_size in self.shard_sizes]
        overlaps = {}
        union = {}
        jaccard = {}
        for k in range(1, self.m + 1):
            for combo in itertools.combinations(range(self.m), k):
                current_intersection = set(shards[combo[0]])
                current_union = set(shards[combo[0]])
                for i in combo[1:]:
                    current_intersection &= set(shards[i])
                    current_union |= set(shards[i])
                overlaps[combo] = len(current_intersection) / self.total_number
                union[combo] = len(current_union) / self.total_number
                jaccard[combo] = (
                    len(current_intersection) / len(current_union)
                    if len(current_union) > 0
                    else 0.0
                )
        return overlaps, union, jaccard

    def simulate_combo_repeat(self, repeat: int) -> dict[tuple[int, ...], float]:
        """
        Simulate repeat times and return the results using Joblib for parallel processing.
        """
        results = Parallel(n_jobs=-1)(
            delayed(self.simulate_combo_once)() for _ in range(repeat)
        )

        aggregate_overlaps: dict[tuple[int, ...], list[int]] = {}
        aggregate_union: dict[tuple[int, ...], list[int]] = {}
        aggregate_jaccard: dict[tuple[int, ...], list[int]] = {}
        for overlaps, union, jaccard in results:
            for combo, value in overlaps.items():
                aggregate_overlaps.setdefault(combo, []).append(value)
            for combo, value in union.items():
                aggregate_union.setdefault(combo, []).append(value)
            for combo, value in jaccard.items():
                aggregate_jaccard.setdefault(combo, []).append(value)
        avg_intersection = {
            combo: sum(values) / len(values)
            for combo, values in aggregate_overlaps.items()
        }
        avg_union = {
            combo: sum(values) / len(values)
            for combo, values in aggregate_union.items()
        }
        avg_jaccard = {
            combo: sum(values) / len(values)
            for combo, values in aggregate_jaccard.items()
        }
        return avg_intersection, avg_union, avg_jaccard

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
    total_number = 100

    shard_sizes = [50, 63, 75]
    simulator = Simulate(total_number, shard_sizes)

    repeat = int(1e5)

    union = simulator.simulate_union_repeat(repeat)
    print("The average union is ", sum(union) / repeat)

    intersection = simulator.simulate_intersection_repeat(repeat)
    print("The average intersection is ", sum(intersection) / repeat)

    avg_intersection, avg_union, avg_jaccard = simulator.simulate_combo_repeat(repeat)

    common_combos = set(avg_intersection) & set(avg_union) & set(avg_jaccard)
    for combo in sorted(common_combos):
        print(
            f"Combination {combo}: Average intersection = {avg_intersection[combo]}, Average union = {avg_union[combo]}, Average Jaccard index = {avg_jaccard[combo]}, Average estimated Jaccard index = {avg_intersection[combo] / avg_union[combo]}"
        )
