from joblib import Parallel, delayed
from occenv.env import DataShare
from collections import Counter


class Simulate:
    def __init__(self, total_number: int, shard_sizes: list[int]):
        self.total_number = total_number
        self.shard_sizes = shard_sizes
        self.m = len(shard_sizes)

    def simulate_bivariate_once(self) -> tuple[int, int]:
        """
        Single run → (U, V) for ALL shards:
        U = union size, V = intersection size.
        """
        if self.m == 0:
            return (0, 0)

        new_mpc = DataShare(self.total_number)
        shards = [set(new_mpc.create_shard(s)) for s in self.shard_sizes]
        U = len(set().union(*shards))
        V = len(set.intersection(*shards))
        return U, V

    def simulate_bivariate_repeat(
        self, repeat: int, block: int = 10_000
    ) -> dict[tuple[int, int], float]:
        """
        Bivariate PMF over (U, V): {(U, V): probability}.
        Runs in blocks to avoid storing all raw samples.
        """

        def run_block(n: int) -> Counter:
            c = Counter()
            for _ in range(n):
                c[self.simulate_bivariate_once()] += 1
            return c

        q, r = divmod(repeat, block)
        blocks = [block] * q + ([r] if r else [])
        parts = Parallel(n_jobs=-1)(delayed(run_block)(n) for n in blocks)

        total = Counter()
        for part in parts:
            total.update(part)

        return {uv: cnt / repeat for uv, cnt in total.items()}

    def simulate_union_once(self) -> int:
        U, _ = self.simulate_bivariate_once()
        return U

    def simulate_union_repeat(self, repeat: int) -> list[int]:
        return Parallel(n_jobs=-1)(
            delayed(self.simulate_union_once)() for _ in range(repeat)
        )

    def simulate_intersection_once(self) -> int:
        _, V = self.simulate_bivariate_once()
        return V

    def simulate_intersection_repeat(self, repeat: int) -> list[int]:
        return Parallel(n_jobs=-1)(
            delayed(self.simulate_intersection_once)() for _ in range(repeat)
        )


if __name__ == "__main__":
    total_number = 100
    shard_sizes = [50, 63, 75]

    simulator = Simulate(total_number, shard_sizes)
    repeat = int(1e6)

    union = simulator.simulate_union_repeat(repeat)
    print("The average union is ", sum(union) / repeat)

    intersection = simulator.simulate_intersection_repeat(repeat)
    print("The average intersection is ", sum(intersection) / repeat)
