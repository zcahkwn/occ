import numpy as np


class ApproximatedResult:
    def __init__(self, total_number: int, shard_sizes: list[int]):
        self.total_number = total_number
        self.shard_sizes = shard_sizes
        self.party_number = len(shard_sizes)
        self.alpha = [shard_size / self.total_number for shard_size in shard_sizes]

    # --- Set Union approximated results ---

    def union_p_approx(self) -> float:
        return 1 - np.prod(1 - np.array(self.alpha))

    def union_mu_approx(self) -> float:
        return self.total_number * self.union_p_approx()

    def union_var_approx(self) -> float:
        p_union = self.union_p_approx()
        return self.total_number * p_union * (1 - p_union) + self.total_number * (
            self.total_number - 1
        ) * (
            -np.prod(1 - np.array(self.alpha)) ** 2
            + np.prod(
                [
                    (self.total_number - shard)
                    * (self.total_number - shard - 1)
                    / (self.total_number * (self.total_number - 1))
                    for shard in self.shard_sizes
                ]
            )
        )

    def union_sd_approx(self) -> float:
        return np.sqrt(self.union_var_approx())

    # --- Set Intersect approximated results ---

    def intersect_p_approx(self) -> float:
        return np.prod(self.alpha)

    def intersect_mu_approx(self) -> float:
        return self.total_number * self.intersect_p_approx()

    def intersect_var_approx(self) -> float:
        p_intersect = self.intersect_p_approx()
        return self.total_number * p_intersect * (
            1 - p_intersect
        ) + self.total_number * (self.total_number - 1) * (
            np.prod([a**2 + (a**2 - a) / (self.total_number - 1) for a in self.alpha])
            - np.prod(self.alpha) ** 2
        )

    def intersect_sd_approx(self) -> float:
        return np.sqrt(self.intersect_var_approx())

    # --- Jaccard index approximated results ---

    def jaccard_mu_approx(self) -> float:
        return (
            self.intersect_p_approx() / self.union_p_approx()
            if self.union_p_approx() > 0
            else 0
        )
