from math import comb, prod
import numpy as np

class Collusion: 
    def __init__(self, total_number: int, shard_sizes: list[int]):
        self.total_number = total_number
        self.shard_sizes = shard_sizes

    def collude_cases(self, total_number: int, shard_sizes: list[int]) -> int:
        """
        Calculate the number of cases of colluding to reconstruct the secret set.
        """

        last_shard = shard_sizes[-1]
        rest_shard = shard_sizes[:-1]
        return sum(
            comb(k, k + last_shard - total_number)
            * comb(total_number, k)
            * (self.collude_cases(k, rest_shard) if rest_shard else 1)
            for k in np.arange(
                start=max(rest_shard + [total_number - last_shard]),
                stop=min(sum(rest_shard), total_number) + 1,
                step=1,
            )
        )

    def collude_prob(self) -> float:
        """
        Calculate the probability of colluding to reconstruct the secret set.
        """
        assert (
            sum(self.shard_sizes) >= self.total_number
        ), "Sum of shard sizes must be greater than total number."

        return self.collude_cases(self.total_number, self.shard_sizes) / prod(
            comb(self.total_number, n) for n in self.shard_sizes
        )


if __name__ == "__main__":
    parties_list = Collusion(10, [3, 4, 3, 5])
    probability = parties_list.collude_prob()
