from math import comb, prod

import numpy as np


def collude_cases(total_number: int, shard_sizes: list[int]) -> int:
    """
    Calculate the number of cases of colluding to reconstruct the secret set.
    """

    if len(shard_sizes) == 1:
        return 1
    else:
        last_shard = shard_sizes[-1]
        rest_shard = shard_sizes[:-1]
        return sum(
            comb(k, k + last_shard - total_number)
            * comb(total_number, k)
            * collude_cases(k, rest_shard)
            for k in np.arange(
                start=max(rest_shard[-1], total_number - last_shard),
                stop=min(sum(rest_shard), total_number) + 1,
                step=1,
            )
        )


def collude_prob(total_number: int, shard_sizes: list[int]) -> float:
    """
    Calculate the probability of colluding to reconstruct the secret set.
    """
    assert (
        sum(shard_sizes) >= total_number
    ), "Sum of shard sizes must be greater than total number."
    shard_sizes.sort()

    return collude_cases(total_number, shard_sizes) / prod(
        comb(total_number, n) for n in shard_sizes
    )


if __name__ == "__main__":
    collude_cases(10, [6, 7])
    collude_prob(10, [6, 5, 4, 3])
