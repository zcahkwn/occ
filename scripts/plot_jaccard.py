from math import comb, prod, ceil, gcd, sqrt, pi, exp
from functools import lru_cache
from fractions import Fraction
from occenv.analytical import AnalyticalResult
import numpy as np
import matplotlib.pyplot as plt

N = 200
shard_sizes = [150, 160, 170]

analytical = AnalyticalResult(N, shard_sizes)

# build Jaccard index PMF
ratios = set()
for v in range(0, min(shard_sizes) + 1):
    for u in range(max(v, 1), N + 1):
        ratios.add(Fraction(v, u))
ratios = sorted(ratios, key=float)

jaccard_list, prob_jaccard_list = [], []
for ratio in ratios:
    prob_jaccard = analytical.jaccard_prob(ratio.numerator, ratio.denominator)
    if prob_jaccard > 0:
        jaccard_list.append(float(ratio))
        prob_jaccard_list.append(prob_jaccard)

# mean & std of the distribution
xs_arr = np.array(jaccard_list)
ps_arr = np.array(prob_jaccard_list)
mu = float(np.sum(xs_arr * ps_arr))
sigma = float(sqrt(np.sum(((xs_arr - mu) ** 2) * ps_arr)))

# Approximated expected Jaccard index
intersect_expect = np.prod([shard_size / N for shard_size in shard_sizes])
union_expect = 1 - np.prod([1 - shard_size / N for shard_size in shard_sizes])
jaccard_approx = intersect_expect / union_expect


"""
Plot the Jaccard index PMF
"""

# Plot histogram
plt.figure(figsize=(9, 4))
bins = np.linspace(
    0, max(jaccard_list) + 1e-6, 100
)  # the smaller the bin size, the thicker the bars
plt.hist(
    jaccard_list,
    bins=bins,
    weights=prob_jaccard_list,
    density=True,
    alpha=0.4,
    label="empirical pmf",
)

# normal pdf overlay
x_grid = np.linspace(0, max(jaccard_list), 400)
norm_pdf = (1 / (sigma * sqrt(2 * pi))) * np.exp(-((x_grid - mu) ** 2) / (2 * sigma**2))
plt.plot(x_grid, norm_pdf, label=f"normal pdf with mean={mu:.2f} and std={sigma:.2f}")

# vertical reference lines
plt.axvline(
    mu, linestyle="-", color="b", label=f"mean of empirical pmf={mu:.2f}"
)  # mean of empirical pmf
plt.axvline(
    jaccard_approx,
    linestyle="--",
    color="r",
    label=f"approx of expected Jaccard index={jaccard_approx:.2f}",
)  # approximated expected Jaccard index
plt.legend()
plt.xlabel("Jaccard index")
plt.ylabel("Density")
plt.title(f"Jaccard index – Histogram vs Normal fit (N={N}, shards={shard_sizes})")
plt.tight_layout()


# Stem plot
plt.figure(figsize=(9, 4))
plt.stem(jaccard_list, prob_jaccard_list)
# draw a line to show intersection/union value in the plot
plt.axvline(
    x=intersect_expect / union_expect,
    color="red",
    linestyle="--",
    label=f"approx of expected Jaccard index={jaccard_approx:.2f}",
)
plt.legend()
plt.xlabel("Jaccard index")
plt.ylabel("Probability")
plt.title(f"Jaccard index - Stem plot (N={N}, shards={shard_sizes})")
plt.tight_layout()
