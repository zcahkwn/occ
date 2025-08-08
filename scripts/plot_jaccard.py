"""
This script plots the Jaccard index PMF for a given N and shard sizes.
It also calculates the MAE and SSE of the empirical PMF and the approximated normal distribution.
"""

from fractions import Fraction
from occenv.analytical import AnalyticalResult
from occenv.approximated import ApproximatedResult
from occenv.utils import (
    mu_calculation,
    sigma_calculation,
    mae_calculation,
    sse_calculation,
    discretize_normal_pmf,
)
from occenv.plotting_2d import plot_hist_with_normal, plot_stem_pmf

N = 300
shard_sizes = [250, 220, 150]
analytical = AnalyticalResult(N, shard_sizes)

# --- Build Jaccard index PMF ---
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

# --- Calculate the mean, std of the distribution and approximated expected Jaccard index ---
mu = mu_calculation(jaccard_list, prob_jaccard_list)
sigma = sigma_calculation(jaccard_list, prob_jaccard_list)
jaccard_mu_approx = ApproximatedResult(N, shard_sizes).jaccard_mu_approx()

# --- Plot the Jaccard index PMF ---
plot_hist_with_normal(
    jaccard_list,
    prob_jaccard_list,
    mu,
    sigma,
    title=f"Jaccard Index – Histogram vs Normal fit (N={N}, shards={shard_sizes})",
    xlabel="Jaccard index",
    vlines=[
        (mu, f"mean of empirical pmf={mu:.2f}", "b", "-"),
        (jaccard_mu_approx, f"Expected Jaccard={jaccard_mu_approx:.2f}", "r", "--"),
    ],
    bins=300,
)

plot_stem_pmf(
    jaccard_list,
    prob_jaccard_list,
    title=f"Jaccard – Stem PMF (N={N}, shards={shard_sizes})",
    xlabel="Jaccard index",
    vlines=[
        (mu, f"mean of empirical pmf={mu:.2f}", "b", "-"),
        (jaccard_mu_approx, f"Expected Jaccard = {jaccard_mu_approx:.3f}", "r", "--"),
    ],
)

# --- Calculate the mae and sse of the empirical pmf and the approximated normal distribution ---
pmf_norm = discretize_normal_pmf(jaccard_list, mu, sigma)
mae = mae_calculation(prob_jaccard_list, pmf_norm)
sse = sse_calculation(prob_jaccard_list, pmf_norm)
print(f"MAE: {mae:.4f}, SSE: {sse:.4f}")
print("sum pmf_emp:", sum(prob_jaccard_list), "sum pmf_norm:", sum(pmf_norm))
