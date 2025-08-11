"""
This script is used to demonstrate the analytical results of the occenv.analytical.AnalyticalResult class.
"""

from occenv.analytical_univariate import AnalyticalUnivariate
from occenv.analytical_bivariate import AnalyticalBivariate
from occenv.approximated import ApproximatedResult

N = 10
shard_sizes = [7, 9]
print(f"The total number is {N} and the shard sizes are {shard_sizes}")
compute_univariate = AnalyticalUnivariate(N, shard_sizes)
compute_bivariate = AnalyticalBivariate(N, shard_sizes)
compute_approximated = ApproximatedResult(N, shard_sizes)

# Calculate union and intersect probability
collusion_probability = compute_univariate.union_prob(N)
sigma_value = compute_approximated.sigma_value()
occ_value = compute_approximated.occ_value()

print("Expected total union =", N * sigma_value)
print("Expected total intersection =", N * occ_value)

union_test = 7
intersect_test = 6

print(
    f"Probability that the union size is {union_test} =",
    compute_univariate.union_prob(union_test),
)
print(
    f"Probability that the intersect size is {intersect_test} =",
    compute_univariate.intersection_prob(intersect_test),
)

print(f"Probability of collusion (i.e. union = {N})=", collusion_probability)

# Calculate bivariate probability
union_pmf = compute_univariate.union_prob(union_test)
intersect_pmf = compute_univariate.intersection_prob(intersect_test)

print(
    f"\n------Test bivariate probability-----\nWhen the union size is {union_test} and the intersect size is {intersect_test}, the following results are obtained: \n"
)
bivariate_prob = compute_bivariate.bivariate_prob(union_test, intersect_test)
print(f"bivariate probability for {union_test, intersect_test} = {bivariate_prob}")

# Calculate Jaccard index
pair = (58, 12)
jaccard_prob = compute_bivariate.jaccard_prob(6, 29)
print(
    f"------Test Jaccard index------\nJaccard probability for {6, 29} = {jaccard_prob}"
)

# Check whether marginal probability for bivariate probability conditional on union size adds up to intersect_pmf when intersect_size is fixed
marginal_prob_intersect = sum(
    compute_bivariate.bivariate_prob(union_size, intersect_test)
    for union_size in range(1, N + 1)
)
print(
    f"When intersect_size = {intersect_test}, \nsum of bivariate probability conditional on union size =",
    marginal_prob_intersect,
)
print(f"Probability that intersect size is {intersect_test} =", intersect_pmf)
if intersect_pmf - marginal_prob_intersect < 1e-6:
    print("=> The marginal probability conditional on union size is correct\n")
else:
    print("=> The marginal probability conditional on union size is not correct\n")

# Check whether marginal probability for bivariate probability conditional on intersect size adds up to union_pmf when union_size is fixed
marginal_prob_union = sum(
    compute_bivariate.bivariate_prob(union_test, intersect_size)
    for intersect_size in range(1, N + 1)
)
print(
    f"When union_size = {union_test}, \nsum of bivariate probability conditional on intersect size =",
    marginal_prob_union,
)
print(f"Probability that union size is {union_test} =", union_pmf)
if union_pmf - marginal_prob_union < 1e5:
    print("=> The marginal probability conditional on intersect size is correct\n")
else:
    print("=> The marginal probability conditional on intersect size is not correct\n")
