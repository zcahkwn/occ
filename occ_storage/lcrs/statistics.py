"""
Statistical estimators for LCRS sketches based on OCC theory.

Ports the union/intersection/Jaccard estimators from the OCC library
for use with LCRS sketches in the storage system.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from .sketch import LCRS
import sys
import os

# Add parent directory to path to import OCC modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from occenv.analytical_univariate import AnalyticalUnivariate
from occenv.approximated import ApproximatedResult


class LCRSStatistics:
    """Statistical analysis for LCRS sketches using OCC theory."""

    def __init__(self, n_prime: int, tau: int):
        """
        Initialize statistics calculator.

        Args:
            n_prime: Number of registers in sketch
            tau: Maximum counter value
        """
        self.n_prime = n_prime
        self.tau = tau

    def estimate_union_intersection(
        self,
        sketches: List[LCRS]
    ) -> Tuple[float, float, float]:
        """
        Estimate union and intersection sizes from sketches.

        Args:
            sketches: List of LCRS sketches

        Returns:
            Tuple of (union_size, intersection_size, jaccard_similarity)
        """
        if not sketches:
            return 0.0, 0.0, 0.0

        # Get shard sizes from individual sketches
        shard_sizes = tuple(sketch.union_size() for sketch in sketches)

        # Total effective universe size is n_prime
        N = self.n_prime

        # Use OCC analytical methods if sizes are reasonable
        if N <= 1000 and len(shard_sizes) <= 10:
            try:
                ana = AnalyticalUnivariate(N, shard_sizes)
                union_mean = ana.union_mu()
                # For intersection, we need bivariate analysis
                # Simplified: use product of probabilities
                intersection_prob = np.prod([s/N for s in shard_sizes])
                intersection_mean = N * intersection_prob
            except:
                # Fall back to approximation if exact calculation fails
                union_mean, intersection_mean = self._approximate_union_intersection(
                    N, shard_sizes
                )
        else:
            # Use CLT approximation for large N
            union_mean, intersection_mean = self._approximate_union_intersection(
                N, shard_sizes
            )

        # Calculate Jaccard similarity
        jaccard = intersection_mean / union_mean if union_mean > 0 else 0.0

        return union_mean, intersection_mean, jaccard

    def _approximate_union_intersection(
        self,
        N: int,
        shard_sizes: Tuple[int, ...]
    ) -> Tuple[float, float]:
        """
        Use CLT approximation for union and intersection.

        Args:
            N: Universe size
            shard_sizes: Sizes of individual shards

        Returns:
            Tuple of (union_mean, intersection_mean)
        """
        try:
            approx = ApproximatedResult(N, shard_sizes)
            union_mean = approx.union_mu_approx()

            # Intersection approximation
            alphas = [s/N for s in shard_sizes]
            intersection_mean = N * np.prod(alphas)

            return union_mean, intersection_mean
        except:
            # Fallback to simple estimates if OCC methods fail
            return self._simple_estimates(N, shard_sizes)

    def _simple_estimates(
        self,
        N: int,
        shard_sizes: Tuple[int, ...]
    ) -> Tuple[float, float]:
        """
        Simple fallback estimates when OCC methods unavailable.

        Args:
            N: Universe size
            shard_sizes: Sizes of individual shards

        Returns:
            Tuple of (union_mean, intersection_mean)
        """
        # Union: Use inclusion-exclusion principle approximation
        alphas = [s/N for s in shard_sizes]
        union_prob = 1 - np.prod([1 - alpha for alpha in alphas])
        union_mean = N * union_prob

        # Intersection: Product of probabilities
        intersection_prob = np.prod(alphas)
        intersection_mean = N * intersection_prob

        return union_mean, intersection_mean

    def estimate_replication_distribution(
        self,
        Z: List[int]
    ) -> Dict[str, float]:
        """
        Estimate replication statistics from level counts.

        Args:
            Z: Level counts array

        Returns:
            Dictionary with replication statistics
        """
        total_items = sum(Z[1:])  # Exclude empty registers
        if total_items == 0:
            return {
                'mean_replication': 0.0,
                'variance_replication': 0.0,
                'entropy': 0.0
            }

        # Mean replication
        mean_rep = sum(level * count for level, count in enumerate(Z)) / total_items

        # Variance of replication
        var_rep = sum(
            count * (level - mean_rep)**2
            for level, count in enumerate(Z)
        ) / total_items

        # Shannon entropy of replication distribution
        entropy = 0.0
        for count in Z[1:]:
            if count > 0:
                p = count / total_items
                entropy -= p * np.log2(p)

        return {
            'mean_replication': mean_rep,
            'variance_replication': var_rep,
            'std_replication': np.sqrt(var_rep),
            'entropy': entropy
        }

    def estimate_coverage_confidence(
        self,
        observed_coverage: int,
        n_prime: int,
        confidence: float = 0.95
    ) -> Tuple[int, int]:
        """
        Compute confidence bounds for coverage estimate.

        Args:
            observed_coverage: Observed W_{≥T} value
            n_prime: Total number of registers
            confidence: Confidence level

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        from scipy import stats

        # Binomial proportion confidence interval
        p_hat = observed_coverage / n_prime if n_prime > 0 else 0

        # Wilson score interval (better for edge cases)
        z = stats.norm.ppf((1 + confidence) / 2)
        denominator = 1 + z**2 / n_prime

        center = (p_hat + z**2 / (2 * n_prime)) / denominator
        margin = z * np.sqrt(
            p_hat * (1 - p_hat) / n_prime + z**2 / (4 * n_prime**2)
        ) / denominator

        lower_p = max(0, center - margin)
        upper_p = min(1, center + margin)

        return int(lower_p * n_prime), int(upper_p * n_prime)

    def jaccard_similarity_bounds(
        self,
        sketches: List[LCRS],
        confidence: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Estimate Jaccard similarity with confidence bounds.

        Args:
            sketches: List of LCRS sketches
            confidence: Confidence level

        Returns:
            Tuple of (jaccard_estimate, lower_bound, upper_bound)
        """
        from scipy import stats

        union_est, inter_est, jaccard_est = self.estimate_union_intersection(sketches)

        # Simplified confidence bounds using delta method
        # In practice, would use full covariance matrix from patent
        if union_est == 0:
            return 0.0, 0.0, 0.0

        # Approximate standard error (simplified)
        se_jaccard = jaccard_est * np.sqrt(
            1/max(1, inter_est) + 1/max(1, union_est)
        ) / 2

        z = stats.norm.ppf((1 + confidence) / 2)
        lower = max(0, jaccard_est - z * se_jaccard)
        upper = min(1, jaccard_est + z * se_jaccard)

        return jaccard_est, lower, upper

    def detect_anomalies(
        self,
        Z: List[int],
        expected_distribution: Optional[Dict[int, float]] = None
    ) -> Dict[str, any]:
        """
        Detect anomalies in replication distribution.

        Args:
            Z: Observed level counts
            expected_distribution: Expected distribution (if known)

        Returns:
            Dictionary with anomaly detection results
        """
        total_items = sum(Z[1:])
        if total_items == 0:
            return {'anomalous': False, 'reason': 'No items stored'}

        # Check for unusual concentration
        max_level = max((i for i, count in enumerate(Z) if count > 0), default=0)
        if max_level == len(Z) - 1:
            # All items at maximum replication
            concentration = Z[max_level] / total_items
            if concentration > 0.5:
                return {
                    'anomalous': True,
                    'reason': f'High concentration at max replication: {concentration:.2%}',
                    'max_level_concentration': concentration
                }

        # Check for too many single-replica items
        if len(Z) > 1:
            single_replica_ratio = Z[1] / total_items
            if single_replica_ratio > 0.5:
                return {
                    'anomalous': True,
                    'reason': f'Too many single-replica items: {single_replica_ratio:.2%}',
                    'single_replica_ratio': single_replica_ratio
                }

        # Chi-square test if expected distribution provided
        if expected_distribution:
            from scipy.stats import chisquare

            observed_dist = [Z[i]/total_items if i < len(Z) else 0
                           for i in range(max(expected_distribution.keys()) + 1)]
            expected = [expected_distribution.get(i, 0)
                       for i in range(len(observed_dist))]

            if sum(expected) > 0:
                chi2, p_value = chisquare(
                    [o * total_items for o in observed_dist],
                    [e * total_items for e in expected]
                )

                if p_value < 0.01:
                    return {
                        'anomalous': True,
                        'reason': f'Distribution differs from expected (p={p_value:.4f})',
                        'chi_square': chi2,
                        'p_value': p_value
                    }

        return {'anomalous': False, 'reason': 'Distribution appears normal'}