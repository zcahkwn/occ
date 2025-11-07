"""
LCRS operations for multi-node aggregation and querying.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from .sketch import LCRS, LCRSConfig
from scipy import stats


class LCRSOperations:
    """Operations for working with multiple LCRS sketches."""

    @staticmethod
    def merge_multiple(sketches: List[LCRS]) -> LCRS:
        """
        Merge multiple LCRS sketches into one.

        Args:
            sketches: List of LCRS sketches to merge

        Returns:
            Single merged LCRS
        """
        if not sketches:
            raise ValueError("Cannot merge empty list of sketches")

        result = sketches[0]
        for sketch in sketches[1:]:
            result = result.merge(sketch)

        return result

    @staticmethod
    def compute_network_statistics(sketches: List[LCRS]) -> Dict:
        """
        Compute network-wide statistics from all node sketches.

        Args:
            sketches: List of LCRS sketches from all nodes

        Returns:
            Dictionary with network statistics
        """
        # Merge all sketches
        merged = LCRSOperations.merge_multiple(sketches)

        # Get level counts
        Z = merged.level_counts()

        # Compute key statistics
        stats = {
            'total_nodes': len(sketches),
            'union_size': merged.union_size(),
            'level_counts': Z,
            'single_replica_items': Z[1] if len(Z) > 1 else 0,
            'fully_replicated_items': Z[-1] if len(Z) > 0 else 0,
            'average_replication': LCRSOperations._compute_avg_replication(Z),
            'replication_distribution': LCRSOperations._compute_replication_dist(Z)
        }

        # Add threshold statistics
        for t in [2, 3, 5]:
            stats[f'items_with_{t}_or_more_replicas'] = merged.coverage_at_threshold(t)

        return stats

    @staticmethod
    def _compute_avg_replication(Z: List[int]) -> float:
        """
        Compute average replication factor from level counts.

        Args:
            Z: Level counts array

        Returns:
            Average replication factor
        """
        total_items = sum(Z[1:])  # Exclude Z[0] (empty registers)
        if total_items == 0:
            return 0.0

        total_replicas = sum(level * count for level, count in enumerate(Z))
        return total_replicas / total_items

    @staticmethod
    def _compute_replication_dist(Z: List[int]) -> Dict[int, float]:
        """
        Compute replication factor distribution.

        Args:
            Z: Level counts array

        Returns:
            Dictionary mapping replication factor to probability
        """
        total_items = sum(Z[1:])
        if total_items == 0:
            return {}

        dist = {}
        for level, count in enumerate(Z):
            if level > 0 and count > 0:
                dist[level] = count / total_items

        return dist

    @staticmethod
    def find_likely_providers(
        chunk_hash: str,
        node_sketches: Dict[str, LCRS],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find nodes most likely to have a specific chunk.

        Args:
            chunk_hash: Hash of the chunk to find
            node_sketches: Dictionary mapping node_id to LCRS sketch
            top_k: Number of top candidates to return

        Returns:
            List of (node_id, probability) tuples, sorted by probability
        """
        candidates = []

        for node_id, sketch in node_sketches.items():
            # Estimate probability this node has the chunk
            prob = sketch.estimate_chunk_probability(chunk_hash)
            if prob > 0:
                candidates.append((node_id, prob))

        # Sort by probability (descending) and return top-k
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    @staticmethod
    def compute_confidence_bounds(
        sketches: List[LCRS],
        threshold: int,
        confidence: float = 0.95
    ) -> Tuple[int, int, int]:
        """
        Compute confidence bounds for W_{≥T} (items with ≥T replicas).

        Args:
            sketches: List of LCRS sketches
            threshold: Replication threshold T
            confidence: Confidence level (e.g., 0.95 for 95%)

        Returns:
            Tuple of (observed_value, lower_bound, upper_bound)
        """
        merged = LCRSOperations.merge_multiple(sketches)
        observed = merged.coverage_at_threshold(threshold)

        # Estimate variance (simplified - in practice use formulas from patent)
        # This is a placeholder for the actual variance calculation
        Z = merged.level_counts()
        n_prime = merged.n_prime

        # Simple binomial approximation for variance
        p_estimate = observed / n_prime if n_prime > 0 else 0
        variance = n_prime * p_estimate * (1 - p_estimate)
        std_dev = np.sqrt(variance)

        # Compute z-score for confidence level
        z_score = stats.norm.ppf((1 + confidence) / 2)

        # Calculate bounds
        lower_bound = max(0, int(observed - z_score * std_dev))
        upper_bound = min(n_prime, int(observed + z_score * std_dev))

        return observed, lower_bound, upper_bound

    @staticmethod
    def detect_underreplicated_chunks(
        sketches: List[LCRS],
        target_replication: int = 3,
        alert_threshold: float = 0.1
    ) -> bool:
        """
        Detect if too many chunks are underreplicated.

        Args:
            sketches: List of LCRS sketches
            target_replication: Desired replication factor
            alert_threshold: Alert if more than this fraction underreplicated

        Returns:
            True if underreplication detected, False otherwise
        """
        merged = LCRSOperations.merge_multiple(sketches)
        Z = merged.level_counts()

        # Count items with less than target replication
        underreplicated = sum(Z[1:target_replication])
        total_items = sum(Z[1:])

        if total_items == 0:
            return False

        underreplicated_fraction = underreplicated / total_items
        return underreplicated_fraction > alert_threshold

    @staticmethod
    def estimate_storage_balance(node_sketches: Dict[str, LCRS]) -> Dict[str, float]:
        """
        Estimate storage load balance across nodes.

        Args:
            node_sketches: Dictionary mapping node_id to LCRS sketch

        Returns:
            Dictionary with balance metrics
        """
        if not node_sketches:
            return {}

        # Collect union sizes for each node
        node_loads = {
            node_id: sketch.union_size()
            for node_id, sketch in node_sketches.items()
        }

        # Compute statistics
        loads = list(node_loads.values())
        mean_load = np.mean(loads)
        std_load = np.std(loads)
        cv = std_load / mean_load if mean_load > 0 else 0

        return {
            'node_loads': node_loads,
            'mean_load': mean_load,
            'std_load': std_load,
            'coefficient_of_variation': cv,
            'max_load': max(loads),
            'min_load': min(loads),
            'imbalance_ratio': max(loads) / min(loads) if min(loads) > 0 else float('inf')
        }