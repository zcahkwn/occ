"""
Level-Count Redundancy Sketch (LCRS) implementation for OCC-based storage.

Based on the patent's saturating counter approach for tracking data distribution
across multiple parties without revealing exact file locations.
"""

import hashlib
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class LCRSConfig:
    """Configuration for LCRS sketch."""
    n_prime: int  # Number of registers
    tau: int      # Maximum counter value (saturation threshold)
    node_id: str  # Unique identifier for this node
    seed: int = 42  # Seed for PRF


class LCRS:
    """Level-Count Redundancy Sketch for distributed storage tracking."""

    def __init__(self, config: LCRSConfig):
        """
        Initialize LCRS with given configuration.

        Args:
            config: LCRS configuration parameters
        """
        self.config = config
        self.n_prime = config.n_prime
        self.tau = config.tau
        self.node_id = config.node_id

        # Initialize saturating counters
        # Each counter uses ceil(log2(tau+1)) bits
        self.counters = np.zeros(self.n_prime, dtype=np.uint8)

        # Track which chunks this node has seen (for deduplication)
        self.seen_chunks = set()

    def _prf(self, chunk_hash: str) -> int:
        """
        Pseudo-random function mapping chunk hash to register index.

        Args:
            chunk_hash: Hash of the chunk

        Returns:
            Register index in [0, n_prime)
        """
        # Combine node_id and chunk_hash for PRF
        combined = f"{self.node_id}:{chunk_hash}:{self.config.seed}"
        hash_bytes = hashlib.sha256(combined.encode()).digest()
        # Convert first 8 bytes to integer and mod by n_prime
        index = int.from_bytes(hash_bytes[:8], 'big') % self.n_prime
        return index

    def update(self, chunk_hashes: List[str]):
        """
        Update sketch with new chunks stored by this node.

        Args:
            chunk_hashes: List of chunk hashes being added
        """
        for chunk_hash in chunk_hashes:
            if chunk_hash not in self.seen_chunks:
                # Map chunk to register using PRF
                register_idx = self._prf(chunk_hash)

                # Increment counter with saturation at tau
                self.counters[register_idx] = min(
                    self.counters[register_idx] + 1,
                    self.tau
                )

                # Mark chunk as seen
                self.seen_chunks.add(chunk_hash)

    def remove(self, chunk_hashes: List[str]):
        """
        Remove chunks from sketch (decrement counters).

        Args:
            chunk_hashes: List of chunk hashes being removed
        """
        for chunk_hash in chunk_hashes:
            if chunk_hash in self.seen_chunks:
                register_idx = self._prf(chunk_hash)

                # Decrement counter (minimum 0)
                if self.counters[register_idx] > 0:
                    self.counters[register_idx] -= 1

                self.seen_chunks.discard(chunk_hash)

    def merge(self, other: 'LCRS') -> 'LCRS':
        """
        Merge two sketches (for aggregating across nodes).

        Args:
            other: Another LCRS sketch to merge with

        Returns:
            New LCRS containing merged counts
        """
        if self.n_prime != other.n_prime or self.tau != other.tau:
            raise ValueError("Cannot merge sketches with different configurations")

        # Create new sketch for result
        merged_config = LCRSConfig(
            n_prime=self.n_prime,
            tau=self.tau,
            node_id=f"{self.node_id}+{other.node_id}",
            seed=self.config.seed
        )
        merged = LCRS(merged_config)

        # Element-wise capped addition
        for i in range(self.n_prime):
            merged.counters[i] = min(
                self.counters[i] + other.counters[i],
                self.tau
            )

        # Union of seen chunks
        merged.seen_chunks = self.seen_chunks.union(other.seen_chunks)

        return merged

    def level_counts(self) -> List[int]:
        """
        Compute level counts Z_ℓ (number of registers with exactly ℓ parties).

        Returns:
            Array where Z[ℓ] = number of registers with count ℓ
        """
        Z = np.zeros(self.tau + 1, dtype=int)
        for count in self.counters:
            Z[count] += 1
        return Z.tolist()

    def union_size(self) -> int:
        """
        Estimate union size U = N' - Z_0.

        Returns:
            Estimated number of unique items across all parties
        """
        Z = self.level_counts()
        return self.n_prime - Z[0]

    def intersection_size(self, m: int) -> int:
        """
        Estimate intersection size I = Z_τ (for τ = m parties).

        Args:
            m: Number of parties

        Returns:
            Estimated number of items held by all m parties
        """
        if m > self.tau:
            return 0
        Z = self.level_counts()
        return Z[m] if m < len(Z) else 0

    def coverage_at_threshold(self, threshold: int) -> int:
        """
        Compute W_{≥T} = number of items with at least T replicas.

        Args:
            threshold: Minimum replication factor T

        Returns:
            Number of registers with count ≥ T
        """
        Z = self.level_counts()
        return sum(Z[threshold:])

    def to_dict(self) -> Dict:
        """
        Serialize sketch to dictionary.

        Returns:
            Dictionary representation of sketch
        """
        return {
            'config': {
                'n_prime': self.config.n_prime,
                'tau': self.config.tau,
                'node_id': self.config.node_id,
                'seed': self.config.seed
            },
            'counters': self.counters.tolist(),
            'seen_chunks': list(self.seen_chunks)
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'LCRS':
        """
        Deserialize sketch from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            Reconstructed LCRS sketch
        """
        config = LCRSConfig(**data['config'])
        sketch = cls(config)
        sketch.counters = np.array(data['counters'], dtype=np.uint8)
        sketch.seen_chunks = set(data['seen_chunks'])
        return sketch

    def estimate_chunk_probability(self, chunk_hash: str) -> float:
        """
        Estimate probability that this sketch contains a specific chunk.

        Args:
            chunk_hash: Hash of the chunk to check

        Returns:
            Estimated probability in [0, 1]
        """
        register_idx = self._prf(chunk_hash)
        count = self.counters[register_idx]

        # Simple estimate: count/tau represents confidence
        # More sophisticated estimates would use the full distribution
        return min(count / max(1, self.tau), 1.0)

    def __repr__(self) -> str:
        """String representation of sketch."""
        Z = self.level_counts()
        union = self.union_size()
        return (
            f"LCRS(node={self.node_id}, n'={self.n_prime}, τ={self.tau}, "
            f"union={union}, levels={Z})"
        )