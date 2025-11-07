"""
File distribution and retrieval for OCC storage network.
"""

import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .chunker import FileChunker, FileMetadata
from ..node.storage import StorageNode
from ..lcrs.operations import LCRSOperations
from ..lcrs.statistics import LCRSStatistics


@dataclass
class DistributionStrategy:
    """Configuration for file distribution strategy."""
    replication_factor: int = 3
    min_nodes: int = 1
    use_probabilistic_routing: bool = True
    prefer_low_load_nodes: bool = True


class FileDistributor:
    """
    Handles file distribution and retrieval across storage nodes.
    """

    def __init__(
        self,
        nodes: Dict[str, StorageNode],
        strategy: Optional[DistributionStrategy] = None
    ):
        """
        Initialize file distributor.

        Args:
            nodes: Dictionary mapping node_id to StorageNode
            strategy: Distribution strategy configuration
        """
        self.nodes = nodes
        self.strategy = strategy or DistributionStrategy()
        self.chunker = FileChunker()

        # Statistics calculator
        if nodes:
            sample_node = next(iter(nodes.values()))
            self.stats = LCRSStatistics(
                n_prime=sample_node.sketch.n_prime,
                tau=sample_node.sketch.tau
            )

    def add_file(
        self,
        file_path: str,
        replication_factor: Optional[int] = None
    ) -> Tuple[FileMetadata, Dict[str, List[str]]]:
        """
        Add a file to the distributed storage network.

        Args:
            file_path: Path to the file to add
            replication_factor: Override default replication factor

        Returns:
            Tuple of (file metadata, distribution map)
        """
        if replication_factor is None:
            replication_factor = self.strategy.replication_factor

        # Chunk the file
        metadata, chunks = self.chunker.chunk_file(file_path)

        # Determine distribution
        distribution = self._determine_distribution(
            metadata,
            replication_factor
        )

        # Store chunks on nodes
        actual_distribution = {}
        for chunk_idx, chunk_data in enumerate(chunks):
            chunk_hash = metadata.chunk_hashes[chunk_idx]
            stored_on = []

            for node_id in distribution.get(chunk_idx, []):
                if node_id in self.nodes:
                    stored_hash = self.nodes[node_id].store_chunk(chunk_data)
                    if stored_hash == chunk_hash:
                        stored_on.append(node_id)

            actual_distribution[chunk_hash] = stored_on

        return metadata, actual_distribution

    def retrieve_file(
        self,
        metadata: FileMetadata,
        output_path: Optional[str] = None
    ) -> str:
        """
        Retrieve a file from the distributed storage network.

        Args:
            metadata: File metadata
            output_path: Path to save retrieved file

        Returns:
            Path to reconstructed file

        Raises:
            FileNotFoundError: If unable to retrieve all chunks
        """
        chunks = []

        for chunk_idx, chunk_hash in enumerate(metadata.chunk_hashes):
            # Find nodes that might have this chunk
            chunk_data = self._retrieve_chunk(chunk_hash)

            if chunk_data is None:
                raise FileNotFoundError(
                    f"Unable to retrieve chunk {chunk_idx} "
                    f"(hash: {chunk_hash})"
                )

            chunks.append(chunk_data)

        # Reconstruct file
        return self.chunker.reconstruct_file(metadata, chunks, output_path)

    def _retrieve_chunk(self, chunk_hash: str) -> Optional[bytes]:
        """
        Retrieve a single chunk from the network.

        Args:
            chunk_hash: Hash of the chunk to retrieve

        Returns:
            Chunk data if found, None otherwise
        """
        if self.strategy.use_probabilistic_routing:
            # Use LCRS sketches to find likely providers
            node_sketches = {
                node_id: node.sketch
                for node_id, node in self.nodes.items()
            }

            likely_providers = LCRSOperations.find_likely_providers(
                chunk_hash,
                node_sketches,
                top_k=5
            )

            # Try likely providers first
            for node_id, _ in likely_providers:
                if node_id in self.nodes:
                    chunk_data = self.nodes[node_id].retrieve_chunk(chunk_hash)
                    if chunk_data is not None:
                        return chunk_data

        # Fall back to checking all nodes
        for node in self.nodes.values():
            chunk_data = node.retrieve_chunk(chunk_hash)
            if chunk_data is not None:
                return chunk_data

        return None

    def _determine_distribution(
        self,
        metadata: FileMetadata,
        replication_factor: int
    ) -> Dict[int, List[str]]:
        """
        Determine which nodes should store which chunks.

        Args:
            metadata: File metadata
            replication_factor: Number of replicas per chunk

        Returns:
            Dictionary mapping chunk_idx to list of node_ids
        """
        distribution = {}
        available_nodes = list(self.nodes.keys())

        if len(available_nodes) < self.strategy.min_nodes:
            raise ValueError(
                f"Not enough nodes: need {self.strategy.min_nodes}, "
                f"have {len(available_nodes)}"
            )

        for chunk_idx in range(metadata.chunk_count):
            if self.strategy.prefer_low_load_nodes:
                # Sort nodes by current load
                node_loads = [
                    (node_id, self.nodes[node_id].sketch.union_size())
                    for node_id in available_nodes
                ]
                node_loads.sort(key=lambda x: x[1])

                # Select least loaded nodes
                selected_nodes = [
                    node_id for node_id, _ in
                    node_loads[:min(replication_factor, len(node_loads))]
                ]
            else:
                # Random selection
                selected_nodes = random.sample(
                    available_nodes,
                    min(replication_factor, len(available_nodes))
                )

            distribution[chunk_idx] = selected_nodes

        return distribution

    def verify_file_availability(self, metadata: FileMetadata) -> Dict[str, any]:
        """
        Verify that a file can be retrieved from the network.

        Args:
            metadata: File metadata to verify

        Returns:
            Dictionary with availability information
        """
        chunk_availability = {}
        missing_chunks = []
        replication_counts = []

        for chunk_idx, chunk_hash in enumerate(metadata.chunk_hashes):
            # Count how many nodes have this chunk
            holders = []
            for node_id, node in self.nodes.items():
                if node.has_chunk(chunk_hash):
                    holders.append(node_id)

            chunk_availability[chunk_hash] = holders
            replication_counts.append(len(holders))

            if len(holders) == 0:
                missing_chunks.append(chunk_idx)

        return {
            'retrievable': len(missing_chunks) == 0,
            'missing_chunks': missing_chunks,
            'chunk_availability': chunk_availability,
            'min_replication': min(replication_counts) if replication_counts else 0,
            'max_replication': max(replication_counts) if replication_counts else 0,
            'avg_replication': sum(replication_counts) / len(replication_counts) if replication_counts else 0
        }

    def rebalance_file(
        self,
        metadata: FileMetadata,
        target_replication: Optional[int] = None
    ) -> Dict[str, List[str]]:
        """
        Rebalance file chunks to meet replication targets.

        Args:
            metadata: File metadata
            target_replication: Desired replication factor

        Returns:
            Updated distribution map
        """
        if target_replication is None:
            target_replication = self.strategy.replication_factor

        new_distribution = {}

        for chunk_idx, chunk_hash in enumerate(metadata.chunk_hashes):
            # Find current holders
            current_holders = []
            for node_id, node in self.nodes.items():
                if node.has_chunk(chunk_hash):
                    current_holders.append(node_id)

            current_replication = len(current_holders)

            if current_replication < target_replication:
                # Need more replicas
                # First, get the chunk data from an existing holder
                chunk_data = None
                for node_id in current_holders:
                    chunk_data = self.nodes[node_id].retrieve_chunk(chunk_hash)
                    if chunk_data:
                        break

                if chunk_data:
                    # Find nodes to add replicas
                    available_nodes = [
                        node_id for node_id in self.nodes.keys()
                        if node_id not in current_holders
                    ]

                    # Sort by load if preferred
                    if self.strategy.prefer_low_load_nodes:
                        available_nodes.sort(
                            key=lambda nid: self.nodes[nid].sketch.union_size()
                        )

                    # Add replicas
                    replicas_needed = target_replication - current_replication
                    for node_id in available_nodes[:replicas_needed]:
                        self.nodes[node_id].store_chunk(chunk_data)
                        current_holders.append(node_id)

            elif current_replication > target_replication:
                # Too many replicas, remove some
                # Sort by load (remove from most loaded nodes)
                current_holders.sort(
                    key=lambda nid: self.nodes[nid].sketch.union_size(),
                    reverse=True
                )

                # Remove excess replicas
                excess = current_replication - target_replication
                for node_id in current_holders[:excess]:
                    self.nodes[node_id].delete_chunk(chunk_hash)
                    current_holders.remove(node_id)

            new_distribution[chunk_hash] = current_holders

        return new_distribution

    def get_network_statistics(self) -> Dict:
        """
        Get comprehensive network statistics.

        Returns:
            Dictionary with network-wide statistics
        """
        sketches = [node.sketch for node in self.nodes.values()]

        # Compute network statistics
        network_stats = LCRSOperations.compute_network_statistics(sketches)

        # Add storage balance
        node_sketches = {
            node_id: node.sketch
            for node_id, node in self.nodes.items()
        }
        balance_stats = LCRSOperations.estimate_storage_balance(node_sketches)

        # Combine statistics
        return {
            **network_stats,
            'balance': balance_stats,
            'total_storage_bytes': sum(
                node.get_storage_stats()['total_size_bytes']
                for node in self.nodes.values()
            ),
            'total_chunks_stored': sum(
                node.get_storage_stats()['total_chunks']
                for node in self.nodes.values()
            )
        }