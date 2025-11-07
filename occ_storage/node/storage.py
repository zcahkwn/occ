"""
Storage node implementation for OCC-based distributed file storage.
"""

import os
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from ..lcrs.sketch import LCRS, LCRSConfig


@dataclass
class ChunkMetadata:
    """Metadata for a stored chunk."""
    chunk_hash: str
    size: int
    created_at: str
    accessed_at: str
    access_count: int = 0


class StorageNode:
    """
    A storage node in the OCC-based distributed storage system.

    Manages local chunk storage and maintains LCRS sketch for
    probabilistic tracking.
    """

    def __init__(
        self,
        node_id: str,
        storage_path: str,
        n_prime: int = 10000,
        tau: int = 10
    ):
        """
        Initialize storage node.

        Args:
            node_id: Unique identifier for this node
            storage_path: Directory path for storing chunks
            n_prime: Number of registers in LCRS sketch
            tau: Maximum counter value in LCRS
        """
        self.node_id = node_id
        self.storage_path = Path(storage_path)
        self.chunks_dir = self.storage_path / "chunks"
        self.metadata_dir = self.storage_path / "metadata"

        # Create storage directories
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # Initialize LCRS sketch
        lcrs_config = LCRSConfig(
            n_prime=n_prime,
            tau=tau,
            node_id=node_id
        )
        self.sketch = LCRS(lcrs_config)

        # Load or initialize metadata
        self.metadata_file = self.metadata_dir / "chunks.json"
        self.chunk_metadata: Dict[str, ChunkMetadata] = self._load_metadata()

        # Initialize sketch from existing chunks
        if self.chunk_metadata:
            self.sketch.update(list(self.chunk_metadata.keys()))

    def _load_metadata(self) -> Dict[str, ChunkMetadata]:
        """Load chunk metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
                return {
                    chunk_hash: ChunkMetadata(**meta_dict)
                    for chunk_hash, meta_dict in data.items()
                }
        return {}

    def _save_metadata(self):
        """Save chunk metadata to disk."""
        data = {
            chunk_hash: asdict(metadata)
            for chunk_hash, metadata in self.chunk_metadata.items()
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _chunk_path(self, chunk_hash: str) -> Path:
        """
        Get file path for a chunk.

        Uses first 2 chars as subdirectory for better file system performance.
        """
        subdir = self.chunks_dir / chunk_hash[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / chunk_hash

    def store_chunk(self, data: bytes) -> str:
        """
        Store a chunk of data.

        Args:
            data: Raw chunk data

        Returns:
            Hash of the stored chunk
        """
        # Calculate chunk hash
        chunk_hash = hashlib.sha256(data).hexdigest()

        # Check if already stored
        if chunk_hash in self.chunk_metadata:
            # Update access time
            self.chunk_metadata[chunk_hash].accessed_at = datetime.now().isoformat()
            self.chunk_metadata[chunk_hash].access_count += 1
            self._save_metadata()
            return chunk_hash

        # Store chunk data
        chunk_path = self._chunk_path(chunk_hash)
        with open(chunk_path, 'wb') as f:
            f.write(data)

        # Update metadata
        now = datetime.now().isoformat()
        self.chunk_metadata[chunk_hash] = ChunkMetadata(
            chunk_hash=chunk_hash,
            size=len(data),
            created_at=now,
            accessed_at=now
        )
        self._save_metadata()

        # Update LCRS sketch
        self.sketch.update([chunk_hash])

        return chunk_hash

    def retrieve_chunk(self, chunk_hash: str) -> Optional[bytes]:
        """
        Retrieve a chunk by its hash.

        Args:
            chunk_hash: Hash of the chunk to retrieve

        Returns:
            Chunk data if found, None otherwise
        """
        if chunk_hash not in self.chunk_metadata:
            return None

        chunk_path = self._chunk_path(chunk_hash)
        if not chunk_path.exists():
            # Metadata inconsistency - chunk file missing
            del self.chunk_metadata[chunk_hash]
            self._save_metadata()
            return None

        # Read chunk data
        with open(chunk_path, 'rb') as f:
            data = f.read()

        # Update access metadata
        self.chunk_metadata[chunk_hash].accessed_at = datetime.now().isoformat()
        self.chunk_metadata[chunk_hash].access_count += 1
        self._save_metadata()

        return data

    def delete_chunk(self, chunk_hash: str) -> bool:
        """
        Delete a chunk from storage.

        Args:
            chunk_hash: Hash of the chunk to delete

        Returns:
            True if deleted, False if not found
        """
        if chunk_hash not in self.chunk_metadata:
            return False

        # Delete chunk file
        chunk_path = self._chunk_path(chunk_hash)
        if chunk_path.exists():
            chunk_path.unlink()

        # Update metadata
        del self.chunk_metadata[chunk_hash]
        self._save_metadata()

        # Update LCRS sketch
        self.sketch.remove([chunk_hash])

        return True

    def has_chunk(self, chunk_hash: str) -> bool:
        """
        Check if node has a specific chunk.

        Args:
            chunk_hash: Hash of the chunk

        Returns:
            True if chunk is stored, False otherwise
        """
        return chunk_hash in self.chunk_metadata

    def list_chunks(self) -> List[str]:
        """
        List all chunk hashes stored by this node.

        Returns:
            List of chunk hashes
        """
        return list(self.chunk_metadata.keys())

    def get_storage_stats(self) -> Dict:
        """
        Get storage statistics for this node.

        Returns:
            Dictionary with storage statistics
        """
        total_size = sum(meta.size for meta in self.chunk_metadata.values())
        total_accesses = sum(meta.access_count for meta in self.chunk_metadata.values())

        return {
            'node_id': self.node_id,
            'total_chunks': len(self.chunk_metadata),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'total_accesses': total_accesses,
            'average_chunk_size': total_size / len(self.chunk_metadata) if self.chunk_metadata else 0,
            'sketch_union_size': self.sketch.union_size(),
            'sketch_level_counts': self.sketch.level_counts()
        }

    def garbage_collect(self, max_age_days: int = 30, min_access_count: int = 1) -> int:
        """
        Remove old, rarely accessed chunks.

        Args:
            max_age_days: Remove chunks older than this many days
            min_access_count: Keep chunks accessed at least this many times

        Returns:
            Number of chunks removed
        """
        from datetime import datetime, timedelta

        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        chunks_to_remove = []

        for chunk_hash, metadata in self.chunk_metadata.items():
            accessed_at = datetime.fromisoformat(metadata.accessed_at)
            if (accessed_at < cutoff_date and
                metadata.access_count < min_access_count):
                chunks_to_remove.append(chunk_hash)

        # Remove identified chunks
        removed_count = 0
        for chunk_hash in chunks_to_remove:
            if self.delete_chunk(chunk_hash):
                removed_count += 1

        return removed_count

    def export_sketch(self) -> Dict:
        """
        Export LCRS sketch for sharing with other nodes.

        Returns:
            Serialized sketch data
        """
        return self.sketch.to_dict()

    def import_sketch(self, sketch_data: Dict) -> LCRS:
        """
        Import an LCRS sketch from another node.

        Args:
            sketch_data: Serialized sketch data

        Returns:
            Imported LCRS sketch
        """
        return LCRS.from_dict(sketch_data)

    def estimate_chunk_probability(self, chunk_hash: str) -> float:
        """
        Estimate probability that this node has a specific chunk.

        Args:
            chunk_hash: Hash of the chunk

        Returns:
            Estimated probability [0, 1]
        """
        # First check actual storage
        if self.has_chunk(chunk_hash):
            return 1.0

        # Otherwise use sketch estimate
        return self.sketch.estimate_chunk_probability(chunk_hash)

    def get_replication_targets(
        self,
        chunk_hash: str,
        target_replication: int,
        other_sketches: Dict[str, LCRS]
    ) -> List[str]:
        """
        Identify nodes that should replicate a chunk.

        Args:
            chunk_hash: Hash of the chunk to replicate
            target_replication: Desired replication factor
            other_sketches: Sketches from other nodes

        Returns:
            List of node IDs that should store replicas
        """
        # Estimate current replication
        current_holders = [self.node_id] if self.has_chunk(chunk_hash) else []

        for node_id, sketch in other_sketches.items():
            if sketch.estimate_chunk_probability(chunk_hash) > 0.5:
                current_holders.append(node_id)

        current_replication = len(current_holders)
        needed_replicas = max(0, target_replication - current_replication)

        if needed_replicas == 0:
            return []

        # Select nodes with lowest current load
        node_loads = [
            (node_id, sketch.union_size())
            for node_id, sketch in other_sketches.items()
            if node_id not in current_holders
        ]
        node_loads.sort(key=lambda x: x[1])

        return [node_id for node_id, _ in node_loads[:needed_replicas]]

    def __repr__(self) -> str:
        """String representation of storage node."""
        stats = self.get_storage_stats()
        return (
            f"StorageNode(id={self.node_id}, "
            f"chunks={stats['total_chunks']}, "
            f"size={stats['total_size_mb']:.2f}MB)"
        )