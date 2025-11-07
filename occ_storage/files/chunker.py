"""
File chunking and reconstruction utilities for OCC storage.
"""

import hashlib
import json
from typing import List, Dict, Tuple, Optional, BinaryIO
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class FileMetadata:
    """Metadata for a chunked file."""
    file_hash: str
    file_name: str
    file_size: int
    chunk_size: int
    chunk_hashes: List[str]
    chunk_count: int


class FileChunker:
    """
    Handles file chunking and reconstruction for distributed storage.
    """

    DEFAULT_CHUNK_SIZE = 256 * 1024  # 256 KB

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE):
        """
        Initialize file chunker.

        Args:
            chunk_size: Size of each chunk in bytes
        """
        self.chunk_size = chunk_size

    def chunk_file(self, file_path: str) -> Tuple[FileMetadata, List[bytes]]:
        """
        Split a file into chunks.

        Args:
            file_path: Path to the file to chunk

        Returns:
            Tuple of (file metadata, list of chunk data)
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        chunks = []
        chunk_hashes = []
        file_size = file_path.stat().st_size

        # Read and chunk file
        with open(file_path, 'rb') as f:
            while True:
                chunk_data = f.read(self.chunk_size)
                if not chunk_data:
                    break

                chunks.append(chunk_data)
                chunk_hash = hashlib.sha256(chunk_data).hexdigest()
                chunk_hashes.append(chunk_hash)

        # Calculate file hash (hash of concatenated chunk hashes)
        file_hash = hashlib.sha256(
            ''.join(chunk_hashes).encode()
        ).hexdigest()

        # Create metadata
        metadata = FileMetadata(
            file_hash=file_hash,
            file_name=file_path.name,
            file_size=file_size,
            chunk_size=self.chunk_size,
            chunk_hashes=chunk_hashes,
            chunk_count=len(chunks)
        )

        return metadata, chunks

    def chunk_stream(self, stream: BinaryIO, filename: str = "stream") -> Tuple[FileMetadata, List[bytes]]:
        """
        Chunk data from a stream.

        Args:
            stream: Binary stream to read from
            filename: Name for the file metadata

        Returns:
            Tuple of (file metadata, list of chunk data)
        """
        chunks = []
        chunk_hashes = []
        total_size = 0

        while True:
            chunk_data = stream.read(self.chunk_size)
            if not chunk_data:
                break

            chunks.append(chunk_data)
            chunk_hash = hashlib.sha256(chunk_data).hexdigest()
            chunk_hashes.append(chunk_hash)
            total_size += len(chunk_data)

        # Calculate file hash
        file_hash = hashlib.sha256(
            ''.join(chunk_hashes).encode()
        ).hexdigest()

        # Create metadata
        metadata = FileMetadata(
            file_hash=file_hash,
            file_name=filename,
            file_size=total_size,
            chunk_size=self.chunk_size,
            chunk_hashes=chunk_hashes,
            chunk_count=len(chunks)
        )

        return metadata, chunks

    def reconstruct_file(
        self,
        metadata: FileMetadata,
        chunks: List[bytes],
        output_path: Optional[str] = None
    ) -> str:
        """
        Reconstruct a file from chunks.

        Args:
            metadata: File metadata
            chunks: List of chunk data in order
            output_path: Path to save reconstructed file (optional)

        Returns:
            Path to reconstructed file

        Raises:
            ValueError: If chunks don't match metadata
        """
        # Verify chunks
        if len(chunks) != metadata.chunk_count:
            raise ValueError(
                f"Chunk count mismatch: expected {metadata.chunk_count}, "
                f"got {len(chunks)}"
            )

        # Verify chunk hashes
        for i, (chunk, expected_hash) in enumerate(zip(chunks, metadata.chunk_hashes)):
            actual_hash = hashlib.sha256(chunk).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Chunk {i} hash mismatch: expected {expected_hash}, "
                    f"got {actual_hash}"
                )

        # Determine output path
        if output_path is None:
            output_path = metadata.file_name

        output_path = Path(output_path)

        # Write reconstructed file
        with open(output_path, 'wb') as f:
            for chunk in chunks:
                f.write(chunk)

        # Verify file size
        actual_size = output_path.stat().st_size
        if actual_size != metadata.file_size:
            raise ValueError(
                f"File size mismatch: expected {metadata.file_size}, "
                f"got {actual_size}"
            )

        return str(output_path)

    def verify_chunk(self, chunk_data: bytes, expected_hash: str) -> bool:
        """
        Verify a chunk matches its expected hash.

        Args:
            chunk_data: Chunk data to verify
            expected_hash: Expected SHA256 hash

        Returns:
            True if chunk is valid, False otherwise
        """
        actual_hash = hashlib.sha256(chunk_data).hexdigest()
        return actual_hash == expected_hash

    def serialize_metadata(self, metadata: FileMetadata) -> str:
        """
        Serialize file metadata to JSON.

        Args:
            metadata: File metadata to serialize

        Returns:
            JSON string
        """
        return json.dumps(asdict(metadata), indent=2)

    def deserialize_metadata(self, json_str: str) -> FileMetadata:
        """
        Deserialize file metadata from JSON.

        Args:
            json_str: JSON string

        Returns:
            FileMetadata object
        """
        data = json.loads(json_str)
        return FileMetadata(**data)

    def calculate_redundancy_chunks(
        self,
        chunk_count: int,
        replication_factor: int = 3
    ) -> int:
        """
        Calculate number of redundant chunks needed.

        Args:
            chunk_count: Number of original chunks
            replication_factor: Desired replication factor

        Returns:
            Total number of chunk replicas needed
        """
        return chunk_count * replication_factor

    def get_chunk_distribution(
        self,
        metadata: FileMetadata,
        num_nodes: int,
        replication_factor: int = 3
    ) -> Dict[int, List[int]]:
        """
        Determine which nodes should store which chunks.

        Uses round-robin with offset for even distribution.

        Args:
            metadata: File metadata
            num_nodes: Number of storage nodes
            replication_factor: Number of replicas per chunk

        Returns:
            Dictionary mapping node_id to list of chunk indices
        """
        distribution = {i: [] for i in range(num_nodes)}

        for chunk_idx in range(metadata.chunk_count):
            # Determine which nodes get this chunk
            for replica in range(replication_factor):
                node_id = (chunk_idx + replica) % num_nodes
                distribution[node_id].append(chunk_idx)

        return distribution

    def erasure_code_chunks(
        self,
        chunks: List[bytes],
        redundancy_factor: float = 0.5
    ) -> Tuple[List[bytes], Dict]:
        """
        Apply erasure coding for fault tolerance.

        Simplified version - in production would use Reed-Solomon or similar.

        Args:
            chunks: Original chunks
            redundancy_factor: Fraction of redundant chunks to create

        Returns:
            Tuple of (all chunks including parity, erasure coding metadata)
        """
        # Simplified: XOR-based parity chunks
        num_parity = int(len(chunks) * redundancy_factor)
        parity_chunks = []

        for i in range(num_parity):
            # XOR a subset of chunks for each parity chunk
            parity_data = bytearray(len(chunks[0]))
            for j in range(i, len(chunks), num_parity + 1):
                chunk = chunks[j]
                for k in range(len(parity_data)):
                    if k < len(chunk):
                        parity_data[k] ^= chunk[k]

            parity_chunks.append(bytes(parity_data))

        # Metadata for reconstruction
        erasure_metadata = {
            'original_count': len(chunks),
            'parity_count': len(parity_chunks),
            'redundancy_factor': redundancy_factor,
            'algorithm': 'xor_parity'
        }

        return chunks + parity_chunks, erasure_metadata