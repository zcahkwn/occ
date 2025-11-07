"""
Simulation framework for testing OCC-based storage system.
"""

import os
import tempfile
import shutil
import random
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..node.storage import StorageNode
from ..files.distributor import FileDistributor, DistributionStrategy
from ..files.chunker import FileChunker
from ..lcrs.operations import LCRSOperations
from ..lcrs.statistics import LCRSStatistics


class StorageSimulator:
    """
    Simulates a distributed storage network using OCC.
    """

    def __init__(
        self,
        num_nodes: int = 10,
        n_prime: int = 10000,
        tau: int = 10,
        temp_dir: Optional[str] = None
    ):
        """
        Initialize storage simulator.

        Args:
            num_nodes: Number of storage nodes to simulate
            n_prime: LCRS sketch size
            tau: LCRS maximum counter value
            temp_dir: Directory for temporary storage (auto-created if None)
        """
        self.num_nodes = num_nodes
        self.n_prime = n_prime
        self.tau = tau

        # Create temporary directory for simulation
        if temp_dir is None:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="occ_sim_"))
            self.cleanup_on_exit = True
        else:
            self.temp_dir = Path(temp_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            self.cleanup_on_exit = False

        # Initialize storage nodes
        self.nodes: Dict[str, StorageNode] = {}
        for i in range(num_nodes):
            node_id = f"node_{i:03d}"
            node_path = self.temp_dir / node_id
            node_path.mkdir(exist_ok=True)

            self.nodes[node_id] = StorageNode(
                node_id=node_id,
                storage_path=str(node_path),
                n_prime=n_prime,
                tau=tau
            )

        # Initialize file distributor
        self.distributor = FileDistributor(self.nodes)

        # Statistics
        self.stats = LCRSStatistics(n_prime, tau)

        # Metrics tracking
        self.metrics = {
            'files_added': 0,
            'files_retrieved': 0,
            'chunks_stored': 0,
            'retrieval_times': [],
            'storage_times': [],
            'query_success_rate': [],
            'replication_factors': []
        }

    def create_test_files(
        self,
        num_files: int,
        min_size: int = 1024,
        max_size: int = 1024 * 1024
    ) -> List[Path]:
        """
        Create test files with random content.

        Args:
            num_files: Number of files to create
            min_size: Minimum file size in bytes
            max_size: Maximum file size in bytes

        Returns:
            List of created file paths
        """
        test_files = []
        test_dir = self.temp_dir / "test_files"
        test_dir.mkdir(exist_ok=True)

        for i in range(num_files):
            file_path = test_dir / f"test_file_{i:04d}.dat"
            file_size = random.randint(min_size, max_size)

            # Generate random content
            content = os.urandom(file_size)
            with open(file_path, 'wb') as f:
                f.write(content)

            test_files.append(file_path)

        return test_files

    def simulate_file_operations(
        self,
        test_files: List[Path],
        num_operations: int = 100,
        write_ratio: float = 0.3,
        replication_factor: int = 3
    ) -> Dict:
        """
        Simulate file storage and retrieval operations.

        Args:
            test_files: List of test file paths
            num_operations: Number of operations to simulate
            write_ratio: Fraction of operations that are writes (vs reads)
            replication_factor: Target replication factor

        Returns:
            Dictionary with simulation results
        """
        stored_files = {}  # Map file_path to metadata
        results = {
            'successful_writes': 0,
            'failed_writes': 0,
            'successful_reads': 0,
            'failed_reads': 0,
            'average_write_time': 0,
            'average_read_time': 0
        }

        write_times = []
        read_times = []

        for op_num in range(num_operations):
            is_write = random.random() < write_ratio or len(stored_files) == 0

            if is_write:
                # Write operation
                file_path = random.choice(test_files)

                start_time = time.time()
                try:
                    metadata, distribution = self.distributor.add_file(
                        str(file_path),
                        replication_factor=replication_factor
                    )
                    elapsed = time.time() - start_time

                    stored_files[str(file_path)] = metadata
                    results['successful_writes'] += 1
                    write_times.append(elapsed)
                    self.metrics['files_added'] += 1
                    self.metrics['storage_times'].append(elapsed)

                    # Track replication
                    avg_replication = np.mean([
                        len(nodes) for nodes in distribution.values()
                    ])
                    self.metrics['replication_factors'].append(avg_replication)

                except Exception as e:
                    print(f"Write failed: {e}")
                    results['failed_writes'] += 1

            else:
                # Read operation
                if stored_files:
                    file_path = random.choice(list(stored_files.keys()))
                    metadata = stored_files[file_path]

                    # Create output path
                    output_dir = self.temp_dir / "retrieved"
                    output_dir.mkdir(exist_ok=True)
                    output_path = output_dir / f"retrieved_{op_num:04d}.dat"

                    start_time = time.time()
                    try:
                        self.distributor.retrieve_file(
                            metadata,
                            output_path=str(output_path)
                        )
                        elapsed = time.time() - start_time

                        results['successful_reads'] += 1
                        read_times.append(elapsed)
                        self.metrics['files_retrieved'] += 1
                        self.metrics['retrieval_times'].append(elapsed)

                        # Verify retrieved file matches original
                        original_data = open(file_path, 'rb').read()
                        retrieved_data = open(output_path, 'rb').read()
                        if original_data != retrieved_data:
                            print(f"WARNING: Retrieved file doesn't match original!")

                    except Exception as e:
                        print(f"Read failed: {e}")
                        results['failed_reads'] += 1

            # Periodically print progress
            if (op_num + 1) % 10 == 0:
                print(f"Completed {op_num + 1}/{num_operations} operations")

        # Calculate averages
        if write_times:
            results['average_write_time'] = np.mean(write_times)
        if read_times:
            results['average_read_time'] = np.mean(read_times)

        return results

    def simulate_node_failures(
        self,
        failure_rate: float = 0.1
    ) -> List[str]:
        """
        Simulate random node failures.

        Args:
            failure_rate: Fraction of nodes to fail

        Returns:
            List of failed node IDs
        """
        num_failures = int(self.num_nodes * failure_rate)
        failed_nodes = random.sample(list(self.nodes.keys()), num_failures)

        for node_id in failed_nodes:
            # Remove node from distributor
            del self.nodes[node_id]

        # Reinitialize distributor with remaining nodes
        self.distributor = FileDistributor(self.nodes)

        print(f"Failed nodes: {failed_nodes}")
        return failed_nodes

    def test_probabilistic_routing(
        self,
        num_queries: int = 100
    ) -> Dict:
        """
        Test accuracy of probabilistic content routing.

        Args:
            num_queries: Number of routing queries to test

        Returns:
            Dictionary with routing test results
        """
        results = {
            'total_queries': num_queries,
            'successful_routes': 0,
            'failed_routes': 0,
            'average_hops': [],
            'false_positive_rate': 0,
            'false_negative_rate': 0
        }

        # Create and store some test chunks
        test_chunks = []
        for i in range(50):
            chunk_data = os.urandom(1024)
            chunk_hash = hashlib.sha256(chunk_data).hexdigest()
            test_chunks.append((chunk_hash, chunk_data))

            # Store on random nodes
            num_replicas = random.randint(1, min(5, self.num_nodes))
            selected_nodes = random.sample(list(self.nodes.keys()), num_replicas)
            for node_id in selected_nodes:
                self.nodes[node_id].store_chunk(chunk_data)

        # Test queries
        for _ in range(num_queries):
            chunk_hash, chunk_data = random.choice(test_chunks)

            # Find actual holders
            actual_holders = []
            for node_id, node in self.nodes.items():
                if node.has_chunk(chunk_hash):
                    actual_holders.append(node_id)

            # Use probabilistic routing
            node_sketches = {
                node_id: node.sketch
                for node_id, node in self.nodes.items()
            }
            predicted_providers = LCRSOperations.find_likely_providers(
                chunk_hash,
                node_sketches,
                top_k=5
            )

            # Check accuracy
            predicted_ids = [node_id for node_id, _ in predicted_providers]
            found = False
            hops = 0

            for node_id in predicted_ids:
                hops += 1
                if node_id in actual_holders:
                    found = True
                    break

            if found:
                results['successful_routes'] += 1
                results['average_hops'].append(hops)
            else:
                results['failed_routes'] += 1

        # Calculate rates
        if results['average_hops']:
            results['average_hops'] = np.mean(results['average_hops'])
        else:
            results['average_hops'] = float('inf')

        results['success_rate'] = (
            results['successful_routes'] / num_queries
            if num_queries > 0 else 0
        )

        return results

    def analyze_sketch_accuracy(self) -> Dict:
        """
        Analyze accuracy of LCRS sketches vs actual storage.

        Returns:
            Dictionary with accuracy analysis
        """
        sketches = [node.sketch for node in self.nodes.values()]

        # Merge all sketches
        merged = LCRSOperations.merge_multiple(sketches)

        # Get actual storage statistics
        actual_unique_chunks = set()
        for node in self.nodes.values():
            actual_unique_chunks.update(node.list_chunks())

        actual_union = len(actual_unique_chunks)
        sketch_union = merged.union_size()

        # Calculate error
        union_error = abs(actual_union - sketch_union)
        union_error_rate = union_error / max(1, actual_union)

        # Analyze level counts
        Z = merged.level_counts()
        actual_replication_dist = {}
        for chunk in actual_unique_chunks:
            count = sum(1 for node in self.nodes.values() if node.has_chunk(chunk))
            actual_replication_dist[count] = actual_replication_dist.get(count, 0) + 1

        return {
            'actual_union': actual_union,
            'sketch_union': sketch_union,
            'union_error': union_error,
            'union_error_rate': union_error_rate,
            'sketch_level_counts': Z,
            'actual_replication_distribution': actual_replication_dist,
            'sketch_statistics': self.stats.estimate_replication_distribution(Z)
        }

    def benchmark_performance(self) -> Dict:
        """
        Run performance benchmarks.

        Returns:
            Dictionary with performance metrics
        """
        results = {}

        # Benchmark sketch update time
        test_chunk = os.urandom(256 * 1024)
        chunk_hash = hashlib.sha256(test_chunk).hexdigest()

        start = time.time()
        for _ in range(100):
            self.nodes['node_000'].sketch.update([chunk_hash])
        results['sketch_update_time_ms'] = (time.time() - start) * 10  # Per update

        # Benchmark sketch merge time
        sketches = [node.sketch for node in self.nodes.values()]
        start = time.time()
        LCRSOperations.merge_multiple(sketches)
        results['sketch_merge_time_ms'] = (time.time() - start) * 1000

        # Benchmark storage operations
        start = time.time()
        self.nodes['node_000'].store_chunk(test_chunk)
        results['chunk_store_time_ms'] = (time.time() - start) * 1000

        start = time.time()
        self.nodes['node_000'].retrieve_chunk(chunk_hash)
        results['chunk_retrieve_time_ms'] = (time.time() - start) * 1000

        # Memory usage (approximate)
        import sys
        sketch_memory = sys.getsizeof(self.nodes['node_000'].sketch.counters)
        results['sketch_memory_bytes'] = sketch_memory
        results['sketch_memory_kb'] = sketch_memory / 1024

        return results

    def generate_report(self) -> str:
        """
        Generate comprehensive simulation report.

        Returns:
            Formatted report string
        """
        network_stats = self.distributor.get_network_statistics()
        accuracy_stats = self.analyze_sketch_accuracy()
        performance = self.benchmark_performance()

        report = []
        report.append("=" * 60)
        report.append("OCC STORAGE SIMULATION REPORT")
        report.append("=" * 60)

        report.append(f"\nNETWORK CONFIGURATION:")
        report.append(f"  Nodes: {self.num_nodes}")
        report.append(f"  LCRS N': {self.n_prime}")
        report.append(f"  LCRS τ: {self.tau}")

        report.append(f"\nOPERATION METRICS:")
        report.append(f"  Files added: {self.metrics['files_added']}")
        report.append(f"  Files retrieved: {self.metrics['files_retrieved']}")

        if self.metrics['storage_times']:
            report.append(f"  Avg storage time: {np.mean(self.metrics['storage_times']):.3f}s")
        if self.metrics['retrieval_times']:
            report.append(f"  Avg retrieval time: {np.mean(self.metrics['retrieval_times']):.3f}s")
        if self.metrics['replication_factors']:
            report.append(f"  Avg replication: {np.mean(self.metrics['replication_factors']):.2f}")

        report.append(f"\nNETWORK STATISTICS:")
        report.append(f"  Union size: {network_stats['union_size']}")
        report.append(f"  Total storage: {network_stats['total_storage_bytes'] / (1024**2):.2f} MB")
        report.append(f"  Total chunks: {network_stats['total_chunks_stored']}")

        report.append(f"\nSKETCH ACCURACY:")
        report.append(f"  Actual union: {accuracy_stats['actual_union']}")
        report.append(f"  Sketch union: {accuracy_stats['sketch_union']}")
        report.append(f"  Error rate: {accuracy_stats['union_error_rate']:.2%}")

        report.append(f"\nPERFORMANCE:")
        report.append(f"  Sketch update: {performance['sketch_update_time_ms']:.3f} ms")
        report.append(f"  Sketch merge: {performance['sketch_merge_time_ms']:.3f} ms")
        report.append(f"  Chunk store: {performance['chunk_store_time_ms']:.3f} ms")
        report.append(f"  Chunk retrieve: {performance['chunk_retrieve_time_ms']:.3f} ms")
        report.append(f"  Sketch memory: {performance['sketch_memory_kb']:.2f} KB")

        report.append("\n" + "=" * 60)

        return "\n".join(report)

    def cleanup(self):
        """Clean up temporary files."""
        if self.cleanup_on_exit and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up temporary directory: {self.temp_dir}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass