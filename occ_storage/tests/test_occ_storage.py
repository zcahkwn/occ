#!/usr/bin/env python3
"""
Tests for OCC-based storage system.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from occ_storage.lcrs.sketch import LCRS, LCRSConfig
from occ_storage.node.storage import StorageNode
from occ_storage.files.chunker import FileChunker
from occ_storage.files.distributor import FileDistributor
from occ_storage.simulation.simulator import StorageSimulator


def test_lcrs_basic():
    """Test basic LCRS operations."""
    print("\n=== Testing LCRS Basic Operations ===")

    config = LCRSConfig(n_prime=1000, tau=5, node_id="test_node")
    sketch = LCRS(config)

    # Test update
    chunks = ["chunk1", "chunk2", "chunk3"]
    sketch.update(chunks)

    union_size = sketch.union_size()
    print(f"✓ Union size after adding 3 chunks: {union_size}")

    # Test level counts
    Z = sketch.level_counts()
    print(f"✓ Level counts: {Z}")

    # Test merge
    config2 = LCRSConfig(n_prime=1000, tau=5, node_id="test_node2")
    sketch2 = LCRS(config2)
    sketch2.update(["chunk2", "chunk3", "chunk4"])

    merged = sketch.merge(sketch2)
    print(f"✓ Merged union size: {merged.union_size()}")

    return True


def test_storage_node():
    """Test storage node operations."""
    print("\n=== Testing Storage Node ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        node = StorageNode(
            node_id="test_node",
            storage_path=temp_dir,
            n_prime=1000,
            tau=5
        )

        # Store chunk
        test_data = b"Hello, OCC Storage!"
        chunk_hash = node.store_chunk(test_data)
        print(f"✓ Stored chunk: {chunk_hash[:12]}...")

        # Retrieve chunk
        retrieved = node.retrieve_chunk(chunk_hash)
        assert retrieved == test_data
        print(f"✓ Retrieved chunk successfully")

        # Check statistics
        stats = node.get_storage_stats()
        print(f"✓ Node statistics: {stats['total_chunks']} chunks, "
              f"{stats['total_size_bytes']} bytes")

        return True


def test_file_chunking():
    """Test file chunking and reconstruction."""
    print("\n=== Testing File Chunking ===")

    chunker = FileChunker(chunk_size=1024)  # 1KB chunks for testing

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        # Create test file
        test_content = b"A" * 2500  # 2.5KB file
        temp_file.write(test_content)
        temp_file.flush()

        # Chunk file
        metadata, chunks = chunker.chunk_file(temp_file.name)
        print(f"✓ Chunked file into {len(chunks)} chunks")

        # Verify chunks
        assert len(chunks) == 3  # Should be 3 chunks (2.5KB / 1KB)
        assert metadata.chunk_count == 3
        print(f"✓ File metadata: {metadata.file_hash[:12]}...")

        # Reconstruct file
        output_file = temp_file.name + ".reconstructed"
        chunker.reconstruct_file(metadata, chunks, output_file)

        # Verify reconstruction
        with open(output_file, 'rb') as f:
            reconstructed = f.read()
        assert reconstructed == test_content
        print(f"✓ File reconstructed successfully")

        # Cleanup
        os.unlink(temp_file.name)
        os.unlink(output_file)

    return True


def test_distributed_storage():
    """Test distributed file storage and retrieval."""
    print("\n=== Testing Distributed Storage ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create storage nodes
        nodes = {}
        for i in range(5):
            node_id = f"node_{i}"
            node_path = temp_path / node_id
            nodes[node_id] = StorageNode(
                node_id=node_id,
                storage_path=str(node_path),
                n_prime=1000,
                tau=10
            )

        # Create distributor
        distributor = FileDistributor(nodes)

        # Create test file
        test_file = temp_path / "test.dat"
        test_content = os.urandom(10 * 1024)  # 10KB random data
        with open(test_file, 'wb') as f:
            f.write(test_content)

        # Add file to network
        metadata, distribution = distributor.add_file(
            str(test_file),
            replication_factor=3
        )
        print(f"✓ Added file with {metadata.chunk_count} chunks")

        # Verify distribution
        for chunk_hash, node_ids in distribution.items():
            assert len(node_ids) >= 1  # At least one replica
            print(f"  Chunk {chunk_hash[:8]}... → {len(node_ids)} replicas")

        # Retrieve file
        output_file = temp_path / "retrieved.dat"
        distributor.retrieve_file(metadata, str(output_file))

        # Verify retrieved file
        with open(output_file, 'rb') as f:
            retrieved_content = f.read()
        assert retrieved_content == test_content
        print(f"✓ File retrieved and verified successfully")

        # Get network statistics
        stats = distributor.get_network_statistics()
        print(f"✓ Network stats: union_size={stats['union_size']}, "
              f"total_nodes={stats['total_nodes']}")

    return True


def test_simulation():
    """Test simulation framework."""
    print("\n=== Testing Simulation Framework ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create simulator
        simulator = StorageSimulator(
            num_nodes=5,
            n_prime=1000,
            tau=5,
            temp_dir=temp_dir
        )

        # Create test files
        test_files = simulator.create_test_files(num_files=5, min_size=1024, max_size=5120)
        print(f"✓ Created {len(test_files)} test files")

        # Run operations
        results = simulator.simulate_file_operations(
            test_files,
            num_operations=20,
            write_ratio=0.5,
            replication_factor=2
        )
        print(f"✓ Simulation: {results['successful_writes']} writes, "
              f"{results['successful_reads']} reads")

        # Test probabilistic routing
        routing_results = simulator.test_probabilistic_routing(num_queries=10)
        print(f"✓ Routing success rate: {routing_results['success_rate']:.1%}")

        # Analyze sketch accuracy
        accuracy = simulator.analyze_sketch_accuracy()
        print(f"✓ Sketch accuracy: actual={accuracy['actual_union']}, "
              f"sketch={accuracy['sketch_union']}, "
              f"error={accuracy['union_error_rate']:.2%}")

        # Benchmark performance
        performance = simulator.benchmark_performance()
        print(f"✓ Performance: update={performance['sketch_update_time_ms']:.3f}ms, "
              f"memory={performance['sketch_memory_kb']:.2f}KB")

        # Generate report
        report = simulator.generate_report()
        print("\n" + "=" * 60)
        print("SIMULATION REPORT PREVIEW:")
        print("=" * 60)
        print("\n".join(report.split("\n")[:20]))  # First 20 lines
        print("...")

        # Cleanup
        simulator.cleanup()

    return True


def test_probabilistic_routing():
    """Test probabilistic content discovery."""
    print("\n=== Testing Probabilistic Routing ===")

    from occ_storage.lcrs.operations import LCRSOperations

    # Create sketches
    sketches = {}
    for i in range(5):
        config = LCRSConfig(n_prime=1000, tau=5, node_id=f"node_{i}")
        sketch = LCRS(config)

        # Add some chunks
        chunks = [f"chunk_{j}" for j in range(i*3, (i+1)*3)]
        sketch.update(chunks)
        sketches[f"node_{i}"] = sketch

    # Find providers for a specific chunk
    providers = LCRSOperations.find_likely_providers(
        "chunk_5",  # This should be in node_1
        sketches,
        top_k=3
    )

    print(f"✓ Likely providers for chunk_5:")
    for node_id, prob in providers:
        print(f"  {node_id}: {prob:.2f}")

    # Compute network statistics
    stats = LCRSOperations.compute_network_statistics(list(sketches.values()))
    print(f"✓ Network statistics: union={stats['union_size']}, "
          f"avg_replication={stats['average_replication']:.2f}")

    return True


def main():
    """Run all tests."""
    print("OCC Storage System Test Suite")
    print("=" * 60)

    tests = [
        ("LCRS Basic Operations", test_lcrs_basic),
        ("Storage Node", test_storage_node),
        ("File Chunking", test_file_chunking),
        ("Distributed Storage", test_distributed_storage),
        ("Probabilistic Routing", test_probabilistic_routing),
        ("Simulation Framework", test_simulation),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED\n")
            else:
                failed += 1
                print(f"✗ {test_name} FAILED\n")
        except Exception as e:
            failed += 1
            print(f"✗ {test_name} FAILED: {e}\n")

    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("\n🎉 All tests passed! The OCC storage system is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())