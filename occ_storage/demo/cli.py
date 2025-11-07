#!/usr/bin/env python3
"""
CLI demo for OCC-based distributed storage system.
"""

import argparse
import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from occ_storage.node.storage import StorageNode
from occ_storage.files.distributor import FileDistributor, DistributionStrategy
from occ_storage.files.chunker import FileChunker
from occ_storage.simulation.simulator import StorageSimulator
from occ_storage.lcrs.operations import LCRSOperations


class OCCStorageCLI:
    """Command-line interface for OCC storage system."""

    def __init__(self, storage_dir: str = None):
        """Initialize CLI with storage directory."""
        if storage_dir is None:
            self.storage_dir = Path(tempfile.mkdtemp(prefix="occ_storage_"))
        else:
            self.storage_dir = Path(storage_dir)
            self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.config_file = self.storage_dir / "config.json"
        self.nodes = {}
        self.distributor = None

        # Load existing configuration if available
        if self.config_file.exists():
            self._load_config()
        else:
            self._init_config()

    def _init_config(self):
        """Initialize default configuration."""
        self.config = {
            'n_prime': 10000,
            'tau': 10,
            'chunk_size': 256 * 1024,  # 256 KB
            'replication_factor': 3,
            'nodes': {}
        }
        self._save_config()

    def _load_config(self):
        """Load configuration from file."""
        with open(self.config_file, 'r') as f:
            self.config = json.load(f)

        # Load nodes
        for node_id, node_path in self.config['nodes'].items():
            self.nodes[node_id] = StorageNode(
                node_id=node_id,
                storage_path=node_path,
                n_prime=self.config['n_prime'],
                tau=self.config['tau']
            )

        if self.nodes:
            self.distributor = FileDistributor(self.nodes)

    def _save_config(self):
        """Save configuration to file."""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)

    def cmd_init(self, num_nodes: int):
        """Initialize storage network with specified number of nodes."""
        print(f"Initializing OCC storage network with {num_nodes} nodes...")

        for i in range(num_nodes):
            node_id = f"node_{i:03d}"
            node_path = self.storage_dir / node_id

            self.nodes[node_id] = StorageNode(
                node_id=node_id,
                storage_path=str(node_path),
                n_prime=self.config['n_prime'],
                tau=self.config['tau']
            )

            self.config['nodes'][node_id] = str(node_path)

        self.distributor = FileDistributor(self.nodes)
        self._save_config()

        print(f"✓ Initialized {num_nodes} storage nodes")
        print(f"✓ Storage directory: {self.storage_dir}")
        print(f"✓ LCRS parameters: N'={self.config['n_prime']}, τ={self.config['tau']}")

    def cmd_add(self, file_path: str, replication: Optional[int] = None):
        """Add a file to the storage network."""
        if not self.nodes:
            print("Error: No storage nodes initialized. Run 'init' first.")
            return

        file_path = Path(file_path)
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            return

        if replication is None:
            replication = self.config['replication_factor']

        print(f"Adding file: {file_path.name}")
        print(f"File size: {file_path.stat().st_size / 1024:.2f} KB")
        print(f"Replication factor: {replication}")

        try:
            metadata, distribution = self.distributor.add_file(
                str(file_path),
                replication_factor=replication
            )

            # Save metadata
            metadata_file = self.storage_dir / "metadata" / f"{metadata.file_hash}.json"
            metadata_file.parent.mkdir(exist_ok=True)

            chunker = FileChunker()
            with open(metadata_file, 'w') as f:
                f.write(chunker.serialize_metadata(metadata))

            print(f"\n✓ File added successfully!")
            print(f"  File hash: {metadata.file_hash}")
            print(f"  Chunks: {metadata.chunk_count}")
            print(f"  Distribution:")

            for chunk_hash, node_ids in distribution.items():
                print(f"    {chunk_hash[:8]}... → {', '.join(node_ids)}")

            # Show network statistics
            stats = self.distributor.get_network_statistics()
            print(f"\n  Network statistics:")
            print(f"    Union size: {stats['union_size']}")
            print(f"    Avg replication: {stats['average_replication']:.2f}")

        except Exception as e:
            print(f"Error adding file: {e}")

    def cmd_get(self, file_hash: str, output_path: Optional[str] = None):
        """Retrieve a file from the storage network."""
        if not self.nodes:
            print("Error: No storage nodes initialized.")
            return

        # Load metadata
        metadata_file = self.storage_dir / "metadata" / f"{file_hash}.json"
        if not metadata_file.exists():
            print(f"Error: File metadata not found for hash: {file_hash}")
            return

        chunker = FileChunker()
        with open(metadata_file, 'r') as f:
            metadata = chunker.deserialize_metadata(f.read())

        print(f"Retrieving file: {metadata.file_name}")
        print(f"File size: {metadata.file_size / 1024:.2f} KB")
        print(f"Chunks: {metadata.chunk_count}")

        try:
            output_file = self.distributor.retrieve_file(metadata, output_path)
            print(f"\n✓ File retrieved successfully!")
            print(f"  Saved to: {output_file}")

        except Exception as e:
            print(f"Error retrieving file: {e}")

    def cmd_status(self):
        """Show storage network status."""
        if not self.nodes:
            print("No storage nodes initialized.")
            return

        print("OCC Storage Network Status")
        print("=" * 60)

        # Node statistics
        print(f"\nNodes ({len(self.nodes)}):")
        for node_id, node in self.nodes.items():
            stats = node.get_storage_stats()
            print(f"  {node_id}:")
            print(f"    Chunks: {stats['total_chunks']}")
            print(f"    Size: {stats['total_size_mb']:.2f} MB")
            print(f"    Union size: {stats['sketch_union_size']}")

        # Network statistics
        if self.distributor:
            network_stats = self.distributor.get_network_statistics()
            print(f"\nNetwork totals:")
            print(f"  Total chunks: {network_stats['total_chunks_stored']}")
            print(f"  Total storage: {network_stats['total_storage_bytes'] / (1024**2):.2f} MB")
            print(f"  Union size: {network_stats['union_size']}")
            print(f"  Average replication: {network_stats['average_replication']:.2f}")

            # Level counts
            Z = network_stats['level_counts']
            print(f"\n  Replication distribution:")
            for level, count in enumerate(Z):
                if count > 0 and level > 0:
                    print(f"    {level} replicas: {count} items")

            # Balance statistics
            if 'balance' in network_stats:
                balance = network_stats['balance']
                print(f"\n  Load balance:")
                print(f"    Mean load: {balance['mean_load']:.1f}")
                print(f"    Std dev: {balance['std_load']:.1f}")
                print(f"    Imbalance ratio: {balance['imbalance_ratio']:.2f}")

    def cmd_list(self):
        """List all files in the network."""
        metadata_dir = self.storage_dir / "metadata"
        if not metadata_dir.exists():
            print("No files stored.")
            return

        print("Stored Files")
        print("=" * 60)

        chunker = FileChunker()
        for metadata_file in metadata_dir.glob("*.json"):
            with open(metadata_file, 'r') as f:
                metadata = chunker.deserialize_metadata(f.read())

            # Check availability
            availability = self.distributor.verify_file_availability(metadata)

            status = "✓" if availability['retrievable'] else "✗"
            print(f"{status} {metadata.file_hash[:12]}... {metadata.file_name}")
            print(f"    Size: {metadata.file_size / 1024:.2f} KB")
            print(f"    Chunks: {metadata.chunk_count}")
            print(f"    Replication: min={availability['min_replication']}, "
                  f"avg={availability['avg_replication']:.1f}, "
                  f"max={availability['max_replication']}")

    def cmd_simulate(self, num_files: int, num_operations: int):
        """Run simulation with synthetic data."""
        print(f"Running simulation...")
        print(f"  Files: {num_files}")
        print(f"  Operations: {num_operations}")

        simulator = StorageSimulator(
            num_nodes=len(self.nodes) if self.nodes else 10,
            n_prime=self.config['n_prime'],
            tau=self.config['tau'],
            temp_dir=str(self.storage_dir / "simulation")
        )

        # Create test files
        print("\nCreating test files...")
        test_files = simulator.create_test_files(num_files)

        # Run operations
        print("\nSimulating operations...")
        results = simulator.simulate_file_operations(
            test_files,
            num_operations=num_operations,
            replication_factor=self.config['replication_factor']
        )

        # Test routing
        print("\nTesting probabilistic routing...")
        routing_results = simulator.test_probabilistic_routing()

        # Generate report
        print("\n" + simulator.generate_report())

        print(f"\nSimulation results:")
        print(f"  Successful writes: {results['successful_writes']}")
        print(f"  Failed writes: {results['failed_writes']}")
        print(f"  Successful reads: {results['successful_reads']}")
        print(f"  Failed reads: {results['failed_reads']}")
        print(f"  Avg write time: {results['average_write_time']:.3f}s")
        print(f"  Avg read time: {results['average_read_time']:.3f}s")
        print(f"  Routing success rate: {routing_results['success_rate']:.2%}")

        # Cleanup
        simulator.cleanup()

    def cmd_verify(self, file_hash: str):
        """Verify file availability and integrity."""
        if not self.nodes:
            print("Error: No storage nodes initialized.")
            return

        # Load metadata
        metadata_file = self.storage_dir / "metadata" / f"{file_hash}.json"
        if not metadata_file.exists():
            print(f"Error: File metadata not found for hash: {file_hash}")
            return

        chunker = FileChunker()
        with open(metadata_file, 'r') as f:
            metadata = chunker.deserialize_metadata(f.read())

        print(f"Verifying file: {metadata.file_name}")

        availability = self.distributor.verify_file_availability(metadata)

        print(f"\nAvailability:")
        print(f"  Retrievable: {'✓ Yes' if availability['retrievable'] else '✗ No'}")
        print(f"  Missing chunks: {len(availability['missing_chunks'])}")

        if availability['missing_chunks']:
            print(f"    Missing: {availability['missing_chunks']}")

        print(f"\nReplication statistics:")
        print(f"  Minimum: {availability['min_replication']}")
        print(f"  Average: {availability['avg_replication']:.2f}")
        print(f"  Maximum: {availability['max_replication']}")

        print(f"\nChunk distribution:")
        for chunk_hash, node_ids in availability['chunk_availability'].items():
            print(f"  {chunk_hash[:8]}... → {len(node_ids)} replicas")
            if len(node_ids) < self.config['replication_factor']:
                print(f"    ⚠ Below target replication factor!")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="OCC-based distributed file storage system"
    )
    parser.add_argument(
        '--storage-dir',
        help='Storage directory (default: temp directory)',
        default=None
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize storage network')
    init_parser.add_argument(
        'nodes',
        type=int,
        help='Number of storage nodes'
    )

    # Add command
    add_parser = subparsers.add_parser('add', help='Add file to network')
    add_parser.add_argument('file', help='File path')
    add_parser.add_argument(
        '--replication',
        type=int,
        help='Replication factor (default: 3)'
    )

    # Get command
    get_parser = subparsers.add_parser('get', help='Retrieve file from network')
    get_parser.add_argument('hash', help='File hash')
    get_parser.add_argument(
        '--output',
        help='Output file path (default: original name)'
    )

    # Status command
    status_parser = subparsers.add_parser('status', help='Show network status')

    # List command
    list_parser = subparsers.add_parser('list', help='List stored files')

    # Simulate command
    sim_parser = subparsers.add_parser('simulate', help='Run simulation')
    sim_parser.add_argument(
        '--files',
        type=int,
        default=10,
        help='Number of test files'
    )
    sim_parser.add_argument(
        '--operations',
        type=int,
        default=50,
        help='Number of operations'
    )

    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify file availability')
    verify_parser.add_argument('hash', help='File hash')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize CLI
    cli = OCCStorageCLI(args.storage_dir)

    # Execute command
    if args.command == 'init':
        cli.cmd_init(args.nodes)
    elif args.command == 'add':
        cli.cmd_add(args.file, args.replication)
    elif args.command == 'get':
        cli.cmd_get(args.hash, args.output)
    elif args.command == 'status':
        cli.cmd_status()
    elif args.command == 'list':
        cli.cmd_list()
    elif args.command == 'simulate':
        cli.cmd_simulate(args.files, args.operations)
    elif args.command == 'verify':
        cli.cmd_verify(args.hash)


if __name__ == '__main__':
    main()