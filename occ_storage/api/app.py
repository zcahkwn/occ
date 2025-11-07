"""
Flask API server for OCC Storage System.
Provides REST endpoints for distributed file storage using OCC algorithms.
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from occ_storage.node.storage import StorageNode
from occ_storage.files.distributor import FileDistributor, DistributionStrategy
from occ_storage.files.chunker import FileChunker
from occ_storage.lcrs.operations import LCRSOperations

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'num_nodes': int(os.environ.get('NUM_NODES', '5')),
    'n_prime': int(os.environ.get('N_PRIME', '10000')),
    'tau': int(os.environ.get('TAU', '10')),
    'replication_factor': int(os.environ.get('REPLICATION_FACTOR', '3')),
    'chunk_size': int(os.environ.get('CHUNK_SIZE', '262144')),  # 256KB
    'storage_dir': os.environ.get('STORAGE_DIR', '/tmp/occ_storage')
}

# Global storage network
storage_network = None
metadata_store = {}  # In-memory metadata store (in production, use database)


def initialize_network():
    """Initialize the OCC storage network."""
    global storage_network

    storage_dir = Path(CONFIG['storage_dir'])
    storage_dir.mkdir(parents=True, exist_ok=True)

    # Create storage nodes
    nodes = {}
    for i in range(CONFIG['num_nodes']):
        node_id = f"node_{i:03d}"
        node_path = storage_dir / node_id
        node_path.mkdir(exist_ok=True)

        nodes[node_id] = StorageNode(
            node_id=node_id,
            storage_path=str(node_path),
            n_prime=CONFIG['n_prime'],
            tau=CONFIG['tau']
        )
        logger.info(f"Initialized {node_id}")

    # Create distributor
    strategy = DistributionStrategy(
        replication_factor=CONFIG['replication_factor'],
        use_probabilistic_routing=True,
        prefer_low_load_nodes=True
    )

    storage_network = {
        'nodes': nodes,
        'distributor': FileDistributor(nodes, strategy),
        'chunker': FileChunker(chunk_size=CONFIG['chunk_size'])
    }

    # Load existing metadata if available
    metadata_file = storage_dir / "metadata.json"
    global metadata_store
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata_store = json.load(f)
        logger.info(f"Loaded {len(metadata_store)} existing files from metadata")

    logger.info(f"OCC Storage Network initialized with {CONFIG['num_nodes']} nodes")


def save_metadata():
    """Save metadata to disk."""
    metadata_file = Path(CONFIG['storage_dir']) / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata_store, f, indent=2)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'nodes': CONFIG['num_nodes'] if storage_network else 0,
        'ready': storage_network is not None
    })


@app.route('/status', methods=['GET'])
def status():
    """Get network status."""
    if not storage_network:
        return jsonify({'error': 'Storage network not initialized'}), 503

    try:
        stats = storage_network['distributor'].get_network_statistics()

        # Get individual node stats
        node_stats = []
        for node_id, node in storage_network['nodes'].items():
            node_info = node.get_storage_stats()
            node_stats.append({
                'node_id': node_id,
                'chunks': node_info['total_chunks'],
                'size_mb': node_info['total_size_mb'],
                'union_size': node_info['sketch_union_size']
            })

        return jsonify({
            'network': {
                'total_nodes': stats['total_nodes'],
                'union_size': stats['union_size'],
                'total_chunks': stats['total_chunks_stored'],
                'total_storage_mb': stats['total_storage_bytes'] / (1024**2),
                'average_replication': stats['average_replication'],
                'level_counts': stats['level_counts']
            },
            'nodes': node_stats,
            'config': CONFIG,
            'files_stored': len(metadata_store)
        })
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload():
    """Upload a file to the OCC storage network."""
    if not storage_network:
        return jsonify({'error': 'Storage network not initialized'}), 503

    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Get optional parameters
        replication_factor = request.form.get('replication_factor',
                                              CONFIG['replication_factor'],
                                              type=int)

        # Save uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir) / secure_filename(file.filename)
        file.save(str(temp_path))

        # Add file to storage network
        metadata, distribution = storage_network['distributor'].add_file(
            str(temp_path),
            replication_factor=replication_factor
        )

        # Store metadata
        metadata_dict = {
            'file_hash': metadata.file_hash,
            'file_name': metadata.file_name,
            'file_size': metadata.file_size,
            'chunk_size': metadata.chunk_size,
            'chunk_hashes': metadata.chunk_hashes,
            'chunk_count': metadata.chunk_count
        }
        metadata_store[metadata.file_hash] = metadata_dict
        save_metadata()

        # Clean up temp file
        os.unlink(temp_path)
        os.rmdir(temp_dir)

        # Prepare response
        response = {
            'success': True,
            'file_hash': metadata.file_hash,
            'file_name': metadata.file_name,
            'file_size': metadata.file_size,
            'chunks': metadata.chunk_count,
            'replication_factor': replication_factor,
            'distribution': {
                chunk_hash[:16]: node_ids
                for chunk_hash, node_ids in distribution.items()
            }
        }

        logger.info(f"Uploaded file: {metadata.file_name} ({metadata.file_hash[:16]}...)")
        return jsonify(response), 201

    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/download/<file_hash>', methods=['GET'])
def download(file_hash):
    """Download a file from the OCC storage network."""
    if not storage_network:
        return jsonify({'error': 'Storage network not initialized'}), 503

    try:
        # Check if file exists in metadata
        if file_hash not in metadata_store:
            return jsonify({'error': 'File not found'}), 404

        # Reconstruct metadata
        metadata_dict = metadata_store[file_hash]
        from occ_storage.files.chunker import FileMetadata
        metadata = FileMetadata(**metadata_dict)

        # Create temp file for retrieval
        temp_dir = tempfile.mkdtemp()
        output_path = Path(temp_dir) / metadata.file_name

        # Retrieve file from network
        retrieved_path = storage_network['distributor'].retrieve_file(
            metadata,
            str(output_path)
        )

        logger.info(f"Downloaded file: {metadata.file_name} ({file_hash[:16]}...)")

        # Send file to client
        return send_file(
            retrieved_path,
            as_attachment=True,
            download_name=metadata.file_name,
            mimetype='application/octet-stream'
        )

    except FileNotFoundError:
        logger.error(f"File not found: {file_hash}")
        return jsonify({'error': 'Unable to retrieve all chunks'}), 404
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/files', methods=['GET'])
def list_files():
    """List all stored files."""
    if not storage_network:
        return jsonify({'error': 'Storage network not initialized'}), 503

    try:
        files = []
        for file_hash, metadata in metadata_store.items():
            # Check availability
            from occ_storage.files.chunker import FileMetadata
            file_meta = FileMetadata(**metadata)
            availability = storage_network['distributor'].verify_file_availability(file_meta)

            files.append({
                'file_hash': file_hash,
                'file_name': metadata['file_name'],
                'file_size': metadata['file_size'],
                'chunks': metadata['chunk_count'],
                'retrievable': availability['retrievable'],
                'min_replication': availability['min_replication'],
                'avg_replication': availability['avg_replication']
            })

        return jsonify({
            'total_files': len(files),
            'files': files
        })

    except Exception as e:
        logger.error(f"Error listing files: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/verify/<file_hash>', methods=['GET'])
def verify(file_hash):
    """Verify file availability and replication status."""
    if not storage_network:
        return jsonify({'error': 'Storage network not initialized'}), 503

    try:
        if file_hash not in metadata_store:
            return jsonify({'error': 'File not found'}), 404

        # Reconstruct metadata
        from occ_storage.files.chunker import FileMetadata
        metadata = FileMetadata(**metadata_store[file_hash])

        # Check availability
        availability = storage_network['distributor'].verify_file_availability(metadata)

        # Get chunk distribution details
        chunk_details = []
        for idx, chunk_hash in enumerate(metadata.chunk_hashes):
            holders = []
            for node_id, node in storage_network['nodes'].items():
                if node.has_chunk(chunk_hash):
                    holders.append(node_id)

            chunk_details.append({
                'index': idx,
                'hash': chunk_hash[:16],
                'replicas': len(holders),
                'nodes': holders
            })

        return jsonify({
            'file_hash': file_hash,
            'file_name': metadata.file_name,
            'retrievable': availability['retrievable'],
            'missing_chunks': availability['missing_chunks'],
            'replication': {
                'min': availability['min_replication'],
                'avg': availability['avg_replication'],
                'max': availability['max_replication']
            },
            'chunks': chunk_details
        })

    except Exception as e:
        logger.error(f"Error verifying file: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/rebalance/<file_hash>', methods=['POST'])
def rebalance(file_hash):
    """Rebalance file chunks to meet replication targets."""
    if not storage_network:
        return jsonify({'error': 'Storage network not initialized'}), 503

    try:
        if file_hash not in metadata_store:
            return jsonify({'error': 'File not found'}), 404

        # Get target replication from request
        target_replication = request.json.get('replication_factor',
                                               CONFIG['replication_factor'])

        # Reconstruct metadata
        from occ_storage.files.chunker import FileMetadata
        metadata = FileMetadata(**metadata_store[file_hash])

        # Rebalance file
        new_distribution = storage_network['distributor'].rebalance_file(
            metadata,
            target_replication
        )

        return jsonify({
            'success': True,
            'file_hash': file_hash,
            'target_replication': target_replication,
            'new_distribution': {
                chunk_hash[:16]: node_ids
                for chunk_hash, node_ids in new_distribution.items()
            }
        })

    except Exception as e:
        logger.error(f"Error rebalancing file: {e}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Initialize network on startup
    initialize_network()

    # Run Flask app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'

    logger.info(f"Starting OCC Storage API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)