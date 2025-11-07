# 🚀 OCC Storage API - Quick Start Guide

## 1-Minute Setup

```bash
# Build and run with Docker Compose
docker-compose up -d

# Upload a file
curl -X POST -F "file=@your-file.pdf" http://localhost:5000/upload

# Download it back (use hash from upload response)
curl -O http://localhost:5000/download/FILE_HASH

# Check status
curl http://localhost:5000/status | jq
```

That's it! You now have a distributed file storage API running locally.

## What You Just Built

A REST API that provides:
- **Upload endpoint** (`POST /upload`) - Store files across distributed nodes
- **Download endpoint** (`GET /download/<hash>`) - Retrieve files using content hash
- **Privacy preserving** - Uses LCRS sketches instead of exact tracking
- **Fault tolerant** - Automatic replication across nodes
- **Space efficient** - Probabilistic tracking uses ~10KB per node vs MB for exact

## Test with Example Image

```bash
# Start the API (if not already running)
docker-compose up -d

# Upload the Exponential Science logo
curl -X POST \
  -F "file=@occ_storage/exp.jpg" \
  -F "replication_factor=3" \
  http://localhost:5000/upload

# Response will include file_hash like:
# {
#   "file_hash": "518d7f295016...",
#   "file_name": "exp.jpg",
#   ...
# }

# Download it back
curl -o retrieved_logo.jpg \
  http://localhost:5000/download/518d7f295016...

# Verify it worked
open retrieved_logo.jpg  # macOS
# or
xdg-open retrieved_logo.jpg  # Linux
```

## Python Example

```python
import requests

# Upload
with open('document.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/upload',
        files={'file': f}
    )
    file_hash = response.json()['file_hash']
    print(f"Uploaded: {file_hash}")

# Download
response = requests.get(f'http://localhost:5000/download/{file_hash}')
with open('downloaded.pdf', 'wb') as f:
    f.write(response.content)
    print("Downloaded successfully!")
```

## JavaScript Example

```javascript
// Upload
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:5000/upload', {
    method: 'POST',
    body: formData
})
.then(r => r.json())
.then(data => {
    console.log('File hash:', data.file_hash);

    // Download
    return fetch(`http://localhost:5000/download/${data.file_hash}`);
})
.then(r => r.blob())
.then(blob => {
    // Create download link
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'file.pdf';
    a.click();
});
```

## Available Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/status` | Network statistics |
| POST | `/upload` | Upload a file |
| GET | `/download/<hash>` | Download a file |
| GET | `/files` | List all files |
| GET | `/verify/<hash>` | Verify file availability |
| POST | `/rebalance/<hash>` | Rebalance replication |

## Configuration Options

Edit `docker-compose.yml` to customize:

```yaml
environment:
  - NUM_NODES=10          # More nodes for better distribution
  - N_PRIME=50000         # Larger sketch for more accuracy
  - TAU=15                # Higher max replication
  - REPLICATION_FACTOR=5  # More copies of each chunk
```

## Running Locally (without Docker)

```bash
# Install dependencies
pip install flask flask-cors numpy scipy pandas joblib

# Run the API
PORT=5555 NUM_NODES=5 python -m occ_storage.api.app

# Test it
curl http://localhost:5555/health
```

## Monitoring

```bash
# Watch logs
docker-compose logs -f

# Check container health
docker ps

# Get detailed stats
curl http://localhost:5000/status | python -m json.tool

# See storage usage
docker exec occ-storage-api df -h /data
```

## Troubleshooting

### Port 5000 in use (macOS)
```bash
# Use different port
sed -i '' 's/5000:5000/8080:8080/g' docker-compose.yml
docker-compose up -d
# Now use http://localhost:8080
```

### Build fails
```bash
docker-compose build --no-cache
```

### Upload fails
```bash
# Check logs
docker-compose logs occ-storage

# Ensure file exists
ls -la your-file.pdf
```

## Performance Tips

- **Increase nodes**: More nodes = better distribution
- **Tune chunk size**: Larger chunks for big files
- **Adjust replication**: Balance redundancy vs storage
- **Use SSD**: Faster I/O for chunks

## Next Steps

1. **Test the API**: Run `python test_api.py`
2. **Deploy to production**: Add HTTPS, authentication
3. **Scale horizontally**: Run multiple API instances
4. **Monitor usage**: Track metrics and performance

---

**That's it!** You now have a working distributed file storage API using the patented OCC algorithms. The system provides IPFS-like functionality with better privacy and efficiency through probabilistic tracking.

Questions? Check the full documentation in `DOCKER_API_README.md`