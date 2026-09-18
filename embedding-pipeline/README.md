# AWS GPU Embedding Pipeline & EBS ChromaDB Server

Fault-tolerant embedding pipeline for processing large multilingual JSONL datasets into 1024-dimensional `BAAI/bge-m3` vectors stored on an AWS EBS ChromaDB instance.

---

## Architecture Overview

```text
       LOCAL MACHINE                                AWS GPU EC2
+------------------------+             +----------------------------------+
| Streaming Worker       |  HTTPS POST | FastAPI App (server/server.py)   |
| - Read .jsonl (utf-8)  | ----------->| - BAAI/bge-m3 (GPU CUDA)         |
| - 128 micro-batches    |  /embed     | - Collections: kcc_docs,         |
| - Retries & Backoff    |             |                other_docs        |
| - Checkpoint Tracker   |             +----------------+-----------------+
+------------------------+                              |
                                                        v
                                             +--------------------+
                                             | EBS gp3 Volume     |
                                             | /mnt/chroma        |
                                             | ChromaDB Storage   |
                                             +--------------------+
```

---

## 1. Setting Up the Server on AWS GPU EC2

### Step A: Format & Mount EBS Volume
Attach your EBS gp3 volume to your GPU EC2 instance, then run:

```bash
# Verify volume device name (e.g., /dev/nvme1n1)
lsblk

# Format if new
sudo mkfs -t ext4 /dev/nvme1n1

# Mount to /mnt/chroma
sudo mkdir -p /mnt/chroma
sudo mount /dev/nvme1n1 /mnt/chroma

# Make writable
sudo chmod -R 777 /mnt/chroma

# Verify mount
df -h /mnt/chroma
```

To persist the mount across reboots, add to `/etc/fstab`:
```text
/dev/nvme1n1 /mnt/chroma ext4 defaults,nofail 0 2
```

### Step B: Download & Deploy FastAPI Server
Download or clone the `server/` directory onto the EC2 instance:

```bash
cd server
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

*(Note: Ensure PyTorch with CUDA support is installed for GPU acceleration)*

### Step C: Launch Server
```bash
# Set environment variables (or create .env file)
export CHROMA_PATH="/mnt/chroma"
export GPU_BATCH_SIZE=32
export DEVICE="cuda"

# Run FastAPI server with Uvicorn
python3 server.py
# Or using uvicorn CLI:
# uvicorn server.server:app --host 0.0.0.0 --port 8000 --workers 1
```

Verify server health:
```bash
curl http://localhost:8000/health
```

---

## 2. Running the Client Locally

Run the local worker to process your `.jsonl` files and stream micro-batches to the AWS EC2 instance.

### Setup
```bash
cd client
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

### Execution
```bash
# Set API URL pointing to your AWS EC2 instance public IP / DNS
export EMBEDDING_API_URL="http://<YOUR-EC2-PUBLIC-IP>:8000"

# Run streaming worker on data folder
python worker.py --input ../data
```

### Features & Protections
1. **Streaming**: Files are read line-by-line (`utf-8`) without memory bloat.
2. **Auto Collection Routing**: Items are routed automatically to `kcc_docs` or `other_docs`.
3. **Checkpoints**: Progress is updated in `checkpoints/progress.json` after successful server persistence.
4. **Idempotency**: Deterministic IDs prevent duplicate vectors upon worker restarts or retries.
5. **Graceful Shutdown**: Pressing `Ctrl+C` completes the active batch and saves state cleanly.

---

## 3. Server Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/health` | `GET` | Server status, model info, active device (`cuda`/`cpu`), and document count per collection. |
| `/stats` | `GET` | Total requests processed, total chunks embedded, average latency. |
| `/embed` | `POST` | Batch embedding payload endpoint. Encodes texts and persists in ChromaDB. |
