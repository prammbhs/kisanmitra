import argparse
import glob
import os
import sys
import signal
import time
from typing import List, Dict, Any

# Adjust path if run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from client.config import HTTP_BATCH_SIZE, EMBEDDING_API_URL
    from client.jsonl_reader import stream_jsonl
    from client.retry import EmbeddingAPIClient
    from client.checkpoint import CheckpointManager
except ImportError:
    from config import HTTP_BATCH_SIZE, EMBEDDING_API_URL
    from jsonl_reader import stream_jsonl
    from retry import EmbeddingAPIClient
    from checkpoint import CheckpointManager

# Graceful shutdown handler flag
shutdown_requested = False

def handle_sigint(signum, frame):
    global shutdown_requested
    shutdown_requested = True

signal.signal(signal.SIGINT, handle_sigint)

def count_file_lines(filepath: str) -> int:
    lines = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for _ in f:
            lines += 1
    return lines

def process_file(
    filepath: str,
    base_dir: str,
    client: EmbeddingAPIClient,
    checkpoint: CheckpointManager,
    batch_size: int,
):
    global shutdown_requested
    rel_path = os.path.relpath(filepath, base_dir).replace("\\", "/")
    completed_lines = checkpoint.get_completed_lines(rel_path)

    total_lines = count_file_lines(filepath)
    if completed_lines >= total_lines and total_lines > 0:
        print(f"[INFO] Skipping {rel_path} (Already completed {completed_lines}/{total_lines} lines)")
        return

    print(f"\n[INFO] Processing {rel_path} (Resuming from line {completed_lines}/{total_lines})")

    batch: List[Dict[str, Any]] = []
    current_collection = None
    last_line = completed_lines

    for item in stream_jsonl(filepath, start_line=completed_lines):
        if shutdown_requested:
            print("\n[INFO] Shutdown flag set. Stopping processing loop...")
            break

        # Group by target collection if target changes mid-batch
        item_collection = item["collection_name"]
        if current_collection is None:
            current_collection = item_collection
        elif current_collection != item_collection and batch:
            # Flush current batch if collection changes
            _flush_batch(client, batch, current_collection, rel_path, checkpoint, last_line)
            batch = []
            current_collection = item_collection

        batch.append(item)
        last_line = item["line_number"]

        if len(batch) >= batch_size:
            _flush_batch(client, batch, current_collection, rel_path, checkpoint, last_line)
            batch = []

    # Flush any remaining items in final batch
    if batch and not shutdown_requested:
        _flush_batch(client, batch, current_collection, rel_path, checkpoint, last_line)

    if not shutdown_requested and last_line >= total_lines:
        print(f"[SUCCESS] Completed processing {rel_path} ({total_lines}/{total_lines} lines)")

def _flush_batch(
    client: EmbeddingAPIClient,
    batch: List[Dict[str, Any]],
    collection_name: str,
    rel_path: str,
    checkpoint: CheckpointManager,
    last_line: int,
):
    start_time = time.time()
    num_items = len(batch)
    print(f"[INFO] Sending micro-batch of {num_items} chunks (lines {batch[0]['line_number']}-{batch[-1]['line_number']}) to collection '{collection_name}'...")

    response = client.send_micro_batch(batch, collection_name)
    elapsed = time.time() - start_time

    if response.get("status") != "success":
        raise RuntimeError(f"Server returned non-success response: {response}")

    print(f"[INFO] Batch processed in {elapsed:.2f}s ({num_items / elapsed:.1f} chunks/sec). ChromaDB updated.")
    
    # Checkpoint ONLY after successful server storage confirmation
    checkpoint.update_checkpoint(rel_path, last_line)

from concurrent.futures import ThreadPoolExecutor

def main():
    parser = argparse.ArgumentParser(description="Streaming JSONL Embedding Worker")
    parser.add_argument("--input", "-i", required=True, help="Input directory containing .jsonl files")
    parser.add_argument("--batch-size", "-b", type=int, default=HTTP_BATCH_SIZE, help="HTTP micro-batch size")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Number of parallel worker threads")
    parser.add_argument("--url", "-u", default=None, help="Embedding Server API URL")
    parser.add_argument("--reset", action="store_true", help="Reset checkpoint progress and start fresh")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input)
    if not os.path.exists(input_dir):
        print(f"[ERROR] Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    api_url = args.url or EMBEDDING_API_URL
    print(f"[INIT] Embedding Client Target URL: {api_url}")
    print(f"[INIT] Micro-batch size: {args.batch_size}")
    print(f"[INIT] Parallel Workers: {args.workers}")

    checkpoint = CheckpointManager()
    if args.reset:
        checkpoint.reset()

    client = EmbeddingAPIClient(api_url=api_url)

    # Recursively find all .jsonl files
    jsonl_files = sorted(glob.glob(os.path.join(input_dir, "**", "*.jsonl"), recursive=True))
    if not jsonl_files:
        print(f"[WARN] No .jsonl files found in {input_dir}")
        sys.exit(0)

    print(f"[INFO] Found {len(jsonl_files)} JSONL files to process.")

    if args.workers > 1:
        print(f"[INFO] Launching ThreadPoolExecutor with {args.workers} workers...")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(process_file, file_path, input_dir, client, checkpoint, args.batch_size)
                for file_path in jsonl_files
            ]
            for future in futures:
                if shutdown_requested:
                    break
                try:
                    future.result()
                except Exception as e:
                    print(f"[ERROR] Worker task encountered exception: {e}")
    else:
        for file_path in jsonl_files:
            if shutdown_requested:
                print("[INFO] Exiting main loop due to shutdown request.")
                break
            process_file(file_path, input_dir, client, checkpoint, args.batch_size)

    print("\n[INFO] Worker task finished.")

if __name__ == "__main__":
    main()
