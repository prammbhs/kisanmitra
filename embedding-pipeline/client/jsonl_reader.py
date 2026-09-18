import json
import os
import hashlib
from typing import Generator, Dict, Any

def get_target_collection(file_path: str, record_metadata: Dict[str, Any] = None) -> str:
    """
    Determine target collection: kcc_docs or other_docs
    """
    if record_metadata and "source" in record_metadata:
        src = str(record_metadata["source"]).lower()
        if "kcc" in src:
            return "kcc_docs"
    
    clean_path = file_path.lower().replace("\\", "/")
    if "kcc" in clean_path:
        return "kcc_docs"
    
    return "other_docs"

def generate_deterministic_id(file_path: str, line_number: int, text: str) -> str:
    """
    Generate a deterministic ID based on file path, line number, and text content hash.
    """
    rel_name = os.path.basename(file_path)
    content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return f"{rel_name}_{line_number:06d}_{content_hash}"

def stream_jsonl(file_path: str, start_line: int = 0) -> Generator[Dict[str, Any], None, None]:
    """
    Stream records from a .jsonl file line by line with utf-8 encoding.
    Skips the first `start_line` records.
    """
    rel_file = os.path.basename(file_path)
    collection_name = get_target_collection(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        for current_line, line in enumerate(f, start=1):
            if current_line <= start_line:
                continue

            line_str = line.strip()
            if not line_str:
                continue

            try:
                record = json.loads(line_str)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping malformed JSON line {current_line} in {rel_file}: {e}")
                continue

            # Extract text
            text = record.get("text", "")
            if not text or not isinstance(text, str):
                print(f"[WARN] Skipping line {current_line} in {rel_file}: missing or invalid 'text'")
                continue

            # Extract or generate ID
            chunk_id = record.get("id")
            if not chunk_id:
                chunk_id = generate_deterministic_id(file_path, current_line, text)

            # Metadata preservation and augmentation
            metadata = record.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {"raw_metadata": str(metadata)}

            metadata["source_file"] = rel_file
            if "kcc" in rel_file.lower() or collection_name == "kcc_docs":
                metadata.setdefault("source_type", "KCC")
            else:
                metadata.setdefault("source_type", "BOOK")

            # Determine collection for item if metadata overrides
            item_collection = get_target_collection(file_path, metadata)

            yield {
                "id": str(chunk_id),
                "text": text,
                "metadata": metadata,
                "collection_name": item_collection,
                "line_number": current_line
            }
