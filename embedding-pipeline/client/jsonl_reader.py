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

def generate_deterministic_id(file_path: str, line_number: int, text: str, record: Dict[str, Any] = None) -> str:
    """
    Generate a deterministic ID based on metadata ID, KCCCallID, or file path + line hash.
    """
    if record:
        metadata = record.get("metadata", {})
        if isinstance(metadata, dict) and "KCCCallID" in metadata and metadata["KCCCallID"]:
            return f"kcc_{metadata['KCCCallID']}"

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
    skipped_lines = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for current_line, line in enumerate(f, start=1):
            if current_line <= start_line:
                continue

            line_str = line.strip()
            if not line_str:
                continue

            try:
                record = json.loads(line_str)
            except json.JSONDecodeError:
                skipped_lines += 1
                continue

            # Extract text (supporting 'text_chunk', 'text', 'content', 'document', 'query')
            text = (
                record.get("text")
                or record.get("text_chunk")
                or record.get("content")
                or record.get("document")
                or record.get("query")
            )
            
            if not text or not isinstance(text, str):
                skipped_lines += 1
                continue

            # Extract or generate ID
            chunk_id = record.get("id")
            if not chunk_id:
                chunk_id = generate_deterministic_id(file_path, current_line, text, record)

            # Metadata preservation and augmentation
            metadata = record.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {"raw_metadata": str(metadata)}

            metadata["source_file"] = rel_file
            if "kcc" in rel_file.lower() or collection_name == "kcc_docs":
                metadata.setdefault("source_type", "KCC")
            else:
                metadata.setdefault("source_type", "BOOK")

            item_collection = get_target_collection(file_path, metadata)

            yield {
                "id": str(chunk_id),
                "text": text,
                "metadata": metadata,
                "collection_name": item_collection,
                "line_number": current_line
            }

    if skipped_lines > 0:
        print(f"[INFO] Skipped {skipped_lines} lines in {rel_file} (empty or missing text field).")
