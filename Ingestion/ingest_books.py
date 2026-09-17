import os
import glob
import json
import time
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CONTENTS_DIR = r"d:\kisanmitra\contents"
DEFAULT_OUTPUT_DIR = r"d:\kisanmitra\Ingestion\books"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
BATCH_SIZE = 5000  # Chunks per JSONL batch file

def process_books_ingestion(contents_dir=DEFAULT_CONTENTS_DIR, output_dir=DEFAULT_OUTPUT_DIR,
                           chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, batch_size=BATCH_SIZE):
    """
    Recursively scans and lazy-loads all PDF and text documents under contents_dir,
    splits them using RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200),
    and exports formatted JSONL batch files.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"=== Starting Books & Contents Ingestion ===")
    print(f"Contents Directory: {contents_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Chunk Size: {chunk_size}, Chunk Overlap: {chunk_overlap}")
    print(f"Batch Size: {batch_size:,} chunks per JSONL file")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    start_time = time.time()
    total_chunks = 0
    total_files_processed = 0
    batch_index = 1
    current_batch_count = 0

    current_file_path = os.path.join(output_dir, f"books_batch_{batch_index:04d}.jsonl")
    current_file = open(current_file_path, "w", encoding="utf-8")

    # Find all PDF files recursively
    pdf_files = glob.glob(os.path.join(contents_dir, "**", "*.pdf"), recursive=True)
    print(f"Found {len(pdf_files)} PDF files to process under {contents_dir}\n")

    try:
        for pdf_path in pdf_files:
            rel_path = os.path.relpath(pdf_path, contents_dir)
            file_name = os.path.basename(pdf_path)
            total_files_processed += 1
            print(f"[{total_files_processed}/{len(pdf_files)}] Processing: {rel_path}...")

            try:
                loader = PyPDFLoader(pdf_path)
                # Lazy load pages one by one
                pages = list(loader.lazy_load())
                
                # Split pages into chunks
                file_chunk_index = 0
                for page_doc in pages:
                    page_num = page_doc.metadata.get("page", 0) + 1  # 1-indexed page
                    chunks = splitter.split_text(page_doc.page_content)

                    for chunk_text in chunks:
                        if not chunk_text.strip():
                            continue
                        
                        file_chunk_index += 1
                        record = {
                            "text_chunk": chunk_text.strip(),
                            "metadata": {
                                "source": file_name,
                                "file_path": rel_path.replace("\\", "/"),
                                "page": page_num,
                                "chunk_id": file_chunk_index
                            }
                        }

                        current_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        total_chunks += 1
                        current_batch_count += 1

                        if current_batch_count >= batch_size:
                            current_file.close()
                            print(f"  -> Saved batch file {current_file_path} ({current_batch_count:,} chunks)")

                            batch_index += 1
                            current_batch_count = 0
                            current_file_path = os.path.join(output_dir, f"books_batch_{batch_index:04d}.jsonl")
                            current_file = open(current_file_path, "w", encoding="utf-8")

                print(f"  Finished {rel_path}: generated {file_chunk_index} chunks")

            except Exception as e:
                print(f"  [ERROR] Failed to process {rel_path}: {e}")

    finally:
        if not current_file.closed:
            current_file.close()
            if current_batch_count > 0:
                print(f"  -> Saved final batch file {current_file_path} ({current_batch_count:,} chunks)")
            else:
                if os.path.exists(current_file_path):
                    os.remove(current_file_path)

    elapsed = time.time() - start_time
    print(f"\n=== Books & Contents Ingestion Finished ===")
    print(f"Total PDF files processed: {total_files_processed}")
    print(f"Total text chunks created: {total_chunks:,}")
    print(f"Total batch files created: {batch_index if current_batch_count > 0 else batch_index - 1}")
    print(f"Time taken: {elapsed:.2f} seconds\n")

if __name__ == "__main__":
    process_books_ingestion()
