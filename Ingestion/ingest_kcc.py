import os
import csv
import json
import time

# Increase CSV field size limit for large text fields
csv.field_size_limit(100_000_000)

DEFAULT_INPUT_CSV = r"d:\kisanmitra\cleanKccData\clean_kcc_data.csv"
DEFAULT_OUTPUT_DIR = r"d:\kisanmitra\Ingestion\kcc"
BATCH_SIZE = 50_000

def stream_kcc_records(csv_path):
    """
    Lazy generator reading KCC CSV records row by row to prevent high memory usage.
    """
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = (row.get("QueryText") or "").strip()
            answer = (row.get("KccAns") or "").strip()
            
            # Format text_chunk carrying ONLY Farmer Query and Expert Answer
            text_chunk = f"Farmer Query: {query}\nExpert Answer: {answer}"
            
            # Store all other fields as metadata
            metadata = {
                "KCCCallID": row.get("KCCCallID") or "",
                "StateName": row.get("StateName") or "",
                "DistrictName": row.get("DistrictName") or "",
                "BlockName": row.get("BlockName") or "",
                "Sector": row.get("Sector") or "",
                "Category": row.get("Category") or "",
                "Crop": row.get("Crop") or "",
                "Season": row.get("Season") or "",
                "QueryType": row.get("QueryType") or "",
                "CreatedOn": row.get("CreatedOn") or ""
            }
            
            yield {
                "text_chunk": text_chunk,
                "metadata": metadata
            }

def process_kcc_ingestion(input_csv=DEFAULT_INPUT_CSV, output_dir=DEFAULT_OUTPUT_DIR, batch_size=BATCH_SIZE, max_records=None):
    """
    Ingests KCC CSV and exports lazy-loaded batch JSONL files.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"=== Starting KCC Ingestion ===")
    print(f"Input: {input_csv}")
    print(f"Output Directory: {output_dir}")
    print(f"Batch Size: {batch_size:,} records per JSONL file")
    
    start_time = time.time()
    total_records = 0
    batch_index = 1
    current_batch_count = 0
    
    current_file_path = os.path.join(output_dir, f"kcc_batch_{batch_index:04d}.jsonl")
    current_file = open(current_file_path, "w", encoding="utf-8")
    
    try:
        for record in stream_kcc_records(input_csv):
            current_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_records += 1
            current_batch_count += 1
            
            if max_records and total_records >= max_records:
                print(f"Reached max_records limit ({max_records:,})")
                break
                
            if current_batch_count >= batch_size:
                current_file.close()
                print(f"Saved {current_file_path} ({current_batch_count:,} records)")
                
                batch_index += 1
                current_batch_count = 0
                current_file_path = os.path.join(output_dir, f"kcc_batch_{batch_index:04d}.jsonl")
                current_file = open(current_file_path, "w", encoding="utf-8")
                
    finally:
        if not current_file.closed:
            current_file.close()
            if current_batch_count > 0:
                print(f"Saved {current_file_path} ({current_batch_count:,} records)")
            else:
                # Remove empty batch file if any
                if os.path.exists(current_file_path):
                    os.remove(current_file_path)

    elapsed = time.time() - start_time
    print(f"=== KCC Ingestion Finished ===")
    print(f"Total records processed: {total_records:,}")
    print(f"Total batch files created: {batch_index if current_batch_count > 0 else batch_index - 1}")
    print(f"Time taken: {elapsed:.2f} seconds\n")

if __name__ == "__main__":
    process_kcc_ingestion()
