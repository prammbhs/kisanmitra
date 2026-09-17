import argparse
import sys
import os

# Ensure current package path is in sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from ingest_kcc import process_kcc_ingestion
from ingest_books import process_books_ingestion

def main():
    parser = argparse.ArgumentParser(description="KisanMitra Ingestion & Text Splitting Pipeline")
    parser.add_argument("--mode", choices=["all", "kcc", "books"], default="all",
                        help="Mode of ingestion: 'kcc', 'books', or 'all' (default: all)")
    parser.add_argument("--kcc-csv", default=r"d:\kisanmitra\cleanKccData\clean_kcc_data.csv",
                        help="Path to cleaned KCC CSV file")
    parser.add_argument("--contents-dir", default=r"d:\kisanmitra\contents",
                        help="Path to contents directory containing Books/PDFs")
    parser.add_argument("--kcc-batch-size", type=int, default=50000,
                        help="Records per JSONL batch for KCC data (default: 50000)")
    parser.add_argument("--books-batch-size", type=int, default=5000,
                        help="Chunks per JSONL batch for Books/PDFs (default: 5000)")
    parser.add_argument("--kcc-max-records", type=int, default=None,
                        help="Optional limit on number of KCC records to process for testing")
    parser.add_argument("--chunk-size", type=int, default=1200,
                        help="Chunk size for RecursiveCharacterTextSplitter (default: 1200)")
    parser.add_argument("--chunk-overlap", type=int, default=200,
                        help="Chunk overlap for RecursiveCharacterTextSplitter (default: 200)")
    parser.add_argument("--encoding", default="utf-8",
                        help="Encoding for reading/writing documents and JSONL batches (default: utf-8)")

    args = parser.parse_args()

    print("==========================================================")
    print("      KISANMITRA RAG INGESTION & TEXT SPLITTER PIPELINE   ")
    print("==========================================================\n")

    if args.mode in ["kcc", "all"]:
        process_kcc_ingestion(
            input_csv=args.kcc_csv,
            output_dir=r"d:\kisanmitra\Ingestion\kcc",
            batch_size=args.kcc_batch_size,
            max_records=args.kcc_max_records
        )

    if args.mode in ["books", "all"]:
        process_books_ingestion(
            contents_dir=args.contents_dir,
            output_dir=r"d:\kisanmitra\Ingestion\books",
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            batch_size=args.books_batch_size,
            encoding=args.encoding
        )

if __name__ == "__main__":
    main()
