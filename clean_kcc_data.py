import os
import glob
import re
import csv
import json
import sys

# Increase CSV field size limit to handle large text cells safely
csv.field_size_limit(100_000_000)

# Input / Output Paths
INPUT_DIR = r"d:\kisanmitra\KCCdata"
OUTPUT_DIR = r"d:\kisanmitra\cleanKccData"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "clean_kcc_data.csv")
OUTPUT_JSONL = os.path.join(OUTPUT_DIR, "clean_kcc_data.jsonl")
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "processing_summary.json")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Keyword patterns for filtering
TEST_CALL_PATTERNS = [
    r"\btest\s*call\b",
    r"\bblank\s*call\b",
    r"\btest\s*query\b",
    r"^test$",
    r"^blank$"
]

INCOMPLETE_PATTERNS = [
    r"incomplete\s*data",
    r"incomplete\s*call",
    r"data\s*incomplete",
    r"call\s*incomplete",
    r"details?\s*not\s*provided",
    r"no\s*query",
    r"wrong\s*number"
]

WEATHER_PATTERNS = [
    r"\bweather\b",
    r"\bforecast\b",
]

def infer_season(season_val, month_val):
    """
    Infers Indian agricultural season from month if missing, NA, or None.
    Rabi: Nov-Mar (11, 12, 1, 2, 3)
    Zaid: Apr-May (4, 5)
    Kharif: Jun-Oct (6, 7, 8, 9, 10)
    """
    season_str = str(season_val).strip() if season_val else ""
    if season_str and season_str.upper() not in ["NA", "N/A", "NONE", "", "NAN", "NULL", "OTHERS", "OTHER"]:
        return season_str

    try:
        m = int(month_val)
        if m in [11, 12, 1, 2, 3]:
            return "Rabi"
        elif m in [4, 5]:
            return "Zaid"
        elif m in [6, 7, 8, 9, 10]:
            return "Kharif"
    except (ValueError, TypeError):
        pass
    
    return None

def is_test_or_blank(text):
    if not text:
        return True
    text_lower = text.strip().lower()
    for pat in TEST_CALL_PATTERNS:
        if re.search(pat, text_lower):
            return True
    return False

def is_incomplete(query_text, ans_text):
    q_lower = str(query_text).strip().lower()
    a_lower = str(ans_text).strip().lower()
    for pat in INCOMPLETE_PATTERNS:
        if re.search(pat, q_lower) or re.search(pat, a_lower):
            return True
    return False

def is_weather_query(query_type, query_text):
    q_type_str = str(query_type).strip().lower() if query_type else ""
    if "weather" in q_type_str:
        return True
    
    q_text_str = str(query_text).strip().lower() if query_text else ""
    for pat in WEATHER_PATTERNS:
        if re.search(pat, q_text_str):
            return True
    return False

def clean_text_field(text, convert_other_to_null=False):
    if text is None:
        return None if convert_other_to_null else ""
    text_str = str(text).strip()
    # Normalize multiple whitespaces and newlines into clean single space
    text_str = re.sub(r"\s+", " ", text_str)
    if not text_str:
        return None if convert_other_to_null else ""
    
    if convert_other_to_null and text_str.lower() in ["others", "other", "others...", "other..."]:
        return None
    return text_str

def build_rag_chunk(row):
    """
    Constructs a rich, formatted text block for RAG vector embedding.
    """
    state = row.get("StateName") or "NA"
    district = row.get("DistrictName") or "NA"
    block = row.get("BlockName") or "NA"
    sector = row.get("Sector") or "NA"
    category = row.get("Category") or "NA"
    crop = row.get("Crop") or "NA"
    season = row.get("Season") or "NA"
    q_type = row.get("QueryType") or "NA"
    query = row.get("QueryText") or ""
    answer = row.get("KccAns") or ""

    chunk = (
        f"Location: State: {state}, District: {district}, Block: {block}\n"
        f"Domain: Sector: {sector} | Category: {category} | Crop: {crop} | Season: {season}\n"
        f"Query Category: {q_type}\n"
        f"Farmer Query: {query}\n"
        f"Expert Answer: {answer}"
    )
    return chunk

FIELDNAMES = [
    "KCCCallID", "StateName", "DistrictName", "BlockName",
    "Sector", "Category", "Crop", "Season", "QueryType",
    "QueryText", "KccAns", "CreatedOn", "text_chunk"
]

def process_all_files():
    csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    print(f"Found {len(csv_files)} CSV files in {INPUT_DIR}")

    stats = {
        "total_files": len(csv_files),
        "total_rows_read": 0,
        "dropped_test_blank": 0,
        "dropped_incomplete": 0,
        "dropped_weather": 0,
        "dropped_missing_or_short": 0,
        "dropped_duplicates": 0,
        "clean_rows_saved": 0
    }

    seen_signatures = set()

    jsonl_file = open(OUTPUT_JSONL, "w", encoding="utf-8")
    csv_file = open(OUTPUT_CSV, "w", encoding="utf-8", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
    csv_writer.writeheader()

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        print(f"Processing file: {file_name}...")

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f_in:
                reader = csv.DictReader(f_in)
                for row in reader:
                    stats["total_rows_read"] += 1
                    
                    q_text = clean_text_field(row.get("QueryText", ""))
                    ans_text = clean_text_field(row.get("KccAns", ""))
                    q_type = clean_text_field(row.get("QueryType", ""), convert_other_to_null=True)

                    # 1. Filter out missing or too short queries/answers
                    if not q_text or not ans_text or len(q_text) < 5 or len(ans_text) < 8:
                        stats["dropped_missing_or_short"] += 1
                        continue

                    # 2. Filter out Test / Blank calls
                    if is_test_or_blank(q_text) or is_test_or_blank(ans_text):
                        stats["dropped_test_blank"] += 1
                        continue

                    # 3. Filter out Incomplete data entries
                    if is_incomplete(q_text, ans_text):
                        stats["dropped_incomplete"] += 1
                        continue

                    # 4. Filter out Weather queries
                    if is_weather_query(q_type, q_text):
                        stats["dropped_weather"] += 1
                        continue

                    # 5. Deduplication check based on signature (State, Crop, Query, Answer)
                    state = clean_text_field(row.get("StateName", ""), convert_other_to_null=True)
                    crop = clean_text_field(row.get("Crop", ""), convert_other_to_null=True)
                    sig_state = state.lower() if state else ""
                    sig_crop = crop.lower() if crop else ""
                    sig = f"{sig_state}|{sig_crop}|{q_text.lower()}|{ans_text.lower()}"
                    if sig in seen_signatures:
                        stats["dropped_duplicates"] += 1
                        continue
                    seen_signatures.add(sig)

                    # 6. Infer Season based on month
                    month = row.get("month", None)
                    season = infer_season(row.get("Season", None), month)

                    # Build cleaned record
                    clean_row = {
                        "KCCCallID": clean_text_field(row.get("KCCCallID", "")),
                        "StateName": state,
                        "DistrictName": clean_text_field(row.get("DistrictName", ""), convert_other_to_null=True),
                        "BlockName": clean_text_field(row.get("BlockName", ""), convert_other_to_null=True),
                        "Sector": clean_text_field(row.get("Sector", ""), convert_other_to_null=True),
                        "Category": clean_text_field(row.get("Category", ""), convert_other_to_null=True),
                        "Crop": crop,
                        "Season": season,
                        "QueryType": q_type,
                        "QueryText": q_text,
                        "KccAns": ans_text,
                        "CreatedOn": clean_text_field(row.get("CreatedOn", ""))
                    }

                    # Construct RAG chunk
                    rag_chunk = build_rag_chunk(clean_row)
                    clean_row["text_chunk"] = rag_chunk

                    # Write outputs
                    jsonl_file.write(json.dumps(clean_row, ensure_ascii=False) + "\n")
                    csv_writer.writerow(clean_row)
                    stats["clean_rows_saved"] += 1

        except Exception as e:
            print(f"Error reading {file_name}: {e}")

        print(f"  Saved cumulative {stats['clean_rows_saved']:,} clean rows so far...")

    jsonl_file.close()
    csv_file.close()

    # Save summary report
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n================ PROCESSING SUMMARY ================")
    for k, v in stats.items():
        print(f"{k}: {v:,}" if isinstance(v, int) else f"{k}: {v}")
    print(f"Clean CSV output saved to: {OUTPUT_CSV}")
    print(f"Clean JSONL output saved to: {OUTPUT_JSONL}")

if __name__ == "__main__":
    process_all_files()
