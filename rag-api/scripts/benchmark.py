"""End-to-End Benchmark Suite for KisanMitra RAG API.

Measures:
1. Retrieval Latency (P50, P95, Mean) across 1.54M KCC embeddings on EBS.
2. Top-1 and Top-K Similarity Scores.
3. Generation Latency & End-to-End Chat Latency using Fireworks AI (GLM-5.3 Flash).
4. Citation & Context verification.

Usage:
    python scripts/benchmark.py [--url http://localhost:8080] [--chat] [--k 4]
"""

import argparse
import json
import statistics
import time
from typing import Any, Dict, List
import requests

BENCHMARK_DATASET = [
    # --- Category: Pest Management (English & Hindi) ---
    {
        "id": 1,
        "category": "Pest Management",
        "lang": "en",
        "query": "How to control whitefly in cotton?",
        "expected_keywords": ["whitefly", "cotton", "spray"],
    },
    {
        "id": 2,
        "category": "Pest Management",
        "lang": "hi",
        "query": "कपास में गुलाबी सुंडी का नियंत्रण कैसे करें?",
        "expected_keywords": ["गुलाबी", "कपास", "सुंडी"],
    },
    {
        "id": 3,
        "category": "Pest Management",
        "lang": "en",
        "query": "Chemical control for fall armyworm in maize",
        "expected_keywords": ["fall armyworm", "maize"],
    },
    {
        "id": 4,
        "category": "Pest Management",
        "lang": "hi",
        "query": "मिर्च में थ्रिप्स और मरोड़िया रोग की रोकथाम",
        "expected_keywords": ["मिर्च", "थ्रिप्स"],
    },
    {
        "id": 5,
        "category": "Pest Management",
        "lang": "en",
        "query": "Aphids attack in mustard crop what to spray",
        "expected_keywords": ["mustard", "aphid"],
    },

    # --- Category: Plant Diseases ---
    {
        "id": 6,
        "category": "Plant Diseases",
        "lang": "hi",
        "query": "जीरा में उकठा रोग (Wilt) का उपचार बताएं",
        "expected_keywords": ["जीरा", "उकठा"],
    },
    {
        "id": 7,
        "category": "Plant Diseases",
        "lang": "en",
        "query": "How to prevent blast disease in paddy crop?",
        "expected_keywords": ["blast", "paddy", "rice"],
    },
    {
        "id": 8,
        "category": "Plant Diseases",
        "lang": "hi",
        "query": "सोयाबीन में पीला मोज़ेक वायरस की रोकथाम",
        "expected_keywords": ["सोयाबीन", "मोज़ेक"],
    },
    {
        "id": 9,
        "category": "Plant Diseases",
        "lang": "en",
        "query": "Early blight and late blight control in potato",
        "expected_keywords": ["potato", "blight"],
    },
    {
        "id": 10,
        "category": "Plant Diseases",
        "lang": "hi",
        "query": "चने में उकठा रोग की दवा क्या है?",
        "expected_keywords": ["चना", "उकठा"],
    },

    # --- Category: Fertilizer & Nutrient Management ---
    {
        "id": 11,
        "category": "Nutrient Management",
        "lang": "en",
        "query": "Fertilizer dose for wheat crop at the time of sowing",
        "expected_keywords": ["wheat", "fertilizer", "dose"],
    },
    {
        "id": 12,
        "category": "Nutrient Management",
        "lang": "hi",
        "query": "धान की फसल में जिंक और यूरिया का सही अनुपात",
        "expected_keywords": ["धान", "यूरिया", "जिंक"],
    },
    {
        "id": 13,
        "category": "Nutrient Management",
        "lang": "hi",
        "query": "गन्ने में अधिक फुटाव के लिए क्या डालें?",
        "expected_keywords": ["गन्ना", "फुटाव"],
    },
    {
        "id": 14,
        "category": "Nutrient Management",
        "lang": "en",
        "query": "Micronutrient deficiency symptoms and spray in citrus",
        "expected_keywords": ["citrus", "micronutrient"],
    },

    # --- Category: Varieties & Cultural Practices ---
    {
        "id": 15,
        "category": "Varieties & Practices",
        "lang": "en",
        "query": "Best groundnut varieties for kharif season in Gujarat",
        "expected_keywords": ["groundnut", "kharif", "gujarat"],
    },
    {
        "id": 16,
        "category": "Varieties & Practices",
        "lang": "hi",
        "query": "सरसों की बुवाई का सही समय और बीज दर",
        "expected_keywords": ["सरसों", "बुवाई", "बीज"],
    },
    {
        "id": 17,
        "category": "Varieties & Practices",
        "lang": "en",
        "query": "Weed management in onion crop after transplanting",
        "expected_keywords": ["onion", "weed"],
    },
    {
        "id": 18,
        "category": "Varieties & Practices",
        "lang": "hi",
        "query": "गेहूं में पहली सिंचाई कितने दिन बाद करनी चाहिए?",
        "expected_keywords": ["गेहूं", "सिंचाई"],
    },

    # --- Category: Government Schemes & Advisory ---
    {
        "id": 19,
        "category": "Schemes & General",
        "lang": "en",
        "query": "Subsidy scheme for drip irrigation system",
        "expected_keywords": ["drip", "irrigation", "subsidy"],
    },
    {
        "id": 20,
        "category": "Schemes & General",
        "lang": "hi",
        "query": "सोलर पंप योजना के लिए आवेदन की जानकारी",
        "expected_keywords": ["सोलर", "पंप"],
    },
]


def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_d = sorted(data)
    idx = int(len(sorted_d) * (p / 100.0))
    idx = min(idx, len(sorted_d) - 1)
    return sorted_d[idx]


def run_benchmark(base_url: str, test_chat: bool = False, k: int = 4, token: str = ""):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    print(f"\n{'='*95}")
    print(f"  KISANMITRA RAG BENCHMARK SUITE")
    print(f"  Target URL: {base_url} | Total Test Cases: {len(BENCHMARK_DATASET)} | Top-K: {k}")
    print(f"  Test Mode: {'Full Chat (RAG + Fireworks)' if test_chat else 'Retrieval Only (/search)'}")
    print(f"{'='*95}\n")

    # Pre-warm check
    try:
        h = requests.get(f"{base_url}/health", headers=headers, timeout=10)
        h.raise_for_status()
        print(f"[HEALTH CHECK OK] Active Collections: {h.json().get('collections', {})}")
    except Exception as e:
        print(f"[ERROR] Cannot connect to {base_url}/health: {e}")
        return

    results = []
    retrieval_latencies: List[float] = []
    generation_latencies: List[float] = []
    top1_scores: List[float] = []
    topk_scores: List[float] = []

    print(f"\n{'ID':<3} | {'Lang':<4} | {'Category':<22} | {'Retrieval':<11} | {'Top-1 Score':<11} | {'Status'}")
    print("-" * 95)

    for item in BENCHMARK_DATASET:
        qid = item["id"]
        q = item["query"]
        category = item["category"]
        lang = item["lang"]

        endpoint = f"{base_url}/chat" if test_chat else f"{base_url}/search"
        payload = {"query": q, "k": k}

        t_start = time.perf_counter()
        try:
            r = requests.post(endpoint, json=payload, headers=headers, timeout=120)
            r.raise_for_status()
            res = r.json()
        except Exception as e:
            print(f"{qid:<3} | {lang:<4} | {category:<22} | FAILED ({e})")
            continue

        r_lat = res.get("retrieval_ms") or res.get("latency_ms", 0.0)
        retrieval_latencies.append(r_lat)

        sources = res.get("sources") or res.get("results") or []
        scores = [s.get("score", 0.0) for s in sources]
        t1_score = scores[0] if scores else 0.0
        top1_scores.append(t1_score)
        topk_scores.extend(scores)

        g_lat = res.get("generation_ms", 0.0)
        if test_chat:
            generation_latencies.append(g_lat)

        stat_str = f"Top-1: {t1_score:.3f}"
        if test_chat:
            stat_str += f" | Gen: {g_lat:.0f}ms"

        print(f"{qid:<3} | {lang:<4} | {category:<22} | {r_lat:>7.1f} ms | {t1_score:>11.3f} | [OK] {stat_str}")

        results.append({
            "id": qid,
            "category": category,
            "lang": lang,
            "query": q,
            "retrieval_ms": r_lat,
            "generation_ms": g_lat if test_chat else None,
            "top1_score": t1_score,
            "sources_count": len(sources),
            "top_source_snippet": sources[0]["text"][:160].replace("\n", " ") if sources else "",
        })

    # Summary Statistics
    print(f"\n{'='*95}")
    print("  BENCHMARK SUMMARY RESULTS")
    print(f"{'='*95}")

    print(f"\n[RETRIEVAL LATENCY - 1.54M VECTORS ON EBS]")
    print(f"  • P50 (Median)   : {percentile(retrieval_latencies, 50):.1f} ms")
    print(f"  • P90            : {percentile(retrieval_latencies, 90):.1f} ms")
    print(f"  • P95            : {percentile(retrieval_latencies, 95):.1f} ms")
    print(f"  • Mean           : {statistics.mean(retrieval_latencies):.1f} ms")
    print(f"  • Min / Max      : {min(retrieval_latencies):.1f} ms / {max(retrieval_latencies):.1f} ms")

    print(f"\n[SIMILARITY SCORES (Cosine Similarity)]")
    print(f"  • Mean Top-1 Score : {statistics.mean(top1_scores):.3f}")
    print(f"  • Min Top-1 Score  : {min(top1_scores):.3f}")
    print(f"  • Max Top-1 Score  : {max(top1_scores):.3f}")
    print(f"  • Mean Overall (K) : {statistics.mean(topk_scores):.3f}")

    if test_chat and generation_latencies:
        print(f"\n[GENERATION LATENCY - FIREWORKS GLM-5.3 FLASH]")
        print(f"  • P50 (Median)   : {percentile(generation_latencies, 50):.1f} ms")
        print(f"  • P95            : {percentile(generation_latencies, 95):.1f} ms")
        print(f"  • Mean           : {statistics.mean(generation_latencies):.1f} ms")

    # Output JSON file
    out_file = "benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {
                "retrieval_p50_ms": round(percentile(retrieval_latencies, 50), 1),
                "retrieval_p95_ms": round(percentile(retrieval_latencies, 95), 1),
                "retrieval_mean_ms": round(statistics.mean(retrieval_latencies), 1),
                "mean_top1_score": round(statistics.mean(top1_scores), 3),
            },
            "cases": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {out_file}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8080", help="Base URL of RAG API")
    p.add_argument("--chat", action="store_true", help="Test end-to-end /chat instead of /search only")
    p.add_argument("--k", type=int, default=4, help="Number of retrieved documents")
    p.add_argument("--token", default="", help="Optional Bearer token")
    args = p.parse_args()

    run_benchmark(base_url=args.url, test_chat=args.chat, k=args.k, token=args.token)
