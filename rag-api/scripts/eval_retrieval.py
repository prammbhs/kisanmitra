"""Retrieval quality check — run against /search to decide if a reranker is needed.

Usage:
  python scripts/eval_retrieval.py --url http://<EC2-IP>:8080 [--token XXX] [--k 10]

For each question, mark the rank of the first truly relevant hit by eye.
Rule of thumb: if the right answer is often in top-10 but NOT in top-3,
a reranker (e.g. Voyage rerank-2.5-lite) will help. If it's not in top-10
at all, fix chunking/filters/query first — a reranker won't help.
"""
import argparse
import requests

QUESTIONS = [
    "How to control whitefly in cotton?",
    "Fertilizer dose for wheat at sowing time",
    "Yellow leaves in paddy nursery what to do",
    "Fall armyworm in maize treatment",
    "Which variety of groundnut for kharif in Gujarat",
    "Thrips control in chilli",
    "जीरा में उकठा रोग का उपचार",
    "Drip irrigation subsidy scheme information",
    "Fruit borer management in tomato",
    "Weed control in soybean after sowing",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8080")
    p.add_argument("--token", default="")
    p.add_argument("--k", type=int, default=10)
    a = p.parse_args()
    headers = {"Authorization": f"Bearer {a.token}"} if a.token else {}

    for q in QUESTIONS:
        r = requests.post(f"{a.url}/search", json={"query": q, "k": a.k}, headers=headers, timeout=60)
        r.raise_for_status()
        data = r.json()
        print("=" * 100)
        print(f"Q: {q}   ({data['latency_ms']} ms)")
        for s in data["results"]:
            snippet = s["text"].replace("\n", " ")[:140]
            print(f"  {s['index']:>2}. [{s['score']:.3f}] ({s['collection']}) {snippet}")


if __name__ == "__main__":
    main()
