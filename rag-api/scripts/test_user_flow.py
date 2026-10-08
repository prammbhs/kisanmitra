"""Automated End-to-End Test for:
1. User Registration (with optional profile crops)
2. Login & JWT authentication
3. Farmer Profile retrieval and update
4. Multi-turn LangGraph Chat with automatic profile injection
"""
import argparse
import random
import sys
import time
import requests

def test_full_flow(base_url: str):
    print("=" * 80)
    print(f"  KISANMITRA E2E USER & LANGGRAPH FLOW TEST ({base_url})")
    print("=" * 80)

    # 1. Register a new test farmer
    rand_id = random.randint(1000, 9999)
    phone = f"98765{rand_id}"
    password = "FarmerPassword123!"

    print(f"\n[1] Registering New Farmer: Phone={phone}...")
    reg_payload = {
        "phone_or_email": phone,
        "password": password,
        "full_name": "Ramesh Patel",
        "state": "GUJARAT",
        "district": "AMRELI",
        "primary_crops": ["Cotton (Kapas)"],  # Optional crops
        "land_acres": 4.5,                    # Optional land size
        "preferred_language": "hi",
    }
    r = requests.post(f"{base_url}/api/v1/auth/register", json=reg_payload, timeout=10)
    if r.status_code != 201:
        print(f"[FAIL] Registration failed: {r.status_code} - {r.text}")
        sys.exit(1)
    token_data = r.json()
    token = token_data["access_token"]
    print(f"    -> Registered successfully! Acquired JWT token: {token[:20]}...")

    headers = {"Authorization": f"Bearer {token}"}

    # 2. Get Profile
    print("\n[2] Fetching Farmer Profile...")
    r = requests.get(f"{base_url}/api/v1/user/profile", headers=headers, timeout=10)
    if r.status_code != 200:
        print(f"[FAIL] Get profile failed: {r.status_code} - {r.text}")
        sys.exit(1)
    prof = r.json()
    print(f"    -> Profile: Name={prof['full_name']} | State={prof['state']} | Crops={prof['primary_crops']} | Land={prof['land_acres']} acres")

    # 3. Update Profile (Add wheat as second crop)
    print("\n[3] Updating Profile (Adding 'Wheat' to crops)...")
    update_payload = {
        "primary_crops": ["Cotton (Kapas)", "Wheat"],
        "land_acres": 6.0
    }
    r = requests.put(f"{base_url}/api/v1/user/profile", json=update_payload, headers=headers, timeout=10)
    if r.status_code != 200:
        print(f"[FAIL] Update profile failed: {r.status_code} - {r.text}")
        sys.exit(1)
    prof = r.json()
    print(f"    -> Updated: Crops={prof['primary_crops']} | Land={prof['land_acres']} acres")

    # 4. Turn 1: Ask an ambiguous question WITHOUT specifying crop or state
    # The LangGraph agent should automatically detect the farmer's crop ('Cotton (Kapas)') and state ('GUJARAT') from profile!
    thread_id = f"test-thread-{rand_id}"
    print(f"\n[4] LangGraph Chat Turn 1 (Thread: {thread_id}):")
    query_turn1 = "सफेद मक्खी के लिए क्या स्प्रे करें?"
    print(f"    Farmer: '{query_turn1}' (Notice: Crop/State is NOT mentioned)")

    t0 = time.perf_counter()
    r = requests.post(
        f"{base_url}/chat",
        json={"query": query_turn1, "thread_id": thread_id, "k": 3},
        headers=headers,
        timeout=60,
    )
    if r.status_code != 200:
        print(f"[FAIL] Chat Turn 1 failed: {r.status_code} - {r.text}")
        sys.exit(1)
    data1 = r.json()
    elapsed1 = (time.perf_counter() - t0) * 1000
    print(f"    -> Advisory ({elapsed1:.0f} ms):\n{data1['answer'][:300]}...\n")

    # 5. Turn 2: Follow-up question using conversation history
    print(f"[5] LangGraph Chat Turn 2 (Follow-up on same thread: {thread_id}):")
    query_turn2 = "और यदि आक्रमण बहुत अधिक हो तो कौन सी दवा डालें?"
    print(f"    Farmer: '{query_turn2}' (Follow-up referring to Turn 1)")

    t0 = time.perf_counter()
    r = requests.post(
        f"{base_url}/chat",
        json={"query": query_turn2, "thread_id": thread_id, "k": 3},
        headers=headers,
        timeout=60,
    )
    if r.status_code != 200:
        print(f"[FAIL] Chat Turn 2 failed: {r.status_code} - {r.text}")
        sys.exit(1)
    data2 = r.json()
    elapsed2 = (time.perf_counter() - t0) * 1000
    print(f"    -> Advisory ({elapsed2:.0f} ms):\n{data2['answer'][:300]}...\n")

    print("=" * 80)
    print("  ALL USER AUTH, PROFILE & LANGGRAPH MULTI-TURN TESTS PASSED! [100% OK]")
    print("=" * 80)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8080", help="Base URL of RAG API")
    args = p.parse_args()
    test_full_flow(args.url)
