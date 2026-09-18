import time
import requests
from typing import Dict, Any, List
from client.config import (
    EMBEDDING_API_URL,
    EMBEDDING_API_KEY,
    MAX_RETRIES,
    INITIAL_BACKOFF,
    REQUEST_TIMEOUT,
)

class EmbeddingAPIClient:
    def __init__(
        self,
        api_url: str = EMBEDDING_API_URL,
        api_key: str = EMBEDDING_API_KEY,
        max_retries: int = MAX_RETRIES,
        initial_backoff: float = INITIAL_BACKOFF,
        timeout: float = REQUEST_TIMEOUT,
    ):
        self.api_url = api_url.rstrip("/")
        self.endpoint = f"{self.api_url}/embed"
        self.api_key = api_key
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.timeout = timeout

    def send_micro_batch(self, items: List[Dict[str, Any]], collection_name: str) -> Dict[str, Any]:
        """
        Send a micro-batch of items to the embedding server with retry support.
        """
        headers = {
            "Content-Type": "application/json; charset=utf-8"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "collection_name": collection_name,
            "items": [
                {
                    "id": item["id"],
                    "text": item["text"],
                    "metadata": item["metadata"],
                }
                for item in items
            ],
        }

        backoff = self.initial_backoff

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )

                # Client error (4xx except 429) -> don't retry blindly
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    raise RuntimeError(
                        f"Client Error HTTP {response.status_code}: {response.text}"
                    )

                response.raise_for_status()
                return response.json()

            except (requests.exceptions.RequestException, RuntimeError) as e:
                if attempt == self.max_retries:
                    print(f"[ERROR] Final retry attempt {attempt}/{self.max_retries} failed for batch: {e}")
                    raise

                print(
                    f"[WARN] Request attempt {attempt}/{self.max_retries} failed ({e}). "
                    f"Retrying in {backoff:.1f}s..."
                )
                time.sleep(backoff)
                backoff *= 2.0

        raise RuntimeError("Unexpected end of retry loop")
