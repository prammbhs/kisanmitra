import time
from typing import List
import voyageai

try:
    from server.config import (
        VOYAGE_API_KEY,
        VOYAGE_MODEL_ID,
        VOYAGE_INPUT_TYPE,
        EMBEDDING_DIMENSIONS,
    )
except ImportError:
    from config import (
        VOYAGE_API_KEY,
        VOYAGE_MODEL_ID,
        VOYAGE_INPUT_TYPE,
        EMBEDDING_DIMENSIONS,
    )

class VoyageEmbeddingManager:
    _instance = None

    def __init__(self):
        print(f"[INIT] Initializing Voyage AI client...")
        print(f"[INIT] Target Model: '{VOYAGE_MODEL_ID}' (input_type: '{VOYAGE_INPUT_TYPE}')")
        start_t = time.time()
        
        self.model_id = VOYAGE_MODEL_ID
        self.input_type = VOYAGE_INPUT_TYPE
        self.dimensions = EMBEDDING_DIMENSIONS

        self.client = voyageai.Client(api_key=VOYAGE_API_KEY if VOYAGE_API_KEY else None)
        elapsed = time.time() - start_t
        print(f"[INIT] Voyage AI client initialized successfully in {elapsed:.2f}s.")

    @classmethod
    def get_instance(cls) -> "VoyageEmbeddingManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Encode a list of texts using Voyage AI synchronous embedding API.
        """
        if not texts:
            return []

        clean_texts = [
            t if (isinstance(t, str) and t.strip()) else " "
            for t in texts
        ]

        try:
            result = self.client.embed(
                texts=clean_texts,
                model=self.model_id,
                input_type=self.input_type if self.input_type else None,
            )
            return result.embeddings
        except Exception as e:
            print(f"[ERROR] Voyage AI API invocation failed ({type(e).__name__}): {e}")
            raise RuntimeError(f"VoyageAI ({type(e).__name__}): {e}") from e

