import time
from typing import List
import torch
from sentence_transformers import SentenceTransformer
from server.config import MODEL_NAME, DEVICE, GPU_BATCH_SIZE

class EmbeddingModelManager:
    _instance = None

    def __init__(self):
        print(f"[INIT] Loading embedding model '{MODEL_NAME}' on device '{DEVICE}'...")
        start_t = time.time()
        self.device = DEVICE
        self.gpu_batch_size = GPU_BATCH_SIZE
        
        # SentenceTransformer loads BAAI/bge-m3
        self.model = SentenceTransformer(MODEL_NAME, device=self.device)
        elapsed = time.time() - start_t
        print(f"[INIT] Model successfully loaded in {elapsed:.2f}s on {self.device}.")

    @classmethod
    def get_instance(cls) -> "EmbeddingModelManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Encode a list of text strings into 1024-dim embeddings using BGE-M3.
        """
        if not texts:
            return []

        # Sanitize text input to valid utf-8 strings
        clean_texts = [
            t if isinstance(t, str) else str(t)
            for t in texts
        ]

        embeddings = self.model.encode(
            clean_texts,
            batch_size=self.gpu_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        dim = embeddings.shape[1] if len(embeddings.shape) > 1 else 0
        if dim != 1024:
            raise ValueError(f"Expected embedding dimension 1024 from BGE-M3, but got {dim}")

        return embeddings.tolist()
