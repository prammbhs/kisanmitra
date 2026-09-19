import json
import time
from typing import List
from concurrent.futures import ThreadPoolExecutor
import boto3
from botocore.exceptions import BotoCoreError, ClientError

from server.config import (
    AWS_REGION,
    BEDROCK_MODEL_ID,
    EMBEDDING_DIMENSIONS,
    NORMALIZE_EMBEDDINGS,
    BEDROCK_MAX_WORKERS,
)

class BedrockTitanEmbeddingManager:
    _instance = None

    def __init__(self):
        print(f"[INIT] Initializing AWS Bedrock runtime client in region '{AWS_REGION}'...")
        print(f"[INIT] Target Model: '{BEDROCK_MODEL_ID}' ({EMBEDDING_DIMENSIONS} dims)")
        start_t = time.time()
        
        self.region = AWS_REGION
        self.model_id = BEDROCK_MODEL_ID
        self.dimensions = EMBEDDING_DIMENSIONS
        self.normalize = NORMALIZE_EMBEDDINGS
        self.max_workers = BEDROCK_MAX_WORKERS

        # Initialize boto3 Bedrock Runtime Client
        self.client = boto3.client("bedrock-runtime", region_name=self.region)
        elapsed = time.time() - start_t
        print(f"[INIT] Bedrock client initialized successfully in {elapsed:.2f}s.")

    @classmethod
    def get_instance(cls) -> "BedrockTitanEmbeddingManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _embed_single_text(self, text: str) -> List[float]:
        """
        Invoke Bedrock Titan Text Embeddings V2 for a single text chunk.
        """
        # Ensure utf-8 text input
        clean_text = text if isinstance(text, str) else str(text)
        if not clean_text.strip():
            clean_text = " "

        native_request = {
            "inputText": clean_text,
            "dimensions": self.dimensions,
            "normalize": self.normalize,
        }

        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(native_request),
            )
            response_body = json.loads(response["body"].read().decode("utf-8"))
            embedding = response_body.get("embedding", [])
            
            if len(embedding) != self.dimensions:
                raise ValueError(
                    f"Expected embedding dimension {self.dimensions}, got {len(embedding)}"
                )
            return embedding

        except (BotoCoreError, ClientError) as e:
            print(f"[ERROR] AWS Bedrock API invocation failed: {e}")
            raise RuntimeError(f"Bedrock invocation error: {e}") from e

    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Concurrently encode a batch of texts using a ThreadPoolExecutor.
        Preserves original input ordering.
        """
        if not texts:
            return []

        # Execute concurrent calls via threadpool for batch speed
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            embeddings = list(executor.map(self._embed_single_text, texts))

        return embeddings
