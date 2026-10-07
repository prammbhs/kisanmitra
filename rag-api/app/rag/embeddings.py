"""Voyage embeddings as a LangChain Embeddings object.

Uses the voyageai SDK directly (same as the ingestion server) so the model,
dimensions and behaviour are guaranteed identical. Queries use
input_type="query"; documents use "document" (what ingestion stored).
"""
from typing import List

import voyageai
from langchain_core.embeddings import Embeddings

from app.config import VOYAGE_API_KEY, VOYAGE_MODEL_ID, EMBEDDING_DIMENSIONS


class VoyageQueryEmbeddings(Embeddings):
    def __init__(self) -> None:
        if not VOYAGE_API_KEY:
            raise RuntimeError("VOYAGE_API_KEY is not set")
        self.client = voyageai.Client(api_key=VOYAGE_API_KEY)
        self.model = VOYAGE_MODEL_ID
        self.dimensions = EMBEDDING_DIMENSIONS

    def _embed(self, texts: List[str], input_type: str) -> List[List[float]]:
        kwargs = {"texts": texts, "model": self.model, "input_type": input_type}
        # Only pass output_dimension if non-default, to stay compatible with ingestion call
        if self.dimensions != 1024:
            kwargs["output_dimension"] = self.dimensions
        return self.client.embed(**kwargs).embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts, "document")

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text], "query")[0]
