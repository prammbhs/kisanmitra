"""Read-only access to the existing Chroma collections on the EBS volume."""
import os
from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import CHROMA_PATH, COLLECTIONS
from app.rag.embeddings import VoyageQueryEmbeddings


class VectorStoreManager:
    def __init__(self) -> None:
        if not os.path.isdir(CHROMA_PATH):
            raise RuntimeError(f"CHROMA_PATH '{CHROMA_PATH}' not found — is the EBS volume mounted?")
        self.client = chromadb.PersistentClient(
            path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False)
        )
        self.embeddings = VoyageQueryEmbeddings()
        existing = {c.name if hasattr(c, "name") else c for c in self.client.list_collections()}
        self.stores: Dict[str, Chroma] = {}
        for name in COLLECTIONS:
            if name not in existing:
                print(f"[WARN] Collection '{name}' not found in {CHROMA_PATH}; skipping")
                continue
            self.stores[name] = Chroma(
                client=self.client,
                collection_name=name,
                embedding_function=self.embeddings,
            )
            print(f"[INIT] Loaded collection '{name}' ({self.count(name)} docs)")

    def count(self, name: str) -> int:
        return self.client.get_collection(name).count()

    def search(
        self,
        query: str,
        collections: List[str],
        k: int,
        where: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        """Embed query once, search each collection, merge by similarity (higher = better)."""
        qvec = self.embeddings.embed_query(query)
        results: List[Tuple[Document, float]] = []
        for name in collections:
            store = self.stores.get(name)
            if not store:
                continue
            hits = store.similarity_search_by_vector_with_relevance_scores(
                qvec, k=k, filter=where or None
            )
            for doc, distance in hits:
                doc.metadata["collection"] = name
                # cosine distance -> similarity
                results.append((doc, 1.0 - float(distance)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]


def build_where(filters: Optional[Dict[str, str]]) -> Optional[dict]:
    """Convert {"StateName": "GUJARAT", "Crop": "Wheat"} to a Chroma where clause."""
    if not filters:
        return None
    clauses = [{k: v} for k, v in filters.items() if v]
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}
