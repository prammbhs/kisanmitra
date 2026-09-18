import os
import chromadb
from typing import List, Dict, Any
from server.config import CHROMA_PATH, ALLOWED_COLLECTIONS, DEFAULT_COLLECTION

class ChromaDBManager:
    _instance = None

    def __init__(self, chroma_path: str = CHROMA_PATH):
        self.chroma_path = os.path.abspath(chroma_path)
        os.makedirs(self.chroma_path, exist_ok=True)
        print(f"[INIT] Initializing ChromaDB PersistentClient at path '{self.chroma_path}'...")
        
        self.client = chromadb.PersistentClient(path=self.chroma_path)
        self.collections: Dict[str, Any] = {}

        # Pre-initialize allowed collections (kcc_docs and other_docs)
        for col_name in ALLOWED_COLLECTIONS:
            self.collections[col_name] = self.client.get_or_create_collection(
                name=col_name,
                metadata={"hnsw:space": "cosine"}
            )
        print(f"[INIT] ChromaDB ready with collections: {list(self.collections.keys())}")

    @classmethod
    def get_instance(cls) -> "ChromaDBManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _get_collection(self, collection_name: str):
        target = collection_name if collection_name in ALLOWED_COLLECTIONS else DEFAULT_COLLECTION
        if target not in self.collections:
            self.collections[target] = self.client.get_or_create_collection(
                name=target,
                metadata={"hnsw:space": "cosine"}
            )
        return self.collections[target]

    def upsert_batch(
        self,
        collection_name: str,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> int:
        """
        Upsert a batch of items into ChromaDB persistent storage on EBS.
        """
        if not ids:
            return 0

        col = self._get_collection(collection_name)

        # Sanitize metadata values for Chroma compatibility (prune None, convert lists to strings if needed)
        sanitized_metadatas = []
        for m in metadatas:
            clean_m = {}
            for k, v in m.items():
                if v is None:
                    continue
                if isinstance(v, (str, int, float, bool)):
                    clean_m[k] = v
                else:
                    clean_m[k] = str(v)
            sanitized_metadatas.append(clean_m)

        col.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=sanitized_metadatas,
        )
        return len(ids)

    def get_collection_count(self, collection_name: str) -> int:
        col = self._get_collection(collection_name)
        return col.count()
