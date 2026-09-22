import os
import sys
import sqlite3
import threading
import faulthandler
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any

faulthandler.enable()

try:
    from server.config import CHROMA_PATH, ALLOWED_COLLECTIONS, DEFAULT_COLLECTION
except ImportError:
    from config import CHROMA_PATH, ALLOWED_COLLECTIONS, DEFAULT_COLLECTION

class ChromaDBManager:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, chroma_path: str = CHROMA_PATH):
        self.chroma_path = os.path.abspath(chroma_path)
        self.db_write_lock = threading.Lock()
        print(f"[DIAGNOSTIC] Python Version: {sys.version}")
        print(f"[DIAGNOSTIC] SQLite3 Version: {sqlite3.sqlite_version}")
        print(f"[DIAGNOSTIC] Target Chroma Path: '{self.chroma_path}'")

        # Step 1: Pre-flight Directory & Write Permission Check
        try:
            os.makedirs(self.chroma_path, exist_ok=True)
            test_file = os.path.join(self.chroma_path, ".perm_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            print(f"[DIAGNOSTIC] Write test to '{self.chroma_path}' SUCCEEDED.")
        except Exception as e:
            print(f"[ERROR] Write test to '{self.chroma_path}' FAILED: {e}")

        print(f"[INIT] Initializing ChromaDB PersistentClient at path '{self.chroma_path}'...")

        # Step 2: Initialize PersistentClient on EBS volume at startup (telemetry disabled)
        try:
            print("[DIAGNOSTIC] Calling chromadb.PersistentClient()...")
            self.client = chromadb.PersistentClient(
                path=self.chroma_path,
                settings=Settings(anonymized_telemetry=False)
            )
            print("[DIAGNOSTIC] chromadb.PersistentClient() initialized successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to initialize ChromaDB PersistentClient: {e}")
            raise e

        self.collections: Dict[str, Any] = {}

        # Pre-initialize target collections (kcc_docs and other_docs)
        for col_name in ALLOWED_COLLECTIONS:
            self.collections[col_name] = self.client.get_or_create_collection(
                name=col_name,
                metadata={"hnsw:space": "cosine"}
            )
            try:
                cnt = self.collections[col_name].count()
                print(f"[INIT] ChromaDB collection '{col_name}' ready (count: {cnt})")
            except Exception as e:
                print(f"[WARN] Could not retrieve count for collection '{col_name}': {e}")

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

        # Sanitize metadata values for Chroma compatibility
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

        with self.db_write_lock:
            col.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=sanitized_metadatas,
            )
        return len(ids)

    def get_collection_count(self, collection_name: str) -> int:
        try:
            col = self._get_collection(collection_name)
            return col.count()
        except Exception as e:
            print(f"[WARN] Error getting count for {collection_name}: {e}")
            return 0
