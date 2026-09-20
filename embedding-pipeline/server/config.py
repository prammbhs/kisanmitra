import os
from dotenv import load_dotenv

load_dotenv()

# Server setup
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Voyage AI & Model settings
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")
VOYAGE_MODEL_ID = os.getenv("VOYAGE_MODEL_ID", "voyage-4-lite")
VOYAGE_INPUT_TYPE = os.getenv("VOYAGE_INPUT_TYPE", "document")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))
MAX_TOKEN_LIMIT = int(os.getenv("MAX_TOKEN_LIMIT", "200000000"))  # Default 200M tokens


# Storage Settings (EBS ChromaDB)
CHROMA_PATH = os.getenv("CHROMA_PATH", "/mnt/chroma")
ALLOWED_COLLECTIONS = ["kcc_docs", "other_docs"]
DEFAULT_COLLECTION = os.getenv("DEFAULT_COLLECTION", "kcc_docs")

# Optional Auth
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")

