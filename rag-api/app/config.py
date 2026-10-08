import os
from dotenv import load_dotenv

load_dotenv()

# --- Server ---
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))
RAG_API_KEY = os.getenv("RAG_API_KEY", "")  # empty = auth disabled

# --- Chroma (EBS mount) ---
CHROMA_PATH = os.getenv("CHROMA_PATH", "/mnt/chroma")
COLLECTIONS = ["kcc_docs"]
DEFAULT_COLLECTIONS = os.getenv("DEFAULT_COLLECTIONS", "kcc_docs").split(",")

# --- Embeddings: MUST match ingestion (voyage-4-lite, 1024d) ---
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")
VOYAGE_MODEL_ID = os.getenv("VOYAGE_MODEL_ID", "voyage-4-lite")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))

# --- LLM: Fireworks ---
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")
FIREWORKS_MODEL = os.getenv("FIREWORKS_MODEL", "accounts/fireworks/models/glm-5p3-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# --- Retrieval ---
TOP_K = int(os.getenv("TOP_K", "6"))
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.0"))  # cosine similarity floor (0..1)
